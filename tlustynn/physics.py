import torch

from .data_loader import inverse_transform_output


class PhysicsConstraints:
    """物理约束模块 — 对数空间计算版（学习 kurucz-a1-main 经验）

    核心改造：
    1. 所有含空间导数的约束在对数空间计算，避免物理空间数值爆炸。
    2. tau 使用模型预测值作为深度坐标（有限差分近似 d/d(tau)）。
    3. 加入边界掩码，排除大气顶部/底部的极端区域。
    4. 引入 sign_loss 强制压强随光深单调增。
    """

    def __init__(self, device='cuda'):
        self.device = device

        self.k_B = torch.tensor(1.380649e-16, device=device)
        self.m_H = torch.tensor(1.6735575e-24, device=device)
        self.a_rad = torch.tensor(7.5657e-15, device=device)
        self.mu = 0.6
        self.eps = 1e-30

        self.stats = None
        self.output_cols = None
        self._idx_map = {}

    def set_stats(self, stats):
        self.stats = stats
        if stats and 'output_cols' in stats:
            self.output_cols = stats['output_cols']
            self._idx_map = {
                'T': self.output_cols.index('T') if 'T' in self.output_cols else 0,
                'ne': self.output_cols.index('ne') if 'ne' in self.output_cols else 1,
                'rho': self.output_cols.index('rho') if 'rho' in self.output_cols else 2,
            }
            exclude_cols = {'T', 'ne', 'rho'}
            self.level_pop_indices = [
                i for i, col in enumerate(self.output_cols) if col not in exclude_cols
            ]
        else:
            self.level_pop_indices = list(range(3, len(self.output_cols or [])))

    def _get_idx(self, name):
        return self._idx_map.get(name, {'T': 0, 'ne': 1, 'rho': 2}.get(name, 0))

    def _denorm_tau(self, tau_norm):
        """将归一化的 tau 反变换到物理单位"""
        if self.stats is None or 'tau_stats' not in self.stats:
            return tau_norm
        tau_stats = self.stats['tau_stats']
        ymin = tau_stats['min']
        ymax = tau_stats['max']
        tau = (tau_norm + 1.0) / 2.0 * (ymax - ymin) + ymin
        if self.stats.get('tau_log_transformed', False):
            tau = 10.0 ** tau
        return torch.clamp(tau, min=self.eps, max=1e6)

    def _denorm_if_needed(self, y_pred):
        if self.stats is not None and self.output_cols is not None:
            return inverse_transform_output(y_pred, self.stats, self.output_cols)
        return y_pred

    def safe_log(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

    def _clip_loss(self, loss, max_val=1e2):
        if not torch.isfinite(loss):
            return torch.tensor(0.0, device=self.device)
        return torch.clamp(loss, max=max_val)

    def _boundary_mask(self, shape, exclude_front=1, exclude_back=5):
        """排除大气顶部和底部的极端区域（类似 kurucz 的前2后20掩码）"""
        B, D = shape
        if D <= exclude_front + exclude_back + 2:
            return torch.ones(B, D, device=self.device)
        mask = torch.zeros(B, D, device=self.device)
        mask[:, exclude_front:D - exclude_back] = 1.0
        return mask

    # ------------------------------------------------------------------
    # 流体静力学平衡 — 对数空间版
    # ------------------------------------------------------------------
    def compute_hydrostatic(self, y_pred, x_input, tau):
        """
        流体静力学平衡: dP/dtau = g/kappa

        由于不知道不透明度 kappa，这里采用软约束策略（全部在对数空间）：
        1. sign_loss : dlnP/dln(tau) > 0  （压强随光深单调增）
        2. range_loss: dlnP/dln(tau) ∈ [0.05, 8.0]  （量级合理）
        3. ratio_loss: P_gas/P_total ∈ [0.01, 0.99]

        数学推导:
            dP/dtau = (P/tau) * (dlnP/dln(tau))
            log(dP/dtau) = lnP - ln(tau) + ln|dlnP/dln(tau)|
        """
        batch_size, n_depths, _ = y_pred.shape
        if n_depths < 5:
            return torch.tensor(0.0, device=self.device)

        iT = self._get_idx('T')
        irho = self._get_idx('rho')

        # 严格裁剪到物理范围（防止反归一化后的极端值）
        T = torch.clamp(y_pred[..., iT], min=1000.0, max=200000.0)
        rho = torch.clamp(y_pred[..., irho], min=1e-30, max=1e3)
        tau = torch.clamp(tau, min=self.eps, max=1e6)

        # 计算气体压强 + 辐射压强
        N = rho / (self.mu * self.m_H)
        P_gas = torch.clamp(N * self.k_B * T, max=1e20)
        P_rad = torch.clamp((self.a_rad / 3.0) * T ** 4, max=1e20)
        P_total = torch.clamp(P_gas + P_rad, min=self.eps, max=1e20)

        # 对数空间
        logP = self.safe_log(P_total)     # [B, D]
        logtau = self.safe_log(tau)       # [B, D]

        # 中心差分计算 dlnP/dln(tau)
        dlnP = logP[:, 2:] - logP[:, :-2]               # [B, D-2]
        dlntau = logtau[:, 2:] - logtau[:, :-2]         # [B, D-2]
        dlnP_dlntau = dlnP / (dlntau + self.eps)        # [B, D-2]
        dlnP_dlntau = torch.clamp(dlnP_dlntau, min=-50.0, max=50.0)

        # 边界掩码：排除前1层和后5层（50层大气，比kurucz的80层更紧凑）
        mask = self._boundary_mask((batch_size, n_depths - 2),
                                   exclude_front=1, exclude_back=5)
        n_valid = mask.sum() + self.eps

        # sign_loss: 压强应随 tau 增加（dlnP/dln(tau) > 0）
        sign_violation = torch.relu(-dlnP_dlntau)
        sign_loss = torch.sum((sign_violation ** 2) * mask) / n_valid

        # range_loss: dlnP/dln(tau) 在合理范围
        too_small = torch.relu(0.05 - dlnP_dlntau)
        too_large = torch.relu(dlnP_dlntau - 8.0)
        range_loss = torch.sum((too_small ** 2 + too_large ** 2) * mask) / n_valid

        # ratio_loss: P_gas / P_total 比值合理
        ratio = P_gas / (P_total + self.eps)
        ratio_loss = torch.mean(
            torch.relu(0.01 - ratio) ** 2 + torch.relu(ratio - 0.99) ** 2
        )

        loss = sign_loss + 0.2 * range_loss + 0.5 * ratio_loss
        return self._clip_loss(loss, max_val=1e2)

    # ------------------------------------------------------------------
    # 辐射平衡 — 对数空间版
    # ------------------------------------------------------------------
    def compute_radiative(self, y_pred, x_input, tau):
        """
        深层辐射平衡（Eddington 近似）

        T^4 = 0.75 * Teff^4 * (tau + 2/3)，仅 tau > 1 生效
        误差在对数空间计算，避免大动态范围问题。
        """
        batch_size, n_depths, _ = y_pred.shape

        T = torch.clamp(y_pred[..., self._get_idx('T')], min=1000.0, max=200000.0)
        Teff = torch.clamp(x_input[..., 0], min=5000.0, max=2e5)

        tau = torch.clamp(tau, min=self.eps, max=1e6)

        T4_expected = 0.75 * Teff.unsqueeze(1) ** 4 * (tau + 2.0 / 3.0)
        T_expected = torch.clamp(T4_expected, min=self.eps) ** 0.25

        mask = (tau > 1.0).float()

        log_T = self.safe_log(T)
        log_T_exp = self.safe_log(T_expected)
        diff = (log_T - log_T_exp) * mask

        valid_count = mask.sum() + self.eps
        loss = torch.sum(diff ** 2) / valid_count

        return self._clip_loss(loss, max_val=1e2)

    # ------------------------------------------------------------------
    # 电离度约束 — 单点约束，数值极稳定
    # ------------------------------------------------------------------
    def compute_ionization(self, y_pred):
        """ne / (rho/(mu*m_H)) ∈ [0.01, 0.99]"""
        ine = self._get_idx('ne')
        irho = self._get_idx('rho')

        ne = torch.clamp(y_pred[..., ine], min=self.eps, max=1e25)
        rho = torch.clamp(y_pred[..., irho], min=self.eps, max=1e3)

        N_total = rho / (self.mu * self.m_H)
        ionization_frac = ne / (N_total + self.eps)

        low = torch.relu(0.01 - ionization_frac)
        high = torch.relu(ionization_frac - 0.99)

        loss = torch.mean(low ** 2 + high ** 2)
        return self._clip_loss(loss, max_val=1e1)

    # ------------------------------------------------------------------
    # 温度深层单调性 — 仅后 60% 层，归一化误差
    # ------------------------------------------------------------------
    def compute_temp_mono(self, y_pred):
        T = torch.clamp(y_pred[..., self._get_idx('T')], min=100.0, max=500000.0)
        if T.shape[1] < 2:
            return torch.tensor(0.0, device=self.device)

        n_depths = T.shape[1]
        deep_start = int(n_depths * 0.4)
        if deep_start >= n_depths - 1:
            return torch.tensor(0.0, device=self.device)

        T_deep = T[:, deep_start:]
        dT = T_deep[:, 1:] - T_deep[:, :-1]
        # 归一化：dT/T，避免大温度值导致惩罚过大
        violation = torch.relu(-dT / (T_deep[:, :-1] + self.eps))

        batch_size = T.shape[0]
        mask = self._boundary_mask((batch_size, T_deep.shape[1] - 1),
                                   exclude_front=0, exclude_back=3)
        n_valid = mask.sum() + self.eps

        loss = torch.sum((violation ** 2) * mask) / n_valid
        return self._clip_loss(loss, max_val=1e1)

    # ------------------------------------------------------------------
    # 密度单调性 — 对数空间
    # ------------------------------------------------------------------
    def compute_density_mono(self, y_pred):
        rho = torch.clamp(y_pred[..., self._get_idx('rho')], min=self.eps, max=1e3)
        if rho.shape[1] < 2:
            return torch.tensor(0.0, device=self.device)

        log_rho = self.safe_log(rho)
        d_log_rho = log_rho[:, 1:] - log_rho[:, :-1]
        violation = torch.relu(-d_log_rho)

        batch_size = rho.shape[0]
        mask = self._boundary_mask((batch_size, rho.shape[1] - 1),
                                   exclude_front=0, exclude_back=3)
        n_valid = mask.sum() + self.eps

        loss = torch.sum((violation ** 2) * mask) / n_valid
        return self._clip_loss(loss, max_val=1e1)

    # ------------------------------------------------------------------
    # 平滑性 — 对数/归一化空间
    # ------------------------------------------------------------------
    def compute_smoothness(self, y_pred):
        batch_size, n_depths, n_outputs = y_pred.shape
        if n_depths < 5:
            return torch.tensor(0.0, device=self.device)

        iT = self._get_idx('T')
        ine = self._get_idx('ne')
        irho = self._get_idx('rho')

        total = torch.tensor(0.0, device=self.device)

        # 温度：归一化二阶导（线性空间）
        T = torch.clamp(y_pred[..., iT], min=100.0, max=500000.0)
        d2T = T[:, 2:] - 2 * T[:, 1:-1] + T[:, :-2]
        d2T_norm = d2T / (torch.abs(T[:, 1:-1]) + 1e-3)
        total += torch.mean(torch.clamp(d2T_norm, min=-10.0, max=10.0) ** 2)

        # ne：对数空间二阶导
        ne = torch.clamp(y_pred[..., ine], min=self.eps)
        log_ne = self.safe_log(ne)
        d2_ne = log_ne[:, 2:] - 2 * log_ne[:, 1:-1] + log_ne[:, :-2]
        total += 0.5 * torch.mean(d2_ne ** 2)

        # rho：对数空间二阶导
        rho = torch.clamp(y_pred[..., irho], min=self.eps)
        log_rho = self.safe_log(rho)
        d2_rho = log_rho[:, 2:] - 2 * log_rho[:, 1:-1] + log_rho[:, :-2]
        total += 0.5 * torch.mean(d2_rho ** 2)

        return self._clip_loss(total, max_val=1e2)

    # ------------------------------------------------------------------
    # 能级布居数 — 仅平滑性（禁用单调递减假设）
    # ------------------------------------------------------------------
    def compute_level_pops_constraints(self, y_pred):
        batch_size, n_depths, n_outputs = y_pred.shape
        level_pop_indices = getattr(self, 'level_pop_indices', [])
        if not level_pop_indices or n_depths < 3:
            return torch.tensor(0.0, device=self.device)

        level_pops = torch.clamp(y_pred[..., level_pop_indices], min=self.eps, max=1e20)
        log_pops = torch.log(level_pops + self.eps)
        d_log_pops = log_pops[:, 1:, :] - log_pops[:, :-1, :]
        large_jump = torch.relu(torch.abs(d_log_pops) - 10.0)

        loss = 0.01 * torch.mean(large_jump ** 2)
        return self._clip_loss(loss, max_val=1e1)

    # ------------------------------------------------------------------
    # 总物理损失
    # ------------------------------------------------------------------
    def compute_all(self, y_pred, x_input, tau_norm, weights):
        losses = {}
        y_phys = self._denorm_if_needed(y_pred)
        tau_phys = self._denorm_tau(tau_norm)

        from .data_loader import denormalize_input_physical
        if self.stats is not None:
            x_phys = denormalize_input_physical(x_input, self.stats)
        else:
            x_phys = x_input

        if weights.get('physics_hydrostatic', 0) > 0:
            losses['hydrostatic'] = self.compute_hydrostatic(y_phys, x_phys, tau_phys)

        if weights.get('physics_radiative', 0) > 0:
            losses['radiative'] = self.compute_radiative(y_phys, x_phys, tau_phys)

        if weights.get('physics_ionization', 0) > 0:
            losses['ionization'] = self.compute_ionization(y_phys)

        if weights.get('physics_temp_mono', 0) > 0:
            losses['temp_mono'] = self.compute_temp_mono(y_phys)

        if weights.get('physics_density_mono', 0) > 0:
            losses['density_mono'] = self.compute_density_mono(y_phys)

        if weights.get('physics_smoothness', 0) > 0:
            losses['smoothness'] = self.compute_smoothness(y_phys)

        if weights.get('physics_level_pops', 0) > 0:
            losses['level_pops'] = self.compute_level_pops_constraints(y_phys)

        total = torch.tensor(0.0, device=self.device)
        for name, loss in losses.items():
            weight = weights.get(f'physics_{name}', 0)
            if weight > 0 and torch.isfinite(loss):
                total = total + weight * loss

        return total, losses
