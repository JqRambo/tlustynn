import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

from .config import AVG_TAU_SAVE_PATH


PHYSICAL_LIMITS = {
    'T': {'min': 100.0, 'max': 1e6},         # 大气温度 T 物理上限约 ~5e5K (99.99 percentile)；1e6 留足安全余量，排除数值异常点
    'Teff': {'min': 9000, 'max': 100000},    # 有效温度（输入参数）范围
    'logg': {'min': 1.5, 'max': 8.0},        # 表面重力
    'ne': {'min': 1e-10, 'max': 1e30},       # 电子数密度 [cm^-3]
    'rho': {'min': 1e-30, 'max': 1e3},       # 质量密度 [g/cm^3]
}


TEFF_LOGG_CONSTRAINTS = {
    'rare_combinations': [(50000, 100000, 2.5)],
    'warning_zones': [(30000, 50000, 2.0)]}


class TlustyDataset(Dataset):
    """TLUSTY 数据集：每个样本是一个恒星大气模型（50层）"""
    def __init__(self, df, input_cols, output_cols, normalize=False,
                 input_stats=None, output_stats=None, log_transform_cols=None):
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.normalize = normalize
        self.input_stats = input_stats
        self.output_stats = output_stats
        self.log_transform_cols = log_transform_cols or []
        # 恒星参数列（不含 tau，tau 作为模型输入）
        self.stellar_input_cols = [c for c in input_cols if c != 'tau']
        
        # 预先把 DataFrame 转成 numpy 数组，避免 __getitem__ 反复做 pandas 过滤
        df = df.reset_index(drop=True)
        df['model_id'] = df[self.stellar_input_cols].astype(str).agg('_'.join, axis=1)
        self.model_ids = df['model_id'].unique()
        self._verify_structure(df, self.model_ids)
        
        sort_col = 'tau' if 'tau' in df.columns else self.stellar_input_cols[0]
        # all_cols 必须包含 tau（用于深度排序和作为模型输入），即使 output_cols 不含 tau
        tau_col = 'tau'
        all_cols = self.stellar_input_cols + [tau_col] + self.output_cols
        
        # 按 model_id 和深度排序，reshape 为 [n_models, 50, n_cols]
        df_sorted = df.sort_values(['model_id', sort_col])
        data = df_sorted[all_cols].values.astype(np.float32)
        n_models = len(self.model_ids)
        data = data.reshape(n_models, 50, len(all_cols))
        
        # x: 恒星参数（每层相同，取第 0 层）
        self._x = data[:, 0, :len(self.stellar_input_cols)]
        # tau: 光深（作为模型输入）
        self._tau = data[:, :, len(self.stellar_input_cols)]
        # y: 输出变量（不含 tau）
        self._y = data[:, :, len(self.stellar_input_cols)+1:]
        # depth 编码：固定 1-50 归一化到 [-1, 1]
        self._depths = (2.0 * (np.arange(50, dtype=np.float32) - 1) / 49.0 - 1.0)
        
        # 释放 pandas 内存（numpy view 已持有数据）
        del df, df_sorted, data
    
    def _verify_structure(self, df, model_ids):
        """验证每个模型是否恰好有 50 层"""
        n_models_checked = min(100, len(model_ids))
        for model_id in model_ids[:n_models_checked]:
            model_data = df[df['model_id'] == model_id]
            if len(model_data) != 50:
                pass
    
    def _normalize_input(self, x):
        """对恒星输入参数进行 Min-Max 归一化到 [-1, 1]"""
        for i, col in enumerate(self.stellar_input_cols):
            if col in self.input_stats:
                xmin = self.input_stats[col]['min']
                xmax = self.input_stats[col]['max']
                if xmax > xmin:
                    x[i] = 2.0 * (x[i] - xmin) / (xmax - xmin) - 1.0
                else:
                    x[i] = 0.0
        return x
    
    def _normalize_output(self, y):
        """对输出变量进行 Min-Max 归一化到 [-1, 1]"""
        for i, col in enumerate(self.output_cols):
            if col in self.output_stats:
                ymin = self.output_stats[col]['min']
                ymax = self.output_stats[col]['max']
                if ymax > ymin:
                    y[:, i] = 2.0 * (y[:, i] - ymin) / (ymax - ymin) - 1.0
                else:
                    y[:, i] = 0.0
        return y
    
    def __len__(self):
        return len(self.model_ids)
    
    def __getitem__(self, idx):
        """获取单个模型样本（向量化版本，直接从预计算 numpy 数组索引）"""
        x = self._x[idx]
        tau = self._tau[idx]
        y = self._y[idx]
        depths = self._depths.copy()
        
        # 对数变换（在归一化之前）
        if self.log_transform_cols:
            for i, col in enumerate(self.output_cols):
                if col in self.log_transform_cols:
                    mask = y[:, i] > 0
                    y[mask, i] = np.log10(y[mask, i])
        
        # 确保恰好 50 层
        if y.shape[0] != 50:
            if y.shape[0] < 50:
                pad = np.zeros((50 - y.shape[0], y.shape[1]), dtype=np.float32)
                y = np.vstack([y, pad])
                tau_pad = np.zeros(50 - len(tau), dtype=np.float32)
                tau = np.concatenate([tau, tau_pad])
                depths = np.concatenate([depths, np.zeros(50 - len(depths), dtype=np.float32)])
            else:
                y = y[:50]
                tau = tau[:50]
                depths = depths[:50]
        
        # 如果需要，进行归一化（通常在 load_and_preprocess_data 中已完成）
        if self.normalize and self.input_stats is not None:
            x = self._normalize_input(x)
        if self.normalize and self.output_stats is not None:
            y = self._normalize_output(y)
            # tau 若在 output_stats 中也进行归一化
            if 'tau' in self.output_stats:
                tau = self._normalize_tau(tau)
        
        result = {
            'x': torch.tensor(x, dtype=torch.float32),
            'tau': torch.tensor(tau, dtype=torch.float32),
            'depth': torch.tensor(depths, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'model_id': self.model_ids[idx]
        }
        
        return result
    


def safe_log10(x, eps=1e-30):
    x = np.where(x <= eps, eps, x)
    return np.log10(x)


def safe_log(x, eps=1e-30):
    x = np.where(x <= eps, eps, x)
    return np.log(x)


def clip_to_physical_range(df, col, physical_type=None):
    """将数据裁剪到物理合理范围"""
    if physical_type and physical_type in PHYSICAL_LIMITS:
        limits = PHYSICAL_LIMITS[physical_type]
        df[col] = df[col].clip(lower=limits['min'], upper=limits['max'])
    return df


def round_to_dataset_grid(teff, logg, mh):
    """将恒星参数四舍五入到数据集的网格上"""
    teff_r = np.round(teff / 500.0) * 500.0
    logg_r = np.round(logg / 0.1) * 0.1
    mh_r = np.round(mh / 0.5) * 0.5
    return teff_r, logg_r, mh_r


def normalize_input_physical(x, stats, input_cols=None):
    """将物理单位的输入归一化到 [-1, 1]"""
    if input_cols is None:
        input_cols = stats.get('input_cols', ['teff', 'logg', 'mh'])
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x_norm = x.clone()
    n_cols = x_norm.shape[-1] if x_norm.dim() >= 1 else len(x_norm)
    for i in range(min(n_cols, len(input_cols))):
        col = input_cols[i]
        if col in stats.get('input', {}):
            xmin = stats['input'][col]['min']
            xmax = stats['input'][col]['max']
            if xmax > xmin:
                if x_norm.dim() >= 1:
                    x_norm[..., i] = 2.0 * (x_norm[..., i] - xmin) / (xmax - xmin) - 1.0
            else:
                if x_norm.dim() >= 1:
                    x_norm[..., i] = 0.0
    return x_norm


def denormalize_input_physical(x_norm, stats, input_cols=None):
    """将 [-1, 1] 的输入反归一化到物理单位"""
    if input_cols is None:
        input_cols = stats.get('input_cols', ['teff', 'logg', 'mh'])
    if isinstance(x_norm, np.ndarray):
        x_norm = torch.tensor(x_norm, dtype=torch.float32)
    x_denorm = x_norm.detach().clone()
    n_cols = x_denorm.shape[-1] if x_denorm.dim() >= 1 else len(x_denorm)
    for i in range(min(n_cols, len(input_cols))):
        col = input_cols[i]
        if col in stats.get('input', {}):
            xmin = stats['input'][col]['min']
            xmax = stats['input'][col]['max']
            if x_denorm.dim() >= 1:
                x_denorm[..., i] = (x_denorm[..., i] + 1.0) / 2.0 * (xmax - xmin) + xmin
    return x_denorm


def load_and_preprocess_data(csv_path, log_transform_cols=None, apply_clipping=True):
    """加载并预处理 TLUSTY 数据"""
    df = pd.read_csv(csv_path)
    
    # 输入列：3 个恒星参数
    input_cols = ['teff', 'logg', 'mh']
    
    # 输出列：tau + T + ne + rho + 55 个能级布居数
    depth_related_cols = ['depth_index', 'depth_index.1', 'DEPTH_INDEX', 'depth']
    exclude_cols = input_cols + depth_related_cols + ['model_id']
    output_cols = [col for col in df.columns if col not in exclude_cols]
    # 确保 tau 在输出列中且排在前面（与原 CSV 顺序一致）
    if 'tau' in output_cols:
        output_cols.remove('tau')
        output_cols = ['tau'] + output_cols
    
    # 生成模型 ID
    df['model_id'] = df[['teff', 'logg', 'mh']].astype(str).agg('_'.join, axis=1)
    
    # 检查异常值
    issue_counts = {}
    for col in output_cols:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col]).sum()
        n_neg = (df[col] < 0).sum()
        n_zero = (df[col] == 0).sum()
        
        if n_nan > 0 or n_inf > 0 or n_neg > 0:
            issue_counts[col] = {'nan': n_nan, 'inf': n_inf, 'neg': n_neg, 'zero': n_zero}
    
    if issue_counts:
        for col, counts in list(issue_counts.items())[:5]:
            print(f"    {col}: NaN={counts['nan']}, Inf={counts['inf']}, Neg={counts['neg']}, Zero={counts['zero']}")
        if len(issue_counts) > 5:
            print(f"    ... and {len(issue_counts) - 5} more columns")
    
    # 清洗数据
    for col in output_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        if df[col].isna().any():
            df[col] = df.groupby('model_id')[col].transform(lambda x: x.ffill().bfill())
        
        if df[col].isna().any():
            col_median = df[col].median()
            if np.isnan(col_median) or col_median == 0:
                col_median = 1e-30
            df[col] = df[col].fillna(col_median)
        
        if col == 'T':
            df[col] = df[col].clip(lower=PHYSICAL_LIMITS['T']['min'],
                                   upper=PHYSICAL_LIMITS['T']['max'])
        else:
            df[col] = df[col].clip(lower=1e-30)
            df[col] = df[col].replace(0, 1e-30)
    
    # 自动决定对数变换列（针对大动态范围变量）
    if log_transform_cols is None:
        log_transform_cols = []
        for col in output_cols:
            # 温度 T 也参与动态范围判断；若 max/min > 100 则做对数变换
            # （原代码此处硬编码跳过 T，导致 T 的归一化动态范围被异常高温点压垮）
            if col.startswith('temp'):
                continue
            
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max / col_min > 100:  # 大动态范围
                log_transform_cols.append(col)
    
    # 执行对数变换
    for col in log_transform_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=1e-30, upper=1e30)
            df[col] = safe_log10(df[col].values)
    
    # 异常值裁剪（5 sigma）
    for col in output_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - 5 * std
        upper = mean + 5 * std
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    # 计算平均 tau 剖面（物理单位），在归一化之前保存
    df_sorted_for_tau = df.sort_values(['model_id', 'tau'])
    tau_data = df_sorted_for_tau.groupby('model_id')['tau'].apply(lambda x: x.values)
    avg_tau_physical = np.mean(np.stack(tau_data.values), axis=0)
    
    avg_tau_dir = os.path.dirname(AVG_TAU_SAVE_PATH)
    os.makedirs(avg_tau_dir, exist_ok=True)
    np.save(AVG_TAU_SAVE_PATH, avg_tau_physical.astype(np.float32))
    print(f"  Saved average physical tau profile to {AVG_TAU_SAVE_PATH}")
    
    # Min-Max 归一化到 [-1, 1]
    normalization = 'minmax'
    
    input_stats = {}
    for col in input_cols:
        xmin = df[col].min()
        xmax = df[col].max()
        if xmax - xmin < 1e-10:
            xmax = xmin + 1.0
        input_stats[col] = {'min': float(xmin), 'max': float(xmax), 'mean': float(df[col].mean()), 'std': float(df[col].std())}
        df[col] = 2.0 * (df[col] - xmin) / (xmax - xmin) - 1.0
        df[col] = df[col].clip(-1, 1)
    
    output_stats = {}
    for col in output_cols:
        ymin = df[col].min()
        ymax = df[col].max()
        if ymax - ymin < 1e-10:
            output_stats[col] = {'min': float(ymin), 'max': float(ymin) + 1.0, 'mean': float(ymin), 'std': 0.0}
            df[col] = 0.0
        else:
            output_stats[col] = {'min': float(ymin), 'max': float(ymax), 'mean': float(df[col].mean()), 'std': float(df[col].std())}
            df[col] = 2.0 * (df[col] - ymin) / (ymax - ymin) - 1.0
            df[col] = df[col].clip(-1, 1)
    
    # 从模型输出中移除 tau，tau 将作为输入
    output_cols_full = output_cols.copy()
    model_output_cols = [c for c in output_cols if c != 'tau']
    
    # 计算平均 tau 剖面（归一化后），用于预测时默认输入
    df_sorted_for_tau = df.sort_values(['model_id', 'tau'])
    tau_data = df_sorted_for_tau.groupby('model_id')['tau'].apply(lambda x: x.values)
    avg_tau_norm = np.mean(np.stack(tau_data.values), axis=0)
    
    stats = {
        'input': input_stats,
        'output': output_stats,
        'log_transform_cols': log_transform_cols,
        'input_cols': input_cols,
        'output_cols': model_output_cols,
        'output_cols_full': output_cols_full,
        'tau_stats': output_stats.get('tau', {}),
        'tau_log_transformed': 'tau' in log_transform_cols,
        'avg_tau_norm': avg_tau_norm.astype(np.float32).tolist(),
        'depth_col': None,  # tau 作为输入，深度编码使用固定层索引或 tau
        'normalization': normalization,
        'norm_range': 'minus1_to_1'
    }
    
    print(f"\nPreprocessing completed:")
    print(f"  Number of models: {len(df) // 50}")
    print(f"  Input range: [{df[input_cols].min().min():.3f}, {df[input_cols].max().max():.3f}] (should be [-1, 1])")
    print(f"  Output range: [{df[model_output_cols].min().min():.3f}, {df[model_output_cols].max().max():.3f}] (should be [-1, 1])")
    print(f"  Normalization: {normalization} to [-1, 1]")
    print(f"  Log-transform columns: {len(log_transform_cols)}")
    
    return df, input_cols, model_output_cols, stats


def inverse_transform_output(y_norm, stats, output_cols):
    """将归一化输出反变换到物理单位"""
    if isinstance(y_norm, np.ndarray):
        y_norm = torch.from_numpy(y_norm).float()
    
    if isinstance(y_norm, torch.Tensor):
        y = y_norm.detach().clone()
    else:
        y = y_norm.copy()
    
    # 反归一化
    for i, col in enumerate(output_cols):
        if col in stats.get('output', {}):
            ymin = stats['output'][col]['min']
            ymax = stats['output'][col]['max']
            if isinstance(y, torch.Tensor):
                y[..., i] = (y[..., i] + 1.0) / 2.0 * (ymax - ymin) + ymin
            else:
                y[..., i] = (y[..., i] + 1.0) / 2.0 * (ymax - ymin) + ymin
    
    # 反对数变换
    log_transform_cols = stats.get('log_transform_cols', [])
    if log_transform_cols:
        for i, col in enumerate(output_cols):
            if col in log_transform_cols:
                if isinstance(y, torch.Tensor):
                    y[..., i] = 10 ** y[..., i]
                else:
                    y[..., i] = 10 ** y[..., i]
    
    # 裁剪到物理合理范围
    for i, col in enumerate(output_cols):
        if col == 'T':
            if isinstance(y, torch.Tensor):
                y[..., i] = torch.clamp(y[..., i], min=PHYSICAL_LIMITS['T']['min'],
                                        max=PHYSICAL_LIMITS['T']['max'])
            else:
                y[..., i] = np.clip(y[..., i], PHYSICAL_LIMITS['T']['min'],
                                    PHYSICAL_LIMITS['T']['max'])
        elif col in ['ne', 'rho']:
            if isinstance(y, torch.Tensor):
                y[..., i] = torch.clamp(y[..., i], min=PHYSICAL_LIMITS[col]['min'])
            else:
                y[..., i] = np.clip(y[..., i], PHYSICAL_LIMITS[col]['min'],
                                    PHYSICAL_LIMITS[col]['max'])
    
    return y


def inverse_transform_input(x_norm, stats, input_cols=None):
    """将归一化输入反变换到物理单位"""
    return denormalize_input_physical(x_norm, stats, input_cols)


def create_data_loaders(df, input_cols, output_cols, stats,
                        batch_size=256, test_ratio=0.15, val_ratio=0.15,
                        random_seed=42, num_workers=4, log_transform_cols=None):
    """创建训练、验证、测试 DataLoader"""
    dataset = TlustyDataset(df, input_cols, output_cols,
                           normalize=False,
                           input_stats=stats['input'],
                           output_stats=stats['output'],
                           log_transform_cols=None)
    
    n_total = len(dataset)
    indices = np.arange(n_total)
    
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=random_seed
    )
    
    val_size = val_ratio / (1 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, random_state=random_seed
    )
    
    print(f"  Train samples: {len(train_idx)} ")
    print(f"  Val samples: {len(val_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True if torch.cuda.is_available() else False,
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 6
    
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader
