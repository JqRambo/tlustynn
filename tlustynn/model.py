import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEncoding(nn.Module):
    """正弦位置编码，用于深度维度"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        half_dim = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=x.device) / half_dim)
        args = x * freqs.unsqueeze(0) * 2 * np.pi
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class FourierFeatures(nn.Module):
    """傅里叶特征映射，用于恒星参数"""
    def __init__(self, input_dim, mapping_size=256, scale=4.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        
        B = torch.randn(input_dim, mapping_size // 2) * scale
        self.B = nn.Parameter(B)  
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    """带门控机制的残差块"""
    def __init__(self, dim, activation='silu', dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        if activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.Tanh()
        
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 2, dim) 
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class TLUSTYNN(nn.Module):
    """TLUSTY 大气模型 NN

    输入：teff, logg, mh（3个恒星参数）+ tau（光深，作为逐层输入）
    输出：T, ne, rho, level_1...level_55（每层的58个大气参数）
    """
    def __init__(self, 
                 input_dim=3,          # 3 个恒星参数
                 output_dim=58,        # T + ne + rho + 55 能级（tau 作为输入）
                 n_depths=50,
                 hidden_layers=None,
                 activation='silu',
                 dropout=0.1,
                 use_fourier=True,
                 fourier_dim=256,
                 fourier_scale=4.0,
                 use_output_clamp=False,
                 output_clamp_range=10.0,
                 use_checkpoint=False):  
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_depths = n_depths
        self.use_output_clamp = use_output_clamp
        self.output_clamp_range = output_clamp_range
        
        if hidden_layers is None:
            hidden_layers = [1024, 2048, 4096, 4096, 2048, 1024, 512]
        
        # 恒星参数维度 = 输入维度（因为 tau 不再是输入）
        self.stellar_dim = input_dim
        
        if use_fourier:
            self.fourier = FourierFeatures(self.stellar_dim, fourier_dim, fourier_scale)
            stellar_encoded_dim = fourier_dim
        else:
            self.fourier = None
            stellar_encoded_dim = self.stellar_dim
        
        self.depth_encoding = SinusoidalPositionEncoding(128)
        depth_encoded_dim = 128
        
        combined_dim = stellar_encoded_dim + depth_encoded_dim
        
        self.input_norm = nn.LayerNorm(combined_dim)
        self.input_proj = nn.Linear(combined_dim, hidden_layers[0])
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        
        self.blocks = nn.ModuleList()
        prev_dim = hidden_layers[0]
        
        for dim in hidden_layers[1:]:
            if dim != prev_dim:
                transition = nn.Sequential(
                    nn.LayerNorm(prev_dim),
                    nn.Linear(prev_dim, dim),
                    nn.SiLU() if activation == 'silu' else nn.Tanh(),
                )
                self.blocks.append(transition)
            
            self.blocks.append(ResidualBlock(dim, activation, dropout))
            prev_dim = dim
        
        self.output_norm = nn.LayerNorm(hidden_layers[-1])
        self.output_proj = nn.Linear(hidden_layers[-1], output_dim)
        
        nn.init.xavier_normal_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)
        
        self.output_scale = nn.Parameter(torch.ones(output_dim) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(output_dim))
        self.output_activation = nn.Tanh()
    
    def forward(self, stellar_params, tau=None, depth=None):
        """前向传播
        
        Args:
            stellar_params: [batch, 3] 恒星参数（已归一化到 [-1, 1]）
            tau: [batch, 50] 光深（已归一化到 [-1, 1]），优先使用
            depth: [batch, 50] 深度编码（可选，tau 为 None 时回退使用层索引）
        
        Returns:
            y: [batch, 50, output_dim] 预测的大气参数（归一化到 [-1, 1]）
        """
        if self.use_checkpoint and self.training:
            return self._forward_impl(stellar_params, tau, depth, use_checkpoint=True)
        return self._forward_impl(stellar_params, tau, depth, use_checkpoint=False)
    
    def _forward_impl(self, stellar_params, tau, depth, use_checkpoint=False):
        batch_size = stellar_params.shape[0]
        
        # 扩展恒星参数到每个深度层
        if stellar_params.dim() == 2:
            stellar_params = stellar_params.unsqueeze(1).expand(-1, self.n_depths, -1)
        
        stellar_params = stellar_params.reshape(batch_size * self.n_depths, self.stellar_dim)
        
        # 傅里叶特征编码
        if self.fourier is not None:
            stellar_encoded = self.fourier(stellar_params)
        else:
            stellar_encoded = stellar_params
        
        # 深度编码：优先使用 tau（光深），否则回退到 depth
        if tau is not None:
            # tau 已经是归一化到 [-1, 1] 的值，直接映射到 [0, 1] 后编码
            depth_norm = (tau + 1.0) / 2.0
            depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
            depth_norm = depth_norm.reshape(batch_size * self.n_depths, 1)
            depth_encoded = self.depth_encoding(depth_norm)
        elif depth is not None:
            depth_norm = (depth + 1.0) / 2.0
            depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
            depth_norm = depth_norm.reshape(batch_size * self.n_depths, 1)
            depth_encoded = self.depth_encoding(depth_norm)
        else:
            depth_vals = torch.linspace(-1, 1, self.n_depths, device=stellar_params.device)
            depth = depth_vals.unsqueeze(0).expand(batch_size, -1)
            depth_norm = (depth + 1.0) / 2.0
            depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
            depth_norm = depth_norm.reshape(batch_size * self.n_depths, 1)
            depth_encoded = self.depth_encoding(depth_norm)
        
        # 合并恒星参数和深度编码
        x = torch.cat([stellar_encoded, depth_encoded], dim=-1)
        
        x = self.input_proj(self.input_norm(x))
        
        for block in self.blocks:
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.output_norm(x)
        y = self.output_proj(x)
        
        y = y * self.output_scale + self.output_bias
        
        y = self.output_activation(y)
        
        y = y.view(batch_size, self.n_depths, self.output_dim)
        
        return y
    
    def predict_single(self, teff, logg, mh, tau=None):
        """预测单个恒星模型（物理单位输入）
        
        Args:
            teff, logg, mh: 恒星参数（物理单位）
            tau: [50] 光深（归一化到 [-1,1]），None 时使用默认层索引
        """
        device = next(self.parameters()).device
        x = torch.tensor([[teff, logg, mh]], dtype=torch.float32, device=device)
        if tau is not None:
            tau = torch.tensor(tau, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return self.forward(x, tau=tau).cpu().numpy()[0]


def create_model(model_type='mlp', **kwargs):
    if model_type == 'mlp':
        return TLUSTYNN(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
