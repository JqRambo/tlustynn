import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEncoding(nn.Module):
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
    def __init__(self, 
                 input_dim=3,          # Three stellar parameters
                 output_dim=59,        # M + T + ne + rho + 55 level populations
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
    
    def forward(self, stellar_params, mass=None, depth=None):
        if self.use_checkpoint and self.training:
            return self._forward_impl(stellar_params, mass, depth, use_checkpoint=True)
        return self._forward_impl(stellar_params, mass, depth, use_checkpoint=False)

    def _forward_impl(self, stellar_params, mass, depth, use_checkpoint=False):
        batch_size = stellar_params.shape[0]

        if stellar_params.dim() == 2:
            stellar_params = stellar_params.unsqueeze(1).expand(-1, self.n_depths, -1)

        stellar_params = stellar_params.reshape(batch_size * self.n_depths, self.stellar_dim)

        if self.fourier is not None:
            stellar_encoded = self.fourier(stellar_params)
        else:
            stellar_encoded = stellar_params

        if mass is not None:
            depth_values = mass
        elif depth is not None:
            depth_values = depth
        else:
            depth_values = torch.linspace(-1, 1, self.n_depths, device=stellar_params.device)
            depth_values = depth_values.unsqueeze(0).expand(batch_size, -1)
        depth_norm = (depth_values + 1.0) / 2.0
        depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
        depth_norm = depth_norm.reshape(batch_size * self.n_depths, 1)
        depth_encoded = self.depth_encoding(depth_norm)

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

        return y.view(batch_size, self.n_depths, self.output_dim)

    def predict_single(self, teff, logg, log_he_h):
        device = next(self.parameters()).device
        x = torch.tensor([[teff, logg, log_he_h]], dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.forward(x).cpu().numpy()[0]


def create_model(model_type='mlp', **kwargs):
    if model_type == 'mlp':
        return TLUSTYNN(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
