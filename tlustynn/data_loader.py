import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


PHYSICAL_LIMITS = {
    'T': {'min': 100.0, 'max': 1e6},         # T 
    'Teff': {'min': 9000, 'max': 100000},    # Teff
    'logg': {'min': 1.5, 'max': 9.0},        # logg
    'ne': {'min': 1e-10, 'max': 1e30},       # ne [cm^-3]
    'rho': {'min': 1e-30, 'max': 1e3},       # rho [g/cm^3]
}


TEFF_LOGG_CONSTRAINTS = {
    'rare_combinations': [(50000, 100000, 2.5)],
    'warning_zones': [(30000, 50000, 2.0)]}


class TlustyDataset(Dataset):
    """Fifty-layer TLUSTY models with M included in the supervised outputs."""
    def __init__(self, df, input_cols, output_cols, normalize=False,
                 input_stats=None, output_stats=None, log_transform_cols=None):
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.normalize = normalize
        self.input_stats = input_stats
        self.output_stats = output_stats
        self.log_transform_cols = log_transform_cols or []
        self.stellar_input_cols = input_cols

        if 'M' not in self.output_cols:
            raise ValueError("M must be included in output_cols")

        df = df.reset_index(drop=True)
        df['model_id'] = df[self.stellar_input_cols].astype(str).agg('_'.join, axis=1)
        self.model_ids = df['model_id'].unique()
        self._verify_structure(df, self.model_ids)

        all_cols = self.stellar_input_cols + self.output_cols
        df_sorted = df.sort_values(['model_id', 'M'])
        data = df_sorted[all_cols].values.astype(np.float32)
        n_models = len(self.model_ids)
        data = data.reshape(n_models, 50, len(all_cols))

        self._x = data[:, 0, :len(self.stellar_input_cols)]
        self._y = data[:, :, len(self.stellar_input_cols):]
        self._mass = self._y[:, :, 0].copy()
        self._depths = 2.0 * np.arange(50, dtype=np.float32) / 49.0 - 1.0

        del df, df_sorted, data

    def _verify_structure(self, df, model_ids):
        counts = df.groupby('model_id').size()
        invalid = counts[counts != 50]
        if not invalid.empty:
            raise ValueError(f"Expected 50 layers per model; found {len(invalid)} invalid models")

    def _normalize_input(self, x):
        for i, col in enumerate(self.stellar_input_cols):
            if col in self.input_stats:
                xmin = self.input_stats[col]['min']
                xmax = self.input_stats[col]['max']
                x[i] = 2.0 * (x[i] - xmin) / (xmax - xmin) - 1.0 if xmax > xmin else 0.0
        return x

    def _normalize_output(self, y):
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
        x = self._x[idx].copy()
        y = self._y[idx].copy()
        mass = self._mass[idx].copy()
        depths = self._depths.copy()

        if self.log_transform_cols:
            for i, col in enumerate(self.output_cols):
                if col in self.log_transform_cols:
                    mask = y[:, i] > 0
                    y[mask, i] = np.log10(y[mask, i])

        if self.normalize and self.input_stats is not None:
            x = self._normalize_input(x)
        if self.normalize and self.output_stats is not None:
            y = self._normalize_output(y)
            if 'M' in self.log_transform_cols:
                mass = np.log10(np.maximum(mass, 1e-30))
            if 'M' in self.output_stats:
                ymin = self.output_stats['M']['min']
                ymax = self.output_stats['M']['max']
                mass = 2.0 * (mass - ymin) / (ymax - ymin) - 1.0

        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'M': torch.tensor(mass, dtype=torch.float32),
            'depth': torch.tensor(depths, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'model_id': self.model_ids[idx]
        }
    


def safe_log10(x, eps=1e-30):
    x = np.where(x <= eps, eps, x)
    return np.log10(x)


def safe_log(x, eps=1e-30):
    x = np.where(x <= eps, eps, x)
    return np.log(x)


def clip_to_physical_range(df, col, physical_type=None):
    if physical_type and physical_type in PHYSICAL_LIMITS:
        limits = PHYSICAL_LIMITS[physical_type]
        df[col] = df[col].clip(lower=limits['min'], upper=limits['max'])
    return df


def round_to_dataset_grid(teff, logg, log_he_h):
    """Round stellar parameters to the dataset grid."""
    teff_r = np.round(teff / 500.0) * 500.0
    logg_r = np.round(logg / 0.1) * 0.1
    log_he_h_r = np.round(log_he_h / 0.5) * 0.5
    return teff_r, logg_r, log_he_h_r


def normalize_input_physical(x, stats, input_cols=None):
    """Normalize physical-unit inputs to [-1, 1]."""
    if input_cols is None:
        input_cols = stats.get('input_cols', ['teff', 'logg', 'log_he_h'])
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
    """Restore inputs from [-1, 1] to physical units."""
    if input_cols is None:
        input_cols = stats.get('input_cols', ['teff', 'logg', 'log_he_h'])
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
    df = pd.read_csv(csv_path)
    
    input_cols = ['teff', 'logg', 'log_he_h']
    
    # M + T + ne + rho + 55 
    depth_related_cols = ['depth_index', 'depth_index.1', 'DEPTH_INDEX', 'depth']
    exclude_cols = input_cols + depth_related_cols + ['model_id']
    output_cols = [col for col in df.columns if col not in exclude_cols]
    if 'M' in output_cols:
        output_cols.remove('M')
        output_cols = ['M'] + output_cols
    
    df['model_id'] = df[['teff', 'logg', 'log_he_h']].astype(str).agg('_'.join, axis=1)
    
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
    
    if log_transform_cols is None:
        log_transform_cols = []
        for col in output_cols:
            if col.startswith('temp'):
                continue
            
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max / col_min > 100:  
                log_transform_cols.append(col)
    
    for col in log_transform_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=1e-30, upper=1e30)
            df[col] = safe_log10(df[col].values)
    
    for col in output_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - 5 * std
        upper = mean + 5 * std
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
    
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

    stats = {
        'input': input_stats,
        'output': output_stats,
        'log_transform_cols': log_transform_cols,
        'input_cols': input_cols,
        'output_cols': output_cols,
        'mass_stats': output_stats.get('M', {}),
        'mass_log_transformed': 'M' in log_transform_cols,
        'depth_col': None,
        'normalization': normalization,
        'norm_range': 'minus1_to_1'
    }

    print(f"\nPreprocessing completed:")
    print(f"  Number of models: {len(df) // 50}")
    print(f"  Outputs per depth: {len(output_cols)} (including predicted M)")
    print(f"  Input range: [{df[input_cols].min().min():.3f}, {df[input_cols].max().max():.3f}] (should be [-1, 1])")
    print(f"  Output range: [{df[output_cols].min().min():.3f}, {df[output_cols].max().max():.3f}] (should be [-1, 1])")
    print(f"  Normalization: {normalization} to [-1, 1]")
    print(f"  Log-transform columns: {len(log_transform_cols)}")

    return df, input_cols, output_cols, stats


def inverse_transform_output(y_norm, stats, output_cols):
    if isinstance(y_norm, np.ndarray):
        y_norm = torch.from_numpy(y_norm).float()
    
    if isinstance(y_norm, torch.Tensor):
        y = y_norm.detach().clone()
    else:
        y = y_norm.copy()
    
    for i, col in enumerate(output_cols):
        if col in stats.get('output', {}):
            ymin = stats['output'][col]['min']
            ymax = stats['output'][col]['max']
            if isinstance(y, torch.Tensor):
                y[..., i] = (y[..., i] + 1.0) / 2.0 * (ymax - ymin) + ymin
            else:
                y[..., i] = (y[..., i] + 1.0) / 2.0 * (ymax - ymin) + ymin
    
    log_transform_cols = stats.get('log_transform_cols', [])
    if log_transform_cols:
        for i, col in enumerate(output_cols):
            if col in log_transform_cols:
                if isinstance(y, torch.Tensor):
                    y[..., i] = 10 ** y[..., i]
                else:
                    y[..., i] = 10 ** y[..., i]
    
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
    return denormalize_input_physical(x_norm, stats, input_cols)


def create_data_loaders(df, input_cols, output_cols, stats,
                        batch_size=256, test_ratio=0.15, val_ratio=0.15,
                        random_seed=42, num_workers=4, log_transform_cols=None):
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
