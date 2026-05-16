import os
import torch

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PACKAGE_ROOT, 'checkpoints')

CSV_PATH = './hhe.csv'

INPUT_COLS = ['teff', 'logg', 'mh']

INPUT_RANGES = {
    'teff': {'min': 10000.0, 'max': 100000.0},
    'logg': {'min': 1.5, 'max': 8.0},
    'mh': {'min': -4.0, 'max': 0.0},
}

INPUT_STEPS = {
    'teff': 500.0,
    'logg': 0.1,
    'mh': 0.5,
}

DATA_CONFIG = {
    'csv_path': CSV_PATH,
    'batch_size': 1280,
    'test_ratio': 0.05,
    'val_ratio': 0.05,
    'random_seed': 42,
    'num_workers': 20,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 6,
    'log_transform_cols': None,
    'normalization': 'minmax',
}


MODEL_CONFIG = {
    'model_type': 'mlp',
    'input_dim': 3,
    'output_dim': 58,
    'n_depths': 50,
    'hidden_layers': [1024, 2048, 2048, 2048, 1024, 512],
    'activation': 'silu',
    'dropout': 0.1,
    'use_fourier': True,
    'fourier_dim': 256,
    'fourier_scale': 4.0,
    'use_output_clamp': False,
    'output_clamp_range': 10.0,
    'use_checkpoint': False,
}


TRAIN_CONFIG = {
    'epochs': 1500,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'gradient_clip': 0.5,
    'scheduler': 'step',
    'step_size': 100,
    'gamma': 0.5,
    'early_stopping_patience': 10000,
    'save_dir': './checkpoints',
    'log_interval': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_amp': True,
    'compile_model': True,
}


VARIABLE_WEIGHTS = {
    'T': 8.0,
    'ne': 2.0,
    'rho': 3.0,
    'default': 2.0,
}


AVG_TAU_SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'avg_tau_physical.npy')

INFERENCE_CONFIG = {
    'checkpoint': 'best_model.pt',
    'output_dir': './predictions',
    'save_format': 'csv',
    'plot_results': True,
}
