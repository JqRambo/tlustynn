import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import TRAIN_CONFIG, DATA_CONFIG, CSV_PATH, CHECKPOINT_DIR
from .data_loader import (round_to_dataset_grid, denormalize_input_physical,
                         inverse_transform_output, inverse_transform_input,
                         load_and_preprocess_data, create_data_loaders)
from .model import create_model
from .predict import TlustyPredictor
from .utils import (create_single_model_plot, create_multi_model_comparison,
                   create_scatter_comparison, create_diagonal_comparison,
                   create_error_by_depth)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(predictor, test_loader, stats, output_dir=None, max_samples=None):
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    total_samples = 0

    for batch in test_loader:
        if max_samples and total_samples >= max_samples:
            break
            
        x = batch['x'].to(predictor.device)
        y_true = batch['y'].to(predictor.device)
        
        depth = batch['depth'].to(predictor.device)

        with torch.no_grad():
            y_pred = predictor.model(x, mass=batch['M'].to(predictor.device), depth=depth)
        
        all_predictions.append(y_pred.cpu())
        all_targets.append(y_true.cpu())
        all_inputs.append(x.cpu())
        total_samples += x.shape[0]
    
    predictions = torch.cat(all_predictions, dim=0)  # [n_models, 50, n_outputs]
    targets = torch.cat(all_targets, dim=0)
    inputs = torch.cat(all_inputs, dim=0)
    
    output_cols = stats['output_cols']
    input_cols = stats.get('input_cols', ['teff', 'logg', 'log_he_h'])
    predictions_denorm = inverse_transform_output(predictions, stats, output_cols)
    targets_denorm = inverse_transform_output(targets, stats, output_cols)

    inputs_denorm = inverse_transform_input(inputs, stats, input_cols=input_cols)
    
    sample_teff = inputs_denorm[0, 0].item()
    print(f"  Sample input after denorm: Teff={sample_teff:.0f}K")
    
    output_cols = stats.get('output_cols', [f'col_{i}' for i in range(predictions_denorm.shape[-1])])
    physical_indices = {name: output_cols.index(name) for name in ['T', 'ne', 'rho', 'M']}

    metrics = {}

    pred_flat = predictions_denorm.numpy().flatten().astype(np.float64)
    target_flat = targets_denorm.numpy().flatten().astype(np.float64)
    valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
    pred_flat_valid = pred_flat[valid_mask]
    target_flat_valid = target_flat[valid_mask]

    print(f"Valid pixels: {valid_mask.sum()}/{len(valid_mask)} ({100*valid_mask.sum()/len(valid_mask):.1f}%)")

    if len(pred_flat_valid) > 0:
        metrics['overall'] = {
            'mse': float(mean_squared_error(target_flat_valid, pred_flat_valid)),
            'mae': float(mean_absolute_error(target_flat_valid, pred_flat_valid)),
            'r2': float(r2_score(target_flat_valid, pred_flat_valid)),
        }
    else:
        metrics['overall'] = {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}

    def compute_metrics_safe(pred, target):
        mask = np.isfinite(pred) & np.isfinite(target)
        p = pred[mask].astype(np.float64)
        t = target[mask].astype(np.float64)
        if len(p) == 0:
            return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'mean_relative_error': float('nan')}
        return {
            'mse': float(mean_squared_error(t, p)),
            'mae': float(mean_absolute_error(t, p)),
            'r2': float(r2_score(t, p)),
            'mean_relative_error': float(np.mean(np.abs(p - t) / (t + 1e-10))),
        }

    metrics['per_variable'] = {}
    for name, idx in physical_indices.items():
        pred = predictions_denorm[:, :, idx].numpy().flatten()
        target = targets_denorm[:, :, idx].numpy().flatten()
        metrics['per_variable'][name] = compute_metrics_safe(pred, target)

    level_indices = [i for i, col in enumerate(output_cols) if col.startswith('level_')]
    level_pred = predictions_denorm[:, :, level_indices].numpy().flatten()
    level_target = targets_denorm[:, :, level_indices].numpy().flatten()
    metrics['per_variable']['level_pops'] = compute_metrics_safe(level_pred, level_target)

    metrics['depth_resolved'] = {f'{name}_mse': [] for name in physical_indices}
    for depth in range(50):
        for name, idx in physical_indices.items():
            pred = predictions_denorm[:, depth, idx].numpy().astype(np.float64)
            target = targets_denorm[:, depth, idx].numpy().astype(np.float64)
            mask = np.isfinite(pred) & np.isfinite(target)
            mse = mean_squared_error(target[mask], pred[mask]) if mask.any() else float('nan')
            metrics['depth_resolved'][f'{name}_mse'].append(float(mse))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        np.save(os.path.join(output_dir, 'predictions.npy'), predictions_denorm.numpy())
        np.save(os.path.join(output_dir, 'targets.npy'), targets_denorm.numpy())
        np.save(os.path.join(output_dir, 'inputs.npy'), inputs_denorm.numpy())
    
    return metrics, predictions_denorm, targets_denorm, inputs_denorm


def _round_to_dataset_grid(inputs):
    """Round input parameters to the dataset grid."""
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.clone()
        arr = inputs
    else:
        inputs = np.array(inputs, dtype=np.float32)
        inputs = inputs.copy()
        arr = inputs
    
    teff_vals = arr[:, 0]
    if (teff_vals.max() <= 1.5 and teff_vals.min() >= -1.5):
        return inputs
    
    for i in range(len(arr)):
        teff, logg, log_he_h = arr[i, 0], arr[i, 1], arr[i, 2]
        arr[i, 0], arr[i, 1], arr[i, 2] = round_to_dataset_grid(teff, logg, log_he_h)
    
    return arr


def plot_results(predictions, targets, inputs, stats, output_dir, n_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    
    output_cols = stats.get('output_cols', [f'col_{i}' for i in range(predictions.shape[-1])])
    try:
        idx_T = output_cols.index('T')
        idx_ne = output_cols.index('ne')
        idx_rho = output_cols.index('rho')
    except ValueError:
        idx_T, idx_ne, idx_rho = 0, 1, 2
    
    n_models = predictions.shape[0]
    print(n_models)
    
    n_examples = min(n_samples, n_models)
    preferred_indices = [7500, 5333, 4500, 6000, 1500]
    sample_indices = [index for index in preferred_indices if index < n_models][:n_examples]
    if len(sample_indices) < n_examples:
        fallback = np.linspace(0, n_models - 1, n_examples, dtype=int).tolist()
        sample_indices.extend(index for index in fallback if index not in sample_indices)
        sample_indices = sample_indices[:n_examples]

    if isinstance(inputs, torch.Tensor):
        inputs_np = inputs.numpy()
    else:
        inputs_np = np.array(inputs)

    if inputs_np[:, 0].max() <= 1.5 and inputs_np[:, 0].min() >= -1.5:
        inputs_np = denormalize_input_physical(inputs_np, stats, input_cols=['teff', 'logg', 'log_he_h']).numpy()

    inputs_display = _round_to_dataset_grid(inputs_np)
    
    for i in range(min(3, len(inputs_display))):
        teff_val = float(inputs_display[i, 0])
        logg_val = float(inputs_display[i, 1])
        log_he_h_val = float(inputs_display[i, 2])
        print(f"  Plot model {i+1}: Teff={teff_val:.0f}K, logg={logg_val:.2f}, log(n_He/n_H)={log_he_h_val:+.2f}")
    
    for i, idx in enumerate(sample_indices):
        create_single_model_plot(predictions, targets, inputs_display, idx, output_dir, stats=stats)
    
    # multi_indices = sample_indices[:3].tolist()
    multi_indices = sample_indices[:3]

    if len(multi_indices) >= 2:
        create_multi_model_comparison(predictions, targets, inputs_display, multi_indices, output_dir, stats=stats)
    
    create_scatter_comparison(predictions, targets, output_dir, max_points=5000, stats=stats)
    create_diagonal_comparison(predictions, targets, output_dir, max_points=5000, stats=stats)
    create_error_by_depth(predictions, targets, output_dir, stats=stats)


EVAL_CONFIG = {
    'checkpoint': None,
    'output_dir': './results',
    'n_samples': 5,
    'random_seed': DATA_CONFIG['random_seed'],
}


def main():
    parser = argparse.ArgumentParser(description='TLUSTY NN Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path (override config)')
    parser.add_argument('--output-dir', type=str, default=None, help='Result save directory (override config)')
    parser.add_argument('--n-samples', type=int, default=None, help='Number of sample plots (override config)')
    args = parser.parse_args()
    
    checkpoint = args.checkpoint if args.checkpoint is not None else EVAL_CONFIG['checkpoint']
    output_dir = args.output_dir if args.output_dir is not None else EVAL_CONFIG['output_dir']
    n_samples = args.n_samples if args.n_samples is not None else EVAL_CONFIG['n_samples']
    
    print("="*70)
    print("TLUSTY NN Evaluation")
    print("="*70)
    print(f"Checkpoint: {checkpoint or 'default'}")
    print(f"Output dir: {output_dir}")
    print(f"N samples: {n_samples}")
    print("="*70)
    
    set_seed(EVAL_CONFIG['random_seed'])
    
    predictor = TlustyPredictor(checkpoint_path=checkpoint)
    
    log_transform_cols = DATA_CONFIG.get('log_transform_cols')
    
    stats_path = os.path.join(CHECKPOINT_DIR, 'stats.json')
    with open(stats_path, 'r') as f:
        training_stats = json.load(f)

    df, input_cols, output_cols, computed_stats = load_and_preprocess_data(CSV_PATH, log_transform_cols=log_transform_cols)
    
    if training_stats is not None:
        stats = training_stats
        stats['input_cols'] = input_cols
        stats['output_cols'] = output_cols
    else:
        stats = computed_stats
    
    _, _, test_loader = create_data_loaders(
        df, input_cols, output_cols, stats,
        batch_size=DATA_CONFIG['batch_size'],
        test_ratio=DATA_CONFIG['test_ratio'],
        val_ratio=DATA_CONFIG['val_ratio'],
        random_seed=DATA_CONFIG['random_seed'],
        num_workers=0,  
        log_transform_cols=log_transform_cols
    )
    
    max_eval_samples = None
    metrics, predictions, targets, inputs = evaluate_model(predictor, test_loader, stats, output_dir, max_samples=max_eval_samples)
    
    print("\nEvaluation Metrics:")
    print(f"  Overall MSE: {metrics['overall']['mse']:.6e}")
    print(f"  Overall MAE: {metrics['overall']['mae']:.6e}")
    print(f"  Overall R2:  {metrics['overall']['r2']:.6f}")
    print(f"  T   -> MSE: {metrics['per_variable']['T']['mse']:.6e}, MAE: {metrics['per_variable']['T']['mae']:.6e}")
    print(f"  ne  -> MSE: {metrics['per_variable']['ne']['mse']:.6e}, MAE: {metrics['per_variable']['ne']['mae']:.6e}")
    print(f"  rho -> MSE: {metrics['per_variable']['rho']['mse']:.6e}, MAE: {metrics['per_variable']['rho']['mae']:.6e}")
    print(f"  M   -> MSE: {metrics['per_variable']['M']['mse']:.6e}, MAE: {metrics['per_variable']['M']['mae']:.6e}")
    
    plot_results(predictions, targets, inputs, stats, output_dir, n_samples=n_samples)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
