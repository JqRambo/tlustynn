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

from tlustynn.config import TRAIN_CONFIG, DATA_CONFIG, CSV_PATH
from tlustynn.data_loader import (round_to_dataset_grid, denormalize_input_physical,
                         inverse_transform_output, inverse_transform_input,
                         load_and_preprocess_data, create_data_loaders)
from tlustynn.model import create_model
from tlustynn.predict import TlustyPredictor
from tlustynn.utils import (create_single_model_plot, create_multi_model_comparison, create_scatter_comparison, create_error_by_depth)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(predictor, test_loader, stats, output_dir=None, max_samples=None):
    """在测试集上评估模型性能"""
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    total_samples = 0

    for batch in test_loader:
        if max_samples and total_samples >= max_samples:
            break
            
        x = batch['x'].to(predictor.device)
        y_true = batch['y'].to(predictor.device)
        
        if 'tau' in batch:
            tau = batch['tau'].to(predictor.device)
        else:
            tau = None
        
        if 'depth' in batch:
            depth = batch['depth'].to(predictor.device)
        else:
            depth = None
        
        with torch.no_grad():
            y_pred = predictor.model(x, tau=tau, depth=depth)
        
        all_predictions.append(y_pred.cpu())
        all_targets.append(y_true.cpu())
        all_inputs.append(x.cpu())
        total_samples += x.shape[0]
    
    predictions = torch.cat(all_predictions, dim=0)  # [n_models, 50, n_outputs]
    targets = torch.cat(all_targets, dim=0)
    inputs = torch.cat(all_inputs, dim=0)
    
    output_cols = stats['output_cols']
    input_cols = stats.get('input_cols', ['teff', 'logg', 'mh'])
    predictions_denorm = inverse_transform_output(predictions, stats, output_cols)
    targets_denorm = inverse_transform_output(targets, stats, output_cols)

    inputs_denorm = inverse_transform_input(inputs, stats, input_cols=input_cols)
    
    # 打印第一个样本的反归一化输入，用于调试
    sample_teff = inputs_denorm[0, 0].item()
    print(f"  Sample input after denorm: Teff={sample_teff:.0f}K")
    
    output_cols = stats.get('output_cols', [f'col_{i}' for i in range(predictions_denorm.shape[-1])])
    try:
        idx_T = output_cols.index('T')
        idx_ne = output_cols.index('ne')
        idx_rho = output_cols.index('rho')
    except ValueError:
        idx_T, idx_ne, idx_rho = 0, 1, 2
    
    metrics = {}
    
    pred_flat = predictions_denorm.numpy().flatten()
    target_flat = targets_denorm.numpy().flatten()
    
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
        p, t = pred[mask], target[mask]
        if len(p) == 0:
            return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'mean_relative_error': float('nan')}
        return {
            'mse': float(mean_squared_error(t, p)),
            'mae': float(mean_absolute_error(t, p)),
            'r2': float(r2_score(t, p)),
            'mean_relative_error': float(np.mean(np.abs(p - t) / (t + 1e-10))),
        }
    
    metrics['per_variable'] = {}
    
    T_pred = predictions_denorm[:, :, idx_T].numpy().flatten()
    T_target = targets_denorm[:, :, idx_T].numpy().flatten()
    metrics['per_variable']['T'] = compute_metrics_safe(T_pred, T_target)
    
    ne_pred = predictions_denorm[:, :, idx_ne].numpy().flatten()
    ne_target = targets_denorm[:, :, idx_ne].numpy().flatten()
    metrics['per_variable']['ne'] = compute_metrics_safe(ne_pred, ne_target)
    
    rho_pred = predictions_denorm[:, :, idx_rho].numpy().flatten()
    rho_target = targets_denorm[:, :, idx_rho].numpy().flatten()
    metrics['per_variable']['rho'] = compute_metrics_safe(rho_pred, rho_target)
    
    level_indices = list(range(idx_rho + 1, len(output_cols)))
    if level_indices:
        level_pred = predictions_denorm[:, :, level_indices].numpy().flatten()
        level_target = targets_denorm[:, :, level_indices].numpy().flatten()
        metrics['per_variable']['level_pops'] = compute_metrics_safe(level_pred, level_target)
    else:
        metrics['per_variable']['level_pops'] = {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'mean_relative_error': float('nan')}
    
    metrics['depth_resolved'] = {
        'T_mse': [],
        'ne_mse': [],
        'rho_mse': [],
    }
    
    for depth in range(50):
        T_pred_d = predictions_denorm[:, depth, idx_T].numpy()
        T_target_d = targets_denorm[:, depth, idx_T].numpy()
        mask = np.isfinite(T_pred_d) & np.isfinite(T_target_d)
        T_mse = mean_squared_error(T_target_d[mask], T_pred_d[mask]) if mask.any() else float('nan')
        
        ne_pred_d = predictions_denorm[:, depth, idx_ne].numpy()
        ne_target_d = targets_denorm[:, depth, idx_ne].numpy()
        mask = np.isfinite(ne_pred_d) & np.isfinite(ne_target_d)
        ne_mse = mean_squared_error(ne_target_d[mask], ne_pred_d[mask]) if mask.any() else float('nan')
        
        rho_pred_d = predictions_denorm[:, depth, idx_rho].numpy()
        rho_target_d = targets_denorm[:, depth, idx_rho].numpy()
        mask = np.isfinite(rho_pred_d) & np.isfinite(rho_target_d)
        rho_mse = mean_squared_error(rho_target_d[mask], rho_pred_d[mask]) if mask.any() else float('nan')
        
        metrics['depth_resolved']['T_mse'].append(float(T_mse))
        metrics['depth_resolved']['ne_mse'].append(float(ne_mse))
        metrics['depth_resolved']['rho_mse'].append(float(rho_mse))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        np.save(os.path.join(output_dir, 'predictions.npy'), predictions_denorm.numpy())
        np.save(os.path.join(output_dir, 'targets.npy'), targets_denorm.numpy())
        np.save(os.path.join(output_dir, 'inputs.npy'), inputs_denorm.numpy())
    
    return metrics, predictions_denorm, targets_denorm, inputs_denorm


def _round_to_dataset_grid(inputs):
    """将输入参数四舍五入到数据集网格"""
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
        teff, logg, mh = arr[i, 0], arr[i, 1], arr[i, 2]
        arr[i, 0], arr[i, 1], arr[i, 2] = round_to_dataset_grid(teff, logg, mh)
    
    return arr


def plot_results(predictions, targets, inputs, stats, output_dir, n_samples=5):
    """绘制评估结果图"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_cols = stats.get('output_cols', [f'col_{i}' for i in range(predictions.shape[-1])])
    try:
        idx_T = output_cols.index('T')
        idx_ne = output_cols.index('ne')
        idx_rho = output_cols.index('rho')
    except ValueError:
        idx_T, idx_ne, idx_rho = 0, 1, 2
    
    n_models = predictions.shape[0]
    
    np.random.seed(42)
    sample_indices = np.random.choice(n_models, min(n_samples, n_models), replace=False)

    if isinstance(inputs, torch.Tensor):
        inputs_np = inputs.numpy()
    else:
        inputs_np = np.array(inputs)

    if inputs_np[:, 0].max() <= 1.5 and inputs_np[:, 0].min() >= -1.5:
        inputs_np = denormalize_input_physical(inputs_np, stats, input_cols=['teff', 'logg', 'mh']).numpy()

    inputs_display = _round_to_dataset_grid(inputs_np)
    
    for i in range(min(3, len(inputs_display))):
        teff_val = float(inputs_display[i, 0])
        logg_val = float(inputs_display[i, 1])
        mh_val = float(inputs_display[i, 2])
        print(f"  Plot model {i+1}: Teff={teff_val:.0f}K, logg={logg_val:.2f}, mh={mh_val:+.2f}")
    
    for i, idx in enumerate(sample_indices):
        create_single_model_plot(predictions, targets, inputs_display, idx, output_dir, stats=stats)
    
    # 绘制多模型对比图（最多 3 个模型）
    multi_indices = sample_indices[:3].tolist()
    if len(multi_indices) >= 2:
        create_multi_model_comparison(predictions, targets, inputs_display, multi_indices, output_dir, stats=stats)
    
    # 额外绘制汇总图
    create_scatter_comparison(predictions, targets, output_dir, max_points=5000, stats=stats)
    create_error_by_depth(predictions, targets, output_dir, stats=stats)


EVAL_CONFIG = {
    'checkpoint': None,
    'output_dir': './results',
    'n_samples': 5,
    'random_seed': 42,
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
    
    stats_path = os.path.join(TRAIN_CONFIG['save_dir'], 'stats.json')
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
    
    # 打印关键指标
    print("\nEvaluation Metrics:")
    print(f"  Overall MSE: {metrics['overall']['mse']:.6e}")
    print(f"  Overall MAE: {metrics['overall']['mae']:.6e}")
    print(f"  Overall R2:  {metrics['overall']['r2']:.6f}")
    print(f"  T   -> MSE: {metrics['per_variable']['T']['mse']:.6e}, MAE: {metrics['per_variable']['T']['mae']:.6e}")
    print(f"  ne  -> MSE: {metrics['per_variable']['ne']['mse']:.6e}, MAE: {metrics['per_variable']['ne']['mae']:.6e}")
    print(f"  rho -> MSE: {metrics['per_variable']['rho']['mse']:.6e}, MAE: {metrics['per_variable']['rho']['mae']:.6e}")
    
    plot_results(predictions, targets, inputs, stats, output_dir, n_samples=n_samples)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
