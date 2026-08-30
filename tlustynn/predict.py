import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .model import create_model
from .data_loader import (normalize_input_physical, denormalize_input_physical,
                         round_to_dataset_grid, inverse_transform_output, inverse_transform_input)
from .config import MODEL_CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG, INPUT_COLS, AVG_MASS_SAVE_PATH, CHECKPOINT_DIR


class TlustyPredictor:
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                INFERENCE_CONFIG['checkpoint']
            )

        print(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=device)

        self.config = self.checkpoint.get('config', MODEL_CONFIG)

        stats_path = os.path.join(CHECKPOINT_DIR, 'stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
        else:
            print("Warning: stats.json not found, normalization may be incorrect.")
            self.stats = None

        self.model = self._create_model()

        state_dict = self.checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()

        # 加载物理单位的平均 mass 剖面
        self.avg_mass_physical = None
        mass_path = AVG_MASS_SAVE_PATH
        if os.path.exists(mass_path):
            self.avg_mass_physical = np.load(mass_path)
        elif self.stats and 'avg_mass_physical_path' in self.stats:
            if os.path.exists(self.stats['avg_mass_physical_path']):
                self.avg_mass_physical = np.load(self.stats['avg_mass_physical_path'])

    def _create_model(self):
        model_type = self.config.get('model_type', 'mlp')

        if self.stats and 'output_cols' in self.stats:
            output_dim = len(self.stats['output_cols'])
        else:
            output_dim = self.config.get('output_dim', 59)

        kwargs = {
            'input_dim': self.config.get('input_dim', 3),
            'output_dim': output_dim,
            'n_depths': self.config.get('n_depths', 50),
        }

        if model_type == 'mlp':
            kwargs.update({
                'hidden_layers': self.config.get('hidden_layers', [1024, 2048, 2048, 2048, 1024, 512]),
                'activation': self.config.get('activation', 'silu'),
                'use_fourier': self.config.get('use_fourier', True),
                'fourier_dim': self.config.get('fourier_dim', 256),
                'fourier_scale': self.config.get('fourier_scale', 4.0),
                'use_output_clamp': self.config.get('use_output_clamp', False),
                'output_clamp_range': self.config.get('output_clamp_range', 10.0),
            })
        elif model_type == 'attention':
            kwargs.update({
                'hidden_dim': self.config.get('attention_hidden_dim', 512),
                'n_heads': self.config.get('attention_heads', 8),
                'n_layers': self.config.get('attention_layers', 4),
            })
        elif model_type == 'lstm':
            kwargs.update({
                'hidden_dim': self.config.get('lstm_hidden_dim', 256),
                'n_layers': self.config.get('lstm_layers', 6),
            })

        return create_model(model_type, **kwargs)

    def _normalize_input(self, x):
        """将物理单位的输入归一化到 [-1, 1]"""
        if isinstance(x, (list, tuple)):
            x = np.array(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_norm = np.zeros_like(x)
        stellar_cols = ['teff', 'logg', 'log_he_h']

        if self.stats and 'input' in self.stats:
            normalization = self.stats.get('normalization', 'minmax')
            for i, col in enumerate(stellar_cols):
                if col in self.stats['input']:
                    if normalization == 'minmax':
                        xmin = self.stats['input'][col]['min']
                        xmax = self.stats['input'][col]['max']
                        if xmax > xmin:
                            x_norm[:, i] = 2.0 * (x[:, i] - xmin) / (xmax - xmin) - 1.0
                        else:
                            x_norm[:, i] = 0.0
                    else:
                        mean = self.stats['input'][col]['mean']
                        std = self.stats['input'][col]['std']
                        if std > 0:
                            x_norm[:, i] = (x[:, i] - mean) / std
                        else:
                            x_norm[:, i] = 0.0
        else:
            from .config import INPUT_RANGES
            for i, col in enumerate(stellar_cols):
                if col in INPUT_RANGES:
                    xmin = INPUT_RANGES[col]['min']
                    xmax = INPUT_RANGES[col]['max']
                    x_norm[:, i] = 2.0 * (x[:, i] - xmin) / (xmax - xmin) - 1.0
                else:
                    x_norm[:, i] = x[:, i]

        x_norm = np.clip(x_norm, -1.0, 1.0)
        return x_norm

    def predict(self, teff, logg, log_he_h, mass=None):
        """预测单个或多个恒星大气模型

        输入物理单位的 teff, logg, log_he_h
        可选传入 mass [n_models, 50] 或 [50]（归一化到 [-1,1]），否则自动使用训练集平均物理 mass 剖面
        返回包含 50 层大气参数的预测结果（T, ne, rho, 55 能级布居数）
        """
        single_input = np.isscalar(teff)

        if single_input:
            teff = [teff]
            logg = [logg]
            log_he_h = [log_he_h]

        teff_arr = np.atleast_1d(teff)
        logg_arr = np.atleast_1d(logg)
        log_he_h_arr = np.atleast_1d(log_he_h)
        n_models = len(teff_arr)

        warnings_list = []

        x_physical = np.column_stack([teff_arr, logg_arr, log_he_h_arr]).astype(np.float32)
        x_norm = self._normalize_input(x_physical)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(self.device)

        # 准备 mass 输入
        if mass is not None:
            mass_arr = np.atleast_2d(mass).astype(np.float32)
            if mass_arr.shape[0] == 1 and n_models > 1:
                mass_arr = np.repeat(mass_arr, n_models, axis=0)
            mass_tensor = torch.tensor(mass_arr, dtype=torch.float32).to(self.device)
        else:
            if self.avg_mass_physical is not None and self.stats:
                mass_stats = self.stats.get('mass_stats', {})
                if mass_stats:
                    ymin = mass_stats['min']
                    ymax = mass_stats['max']
                    mass_norm = 2.0 * (self.avg_mass_physical - ymin) / (ymax - ymin) - 1.0
                    mass_tensor = torch.tensor(mass_norm, dtype=torch.float32).unsqueeze(0).expand(n_models, -1).to(self.device)
                else:
                    mass_tensor = torch.tensor(self.avg_mass_physical, dtype=torch.float32).unsqueeze(0).expand(n_models, -1).to(self.device)
            else:
                avg_mass = self.stats.get('avg_mass_norm') if self.stats else None
                if avg_mass is not None:
                    mass_tensor = torch.tensor(avg_mass, dtype=torch.float32).unsqueeze(0).expand(n_models, -1).to(self.device)
                else:
                    mass_tensor = None

        with torch.no_grad():
            y_pred_norm = self.model(x_tensor, mass=mass_tensor)

        if self.stats:
            output_cols = self.stats.get('output_cols', [f'col_{i}' for i in range(y_pred_norm.shape[-1])])
            y_pred = inverse_transform_output(y_pred_norm.cpu(), self.stats, output_cols)
        else:
            y_pred = y_pred_norm.cpu()

        result = {
            'input': x_physical,
            'prediction': y_pred.numpy(),
            'warnings': warnings_list,
        }

        return result

    def predict_to_fort7(self, teff, logg, log_he_h, output_path=None):
        """预测并保存为类似 fort.7 格式的 CSV"""
        result = self.predict(teff, logg, log_he_h)
        y_pred = result['prediction'][0]

        depth_indices = np.arange(1, 51)

        if self.stats and 'output_cols' in self.stats:
            output_cols = self.stats['output_cols']
        else:
            output_cols = [f'col_{i}' for i in range(y_pred.shape[1])]

        df = pd.DataFrame(y_pred, columns=output_cols)
        df.insert(0, 'depth_index', depth_indices)
        df['teff'] = teff
        df['logg'] = logg
        df['log_he_h'] = log_he_h

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Saved fort.7 format to {output_path}")

        return df

    def generate_grid_predictions(self, teff_range, logg_range, log_he_h_values,
                                   output_dir=None, plot=True):
        """对恒星参数网格进行预测"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        teff_vals = np.linspace(teff_range[0], teff_range[1], teff_range[2])
        logg_vals = np.linspace(logg_range[0], logg_range[1], logg_range[2])

        results = []

        for log_he_h in log_he_h_values:
            print(f"Processing log(n_He/n_H)={log_he_h}...")

            teff_grid, logg_grid = np.meshgrid(teff_vals, logg_vals)
            teff_flat = teff_grid.flatten()
            logg_flat = logg_grid.flatten()
            log_he_h_flat = np.full_like(teff_flat, log_he_h)

            batch_size = 32
            all_predictions = []

            for i in range(0, len(teff_flat), batch_size):
                batch_teff = teff_flat[i:i+batch_size]
                batch_logg = logg_flat[i:i+batch_size]
                batch_log_he_h = log_he_h_flat[i:i+batch_size]

                result = self.predict(batch_teff, batch_logg, batch_log_he_h)
                all_predictions.append(result['prediction'])

            predictions = np.concatenate(all_predictions, axis=0)

            if output_dir:
                np.save(
                    os.path.join(output_dir, f'predictions_log_he_h{log_he_h:.1f}.npy'),
                    predictions
                )

                grid_info = {
                    'teff_range': teff_range,
                    'logg_range': logg_range,
                    'log_he_h': log_he_h,
                    'teff_vals': teff_vals.tolist(),
                    'logg_vals': logg_vals.tolist(),
                }
                with open(os.path.join(output_dir, f'grid_info_log_he_h{log_he_h:.1f}.json'), 'w') as f:
                    json.dump(grid_info, f, indent=2)

            results.append({
                'log_he_h': log_he_h,
                'predictions': predictions,
                'teff_grid': teff_grid,
                'logg_grid': logg_grid,
            })

            if plot and output_dir:
                self._plot_grid_results(results[-1], output_dir)

        return results

    def _plot_grid_results(self, result, output_dir):
        """绘制网格预测结果"""
        log_he_h = result['log_he_h']
        predictions = result['predictions']
        teff_grid = result['teff_grid']
        logg_grid = result['logg_grid']

        n_logg, n_teff = teff_grid.shape

        if self.stats and 'output_cols' in self.stats:
            output_cols = self.stats['output_cols']
            idx_T = output_cols.index('T') if 'T' in output_cols else 1
        else:
            idx_T = 1

        depths_to_plot = [0, 24, 49]
        depth_names = ['Surface', 'Middle', 'Deep']

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, depth_idx, depth_name in zip(axes, depths_to_plot, depth_names):
            T_depth = predictions[:, depth_idx, idx_T].reshape(n_logg, n_teff)

            im = ax.contourf(teff_grid, logg_grid, T_depth, levels=20, cmap='hot')
            ax.set_xlabel('Teff [K]')
            ax.set_ylabel('logg')
            ax.set_title(f'Temperature ({depth_name}, depth={depth_idx+1})')
            plt.colorbar(im, ax=ax, label='T [K]')

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'Temperature_grid_log_he_h{log_he_h:.1f}.png'),
            dpi=150
        )
        plt.close()

    def compare_with_data(self, df_test, n_samples=5, output_dir=None):
        """将预测结果与测试数据进行对比"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        stellar_cols = ['teff', 'logg', 'log_he_h']
        df_test['model_id'] = df_test[stellar_cols].astype(str).agg('_'.join, axis=1)
        model_ids = df_test['model_id'].unique()

        np.random.seed(42)
        sample_ids = np.random.choice(model_ids, min(n_samples, len(model_ids)), replace=False)

        results = []

        for i, model_id in enumerate(sample_ids):
            model_data = df_test[df_test['model_id'] == model_id].sort_values('M')

            teff = model_data['teff'].iloc[0]
            logg = model_data['logg'].iloc[0]
            log_he_h = model_data['log_he_h'].iloc[0]

            output_cols = [c for c in model_data.columns if c not in INPUT_COLS + ['depth_index', 'model_id']]
            if 'M' in output_cols:
                output_cols.remove('M')
            y_true = model_data[output_cols].values

            # 使用真实 mass 进行更准确的对比
            mass_true_norm = None
            if self.stats and 'avg_mass_norm' in self.stats:
                mass_stats = self.stats.get('mass_stats', {})
                if mass_stats:
                    mass_phys = model_data['M'].values.astype(np.float32)
                    ymin = mass_stats['min']
                    ymax = mass_stats['max']
                    mass_true_norm = 2.0 * (mass_phys - ymin) / (ymax - ymin) - 1.0

            result = self.predict(teff, logg, log_he_h, mass=mass_true_norm)
            y_pred = result['prediction'][0]

            mse = np.mean((y_pred - y_true) ** 2)
            mae = np.mean(np.abs(y_pred - y_true))

            results.append({
                'teff': teff,
                'logg': logg,
                'log_he_h': log_he_h,
                'mse': mse,
                'mae': mae,
                'y_true': y_true,
                'y_pred': y_pred,
            })

            if output_dir:
                self._plot_comparison(results[-1], i, output_dir)

        all_mse = [r['mse'] for r in results]
        all_mae = [r['mae'] for r in results]

        print(f"\nComparison summary ({n_samples} samples):")
        print(f"  Mean MSE: {np.mean(all_mse):.6e}")
        print(f"  Mean MAE: {np.mean(all_mae):.6e}")

        return {
            'results': results,
            'mean_mse': np.mean(all_mse),
            'mean_mae': np.mean(all_mae),
        }

    def _plot_comparison(self, result, idx, output_dir):
        """绘制预测与真实数据的对比图"""
        depth = np.arange(1, 51)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        titles = ['Temperature [K]', 'Electron Density [cm^-3]', 'Mass Density [g/cm^3]']
        scales = ['linear', 'log', 'log']

        output_cols = self.stats.get('output_cols', [f'col_{i}' for i in range(result['y_true'].shape[1])]) if self.stats else [f'col_{i}' for i in range(result['y_true'].shape[1])]
        try:
            var_indices = [output_cols.index('T'), output_cols.index('ne'), output_cols.index('rho')]
        except ValueError:
            var_indices = [0, 1, 2]

        for ax, var_idx, title, scale in zip(axes[:3], var_indices, titles, scales):
            y_true = result['y_true'][:, var_idx]
            y_pred = result['y_pred'][:, var_idx]

            ax.plot(depth, y_true, 'b-', label='True', linewidth=2)
            ax.plot(depth, y_pred, 'r--', label='Predicted', linewidth=2)
            ax.set_xlabel('Depth Index')
            ax.set_ylabel(title)
            ax.set_yscale(scale)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[3].text(0.5, 0.5,
                    f"Teff={result['teff']:.0f}\n"
                    f"logg={result['logg']:.2f}\n"
                    f"log(n_He/n_H)={result['log_he_h']:.2f}\n"
                    f"MSE={result['mse']:.6e}",
                    transform=axes[3].transAxes,
                    ha='center', va='center',
                    fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[3].axis('off')

        axes[4].axis('off')
        axes[5].axis('off')

        plt.suptitle(f'Model {idx+1} Comparison')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'comparison_{idx+1}.png'),
            dpi=150
        )
        plt.close()


PREDICT_CONFIG = {
    'checkpoint': None,
    'teff': 10000,
    'logg': 4.0,
    'log_he_h': 0.0,
    'output': None,
    'plot': True,
    'output_dir': './predictions',
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description='TLUSTY NN Prediction')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--teff', type=float, default=None, help='Effective temperature [K]')
    parser.add_argument('--logg', type=float, default=None, help='Surface gravity [log10(cm/s^2)]')
    parser.add_argument('--log_he_h', type=float, default=None, help='log(n_He/n_H) [dex]')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--plot', action='store_true', help='Generate plot')
    parser.add_argument('--no-plot', action='store_true', help='Do not generate plot')

    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint is not None else PREDICT_CONFIG['checkpoint']
    teff = args.teff if args.teff is not None else PREDICT_CONFIG['teff']
    logg = args.logg if args.logg is not None else PREDICT_CONFIG['logg']
    log_he_h = args.log_he_h if args.log_he_h is not None else PREDICT_CONFIG['log_he_h']
    output = args.output if args.output is not None else PREDICT_CONFIG['output']
    plot = PREDICT_CONFIG['plot']
    if args.plot:
        plot = True
    if args.no_plot:
        plot = False
    output_dir = PREDICT_CONFIG['output_dir']

    print("="*70)
    print("TLUSTY NN Prediction")
    print("="*70)
    print(f"Checkpoint: {checkpoint or 'default'}")
    print(f"  Teff: {teff} K")
    print(f"  logg: {logg}")
    print(f"  [M/H]: {log_he_h}")
    print("="*70)

    predictor = TlustyPredictor(checkpoint_path=checkpoint)

    result = predictor.predict(teff, logg, log_he_h)

    print(f"\nTeff={teff}, logg={logg}, log(n_He/n_H)={log_he_h}")
    print("Prediction completed.")

    predictor.predict_to_fort7(teff, logg, log_he_h, output)

    if plot:
        os.makedirs(output_dir, exist_ok=True)

        y_pred = result['prediction'][0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        depth = np.arange(1, 51)
        titles = ['T [K]', 'ne [cm^-3]', 'rho [g/cm^3]']
        scales = ['linear', 'log', 'log']

        output_cols = predictor.stats.get('output_cols', [f'col_{i}' for i in range(y_pred.shape[1])]) if predictor.stats else [f'col_{i}' for i in range(y_pred.shape[1])]
        try:
            var_indices = [output_cols.index('T'), output_cols.index('ne'), output_cols.index('rho')]
        except ValueError:
            var_indices = [0, 1, 2]

        for ax, var_idx, title, scale in zip(axes, var_indices, titles, scales):
            ax.plot(depth, y_pred[:, var_idx], 'b-', linewidth=2)
            ax.set_xlabel('Depth Index')
            ax.set_ylabel(title)
            ax.set_yscale(scale)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Teff={teff}, logg={logg}, log(n_He/n_H)={log_he_h}')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'prediction_T{teff}_g{logg}_m{log_he_h}.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.show()


if __name__ == '__main__':
    main()
