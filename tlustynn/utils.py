import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch

try:
    from .data_loader import round_to_dataset_grid
except ImportError:
    def round_to_dataset_grid(teff, logg, log_he_h):
        return (np.round(teff / 500.0) * 500.0,
                np.round(logg / 0.1) * 0.1,
                np.round(log_he_h / 0.5) * 0.5)


PROFILE_VARIABLES = [
    ('T', r'$T$ [K]', 'linear'),
    ('ne', r'$n_e$ [cm$^{-3}$]', 'log'),
    ('rho', r'$\rho$ [g cm$^{-3}$]', 'log'),
    ('M', r'$M$ [g cm$^{-2}$]', 'log'),
]


def _as_numpy(values):
    return values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)


def _get_var_indices(stats):
    output_cols = stats['output_cols']
    return {name: output_cols.index(name) for name, _, _ in PROFILE_VARIABLES}


def _level_indices(stats):
    return [i for i, col in enumerate(stats['output_cols']) if col.startswith('level_')]


def _style_axis(ax, label_size=16):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=7,
                   width=1.2, colors='black', labelsize=label_size)
    ax.tick_params(axis='both', which='minor', direction='in', length=4,
                   width=1.0, colors='black')


def _plot_profile(ax, predictions, targets, model_idx, variable_idx, scale,
                  label=False, tick_size=16):
    depth = np.arange(1, 51)
    y_true = _as_numpy(targets[model_idx, :, variable_idx])
    y_pred = _as_numpy(predictions[model_idx, :, variable_idx])
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    ax.plot(depth[valid], y_true[valid], color='dodgerblue', linewidth=2.2,
            label='TLUSTY' if label else None)
    ax.plot(depth[valid], y_pred[valid], color='deeppink', linewidth=2.2,
            linestyle='--', label='NN' if label else None)
    ax.set_yscale(scale)
    ax.set_xlim(1, 50)
    _style_axis(ax, tick_size)


def create_single_model_plot(predictions, targets, inputs, idx, output_dir, stats=None):
    inputs_np = _as_numpy(inputs)
    teff, logg, log_he_h = map(float, inputs_np[idx, :3])
    teff_r, logg_r, log_he_h_r = round_to_dataset_grid(teff, logg, log_he_h)
    var_indices = _get_var_indices(stats)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    for j, (name, ylabel, scale) in enumerate(PROFILE_VARIABLES):
        _plot_profile(axes[j], predictions, targets, idx, var_indices[name], scale,
                      label=(j == 0))
        axes[j].set_xlabel('Depth Layer', fontsize=18)
        axes[j].set_ylabel(ylabel, fontsize=18)
    axes[0].legend(loc='best', frameon=False, prop={'size': 16})

    fig.tight_layout()
    path = (f'{output_dir}/model_{idx + 1:02d}_Teff{teff_r:.0f}_logg{logg_r:.1f}'
            f'_log(nHe_nH)_{log_he_h_r:.3f}.pdf')
    fig.savefig(path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_multi_model_comparison(predictions, targets, inputs, indices, output_dir, stats=None):
    inputs_np = _as_numpy(inputs)
    indices = list(indices)[:3]
    if len(indices) < 2:
        return

    var_indices = _get_var_indices(stats)
    fig, axes = plt.subplots(len(indices), 4, figsize=(22, 4.7 * len(indices)),
                             squeeze=False)
    for row, idx in enumerate(indices):
        teff, logg, log_he_h = map(float, inputs_np[idx, :3])
        teff_r, logg_r, log_he_h_r = round_to_dataset_grid(teff, logg, log_he_h)
        info = (rf'$T_{{\rm eff}}={teff_r:.0f}$' + '\n' +
                rf'$\log g={logg_r:.2f}$' + '\n' +
                rf'$\log(n_{{\rm He}}/n_{{\rm H}})={log_he_h_r:+.2f}$')
        for col, (name, ylabel, scale) in enumerate(PROFILE_VARIABLES):
            ax = axes[row, col]
            _plot_profile(ax, predictions, targets, idx, var_indices[name], scale,
                          label=(row == 0 and col == 0), tick_size=20)
            ax.set_ylabel(ylabel, fontsize=22)
            if row == len(indices) - 1:
                ax.set_xlabel('Depth Layer', fontsize=22)
        axes[row, 0].text(0.04, 0.96, info, transform=axes[row, 0].transAxes,
                          ha='left', va='top', fontsize=20, linespacing=1.4)
    axes[0, 0].legend(loc='lower right', frameon=False, prop={'size': 20})

    fig.tight_layout()
    fig.savefig(f'{output_dir}/multi_model_comparison.pdf', dpi=400,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _residual_summary(residual):
    median = np.nanmedian(residual, axis=0)
    lower = np.nanpercentile(residual, 16, axis=0)
    upper = np.nanpercentile(residual, 84, axis=0)
    return median, lower, upper


def _log_residual(pred, true):
    pred = _as_numpy(pred)
    true = _as_numpy(true)
    valid = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    residual = np.full(pred.shape, np.nan, dtype=np.float32)
    residual[valid] = np.log10(pred[valid]) - np.log10(true[valid])
    return residual


def create_scatter_comparison(predictions, targets, output_dir, max_points=5000, stats=None):
    var_indices = _get_var_indices(stats)
    level_indices = _level_indices(stats)
    fig = plt.figure(figsize=(22, 10))
    grid = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.15],
                    hspace=0.38, wspace=0.32)

    for col, (name, label, _) in enumerate(PROFILE_VARIABLES):
        ax = fig.add_subplot(grid[0, col])
        idx = var_indices[name]
        if name == 'T':
            residual = _as_numpy(predictions[:, :, idx]) - _as_numpy(targets[:, :, idx])
            ylabel = 'Residual (NN - TLUSTY) [K]'
        else:
            residual = _log_residual(predictions[:, :, idx], targets[:, :, idx])
            ylabel = r'Residual $\log_{10}({\rm NN}/{\rm TLUSTY})$ [dex]'
        median, lower, upper = _residual_summary(residual)
        depth = np.arange(1, 51)
        ax.fill_between(depth, lower, upper, color='dodgerblue', alpha=0.22,
                        linewidth=0, label='16–84 percentile')
        ax.plot(depth, median, color='navy', linewidth=1.8, label='Median')
        ax.axhline(0.0, color='black', linestyle='--', linewidth=0.9)
        ax.set_xlim(1, 50)
        ax.set_xlabel('Depth Layer', fontsize=17)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.text(0.04, 0.95, label.split(' [')[0], transform=ax.transAxes, fontsize=18,
                ha='left', va='top')
        _style_axis(ax, 15)
        if col == 0:
            ax.legend(frameon=False, fontsize=14, loc='upper right')

    ax_lp = fig.add_subplot(grid[1, :])
    medians, lowers, uppers = [], [], []
    for idx in level_indices:
        residual = _log_residual(predictions[:, :, idx], targets[:, :, idx]).ravel()
        residual = residual[np.isfinite(residual)]
        if residual.size:
            medians.append(np.median(residual))
            lowers.append(np.percentile(residual, 16))
            uppers.append(np.percentile(residual, 84))
        else:
            medians.append(np.nan)
            lowers.append(np.nan)
            uppers.append(np.nan)
    levels = np.arange(1, len(level_indices) + 1)
    ax_lp.fill_between(levels, lowers, uppers, color='dodgerblue', alpha=0.22,
                       linewidth=0, label='16–84 percentile')
    ax_lp.plot(levels, medians, color='navy', linewidth=1.8, label='Median')
    ax_lp.axhline(0.0, color='black', linestyle='--', linewidth=0.9)
    ax_lp.set_xlim(1, 55)
    ax_lp.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    ax_lp.set_xlabel('Level Population Index (LP1–LP55)', fontsize=18)
    ax_lp.set_ylabel(r'Residual $\log_{10}({\rm NN}/{\rm TLUSTY})$ [dex]', fontsize=17)
    ax_lp.text(0.02, 0.95, 'LP', transform=ax_lp.transAxes,
               fontsize=18, ha='left', va='top')
    ax_lp.legend(frameon=False, fontsize=15, loc='upper right')
    _style_axis(ax_lp, 16)

    fig.savefig(f'{output_dir}/scatter_comparison.pdf', dpi=400,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_diagonal_comparison(predictions, targets, output_dir, max_points=5000, stats=None):
    var_indices = _get_var_indices(stats)
    level_indices = _level_indices(stats)
    quantities = [(name, label) for name, label, _ in PROFILE_VARIABLES]
    quantities.append(('LP', r'$LP$'))
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    for ax, (name, label) in zip(axes.flat, quantities):
        if name == 'LP':
            pred = _as_numpy(predictions[:, :, level_indices]).ravel()
            true = _as_numpy(targets[:, :, level_indices]).ravel()
        else:
            idx = var_indices[name]
            pred = _as_numpy(predictions[:, :, idx]).ravel()
            true = _as_numpy(targets[:, :, idx]).ravel()

        valid = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
        pred = pred[valid]
        true = true[valid]
        if pred.size > max_points:
            selected = rng.choice(pred.size, max_points, replace=False)
            pred = pred[selected]
            true = true[selected]

        ax.scatter(true, pred, alpha=0.6, s=4, c='lime', edgecolors='none',
                   rasterized=True)
        lower = min(true.min(), pred.min())
        upper = max(true.max(), pred.max())
        ax.plot([lower, upper], [lower, upper], 'k--', linewidth=1.5,
                alpha=0.8, label=r'$y=x$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('TLUSTY', fontsize=22)
        ax.set_ylabel('NN', fontsize=22)
        ax.text(0.04, 0.95, label.split(' [')[0], transform=ax.transAxes, fontsize=26,
                ha='left', va='top')
        ax.legend(loc='upper right', frameon=False, fontsize=20)
        _style_axis(ax, 20)

    for ax in axes.flat:
        ax.set_box_aspect(1)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.30)
    fig.savefig(f'{output_dir}/scatter_diagonal.pdf', dpi=400,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_error_by_depth(predictions, targets, output_dir, stats=None):
    var_indices = _get_var_indices(stats)
    depth = np.arange(1, 51)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    for ax, (name, label, _) in zip(axes, PROFILE_VARIABLES):
        idx = var_indices[name]
        pred = _as_numpy(predictions[:, :, idx]).astype(np.float64)
        true = _as_numpy(targets[:, :, idx]).astype(np.float64)
        rmse = []
        for d in range(50):
            valid = np.isfinite(pred[:, d]) & np.isfinite(true[:, d])
            rmse.append(np.sqrt(np.mean((pred[valid, d] - true[valid, d]) ** 2))
                        if valid.any() else np.nan)
        rmse = np.asarray(rmse)
        valid = np.isfinite(rmse)
        ax.fill_between(depth[valid], 0, rmse[valid], color='dodgerblue', alpha=0.2)
        ax.plot(depth[valid], rmse[valid], color='navy', linewidth=1.8)
        ax.set_xlim(1, 50)
        ax.set_xlabel('Depth Layer', fontsize=22)
        ax.set_ylabel('RMSE', fontsize=22)
        ax.text(0.04, 0.95, label.split(' [')[0], transform=ax.transAxes, fontsize=28,
                ha='left', va='top')
        _style_axis(ax, 20)
        ax.yaxis.get_offset_text().set_fontsize(20)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/error_by_depth.pdf', dpi=400,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_single_model(predictions, targets, inputs, idx, output_dir, stats=None):
    create_single_model_plot(predictions, targets, inputs, idx, output_dir, stats)


def plot_scatter(predictions, targets, output_dir, stats=None):
    create_scatter_comparison(predictions, targets, output_dir, stats=stats)


def plot_error_depth(predictions, targets, output_dir, stats=None):
    create_error_by_depth(predictions, targets, output_dir, stats)
