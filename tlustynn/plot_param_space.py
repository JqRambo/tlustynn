import json
import os

import matplotlib.pyplot as plt
import numpy as np

NN_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(NN_DIR, 'results')
QUANTITIES = [
    ('T', r'$T$'),
    ('ne', r'$n_e$'),
    ('rho', r'$\rho$'),
    ('M', r'$M$'),
    ('LP', r'$LP$'),
    ('Mean', 'Mean'),
]


def _mean_log_error(pred, true):
    n_models = pred.shape[0]
    valid = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    safe_pred = np.where(valid, pred, 1.0)
    safe_true = np.where(valid, true, 1.0)
    residual = np.where(valid, np.abs(np.log10(safe_pred) - np.log10(safe_true)), np.nan)
    return np.nanmean(residual.reshape(n_models, -1), axis=1)


def load_mae():
    pred = np.load(os.path.join(OUT_DIR, 'predictions.npy'))
    true = np.load(os.path.join(OUT_DIR, 'targets.npy'))
    inputs = np.load(os.path.join(OUT_DIR, 'inputs.npy'))
    with open(os.path.join(NN_DIR, 'checkpoints', 'stats.json')) as f:
        cols = json.load(f)['output_cols']

    mae = [_mean_log_error(pred[:, :, cols.index(name)], true[:, :, cols.index(name)])
           for name in ['T', 'ne', 'rho', 'M']]
    levels = [i for i, col in enumerate(cols) if col.startswith('level_')]
    mae.append(_mean_log_error(pred[:, :, levels], true[:, :, levels]))
    mae.append(np.nanmean(np.stack(mae, axis=1), axis=1))
    return inputs, np.stack(mae, axis=1)


def plot_param_space(inputs, mae):
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    for ax, (_, label), j in zip(axes.flat, QUANTITIES, range(len(QUANTITIES))):
        finite = np.isfinite(mae[:, j])
        vmax = float(np.nanpercentile(mae[:, j], 95)) if finite.any() else 1.0
        if vmax <= 0:
            vmax = 1.0
        sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=mae[:, j], cmap='turbo',
                        vmin=0, vmax=vmax, s=5, rasterized=True)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel(r'$T_{\rm eff}$ [K]', fontsize=15)
        ax.set_ylabel(r'$\log g$', fontsize=15)
        ax.tick_params(axis='both', which='major', direction='in', length=6,
                       width=1.1, labelsize=13)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=3,
                       width=0.9)
        ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=17,
                va='top', ha='left')
        colorbar = fig.colorbar(sc, ax=ax)
        colorbar.set_label(r'MAE [dex]', fontsize=14)
        colorbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    output = os.path.join(OUT_DIR, 'NN_param_space.pdf')
    fig.savefig(output, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {output}')


def plot_param_space_heatmap(inputs, mae, bins=(90, 38)):
    teff_edges = np.linspace(inputs[:, 0].min(), inputs[:, 0].max(), bins[0] + 1)
    logg_edges = np.linspace(inputs[:, 1].min(), inputs[:, 1].max(), bins[1] + 1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    for ax, (_, label), j in zip(axes.flat, QUANTITIES, range(len(QUANTITIES))):
        finite = np.isfinite(mae[:, j])
        vmax = float(np.nanpercentile(mae[:, j], 95)) if finite.any() else 1.0
        if vmax <= 0:
            vmax = 1.0
        sums, _, _ = np.histogram2d(inputs[finite, 0], inputs[finite, 1],
                                    bins=[teff_edges, logg_edges],
                                    weights=mae[finite, j])
        counts, _, _ = np.histogram2d(inputs[finite, 0], inputs[finite, 1],
                                      bins=[teff_edges, logg_edges])
        with np.errstate(invalid='ignore', divide='ignore'):
            heatmap = np.ma.masked_invalid(sums / counts)
        image = ax.pcolormesh(teff_edges, logg_edges, heatmap.T, cmap='turbo',
                              vmin=0, vmax=vmax, shading='auto', rasterized=True)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel(r'$T_{\rm eff}$ [K]', fontsize=15)
        ax.set_ylabel(r'$\log g$', fontsize=15)
        ax.tick_params(axis='both', which='major', direction='in', length=6,
                       width=1.1, labelsize=13)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=3,
                       width=0.9)
        ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=17,
                va='top', ha='left')
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(r'MAE [dex]', fontsize=14)
        colorbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    output = os.path.join(OUT_DIR, 'NN_param_space_heatmap.pdf')
    fig.savefig(output, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {output}')


def main():
    inputs, mae = load_mae()
    plot_param_space(inputs, mae)
    plot_param_space_heatmap(inputs, mae)


if __name__ == '__main__':
    main()
