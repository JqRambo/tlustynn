import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch

try:
    from .data_loader import round_to_dataset_grid
except ImportError:
    def round_to_dataset_grid(teff, logg, mh):
        teff_r = np.round(teff / 500.0) * 500.0
        logg_r = np.round(logg / 0.1) * 0.1
        mh_r = np.round(mh / 0.5) * 0.5
        return teff_r, logg_r, mh_r

COLORS = {
    'true': '#2E86AB',      
    'predicted': '#E94F37', 
    'grid': '#E0E0E0',     
    'text': '#333333',    
    'accent': '#F18F01', 
}

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300


def _get_var_indices(stats):
    """Get T, ne, rho indices from stats or return defaults."""
    if stats and 'output_cols' in stats:
        output_cols = stats['output_cols']
        try:
            return output_cols.index('T'), output_cols.index('ne'), output_cols.index('rho')
        except ValueError:
            pass
    return 0, 1, 2 


def _format_model_title(teff, logg, mh):
    teff_r, logg_r, mh_r = round_to_dataset_grid(teff, logg, mh)
    return f"Teff={teff_r:.0f}K, logg={logg_r:.2f}, [M/H]={mh_r:+.2f}"



def create_single_model_plot(predictions, targets, inputs, idx, output_dir, stats=None):

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    if isinstance(inputs, torch.Tensor):
        inputs_np = inputs.numpy()
    else:
        inputs_np = np.array(inputs)
    
    teff = float(inputs_np[idx, 0])
    logg = float(inputs_np[idx, 1])
    mh = float(inputs_np[idx, 2])
    
    depth = np.arange(1, 51)
    
    idx_T, idx_ne, idx_rho = _get_var_indices(stats)
    
    teff_r, logg_r, mh_r = round_to_dataset_grid(teff, logg, mh)
    info_text = (
        f"Teff = {teff_r:,.0f} K\n"
        f"log g = {logg_r:.2f}\n"
        f"log(nHe/nH) = {mh_r:+.2f}")
    
    if abs(teff - teff_r) > 0.1 or abs(logg - logg_r) > 0.01 or abs(mh - mh_r) > 0.01:
        info_text += f"\n\n(Original: {teff:.0f}, {logg:.2f}, {mh:+.2f})"
    
    y_true_T = targets[idx, :, idx_T].numpy()
    y_pred_T = predictions[idx, :, idx_T].numpy()
    valid_mask_T = np.isfinite(y_true_T) & np.isfinite(y_pred_T)
    
    ax[0].plot(depth[valid_mask_T], y_true_T[valid_mask_T], color=COLORS['true'], linewidth=2.5, linestyle='--', label='TLUSTY', alpha=0.9)
    ax[0].plot(depth[valid_mask_T], y_pred_T[valid_mask_T], color=COLORS['predicted'], linewidth=2.5, linestyle='--', label='NN', alpha=0.9)
    ax[0].set_xlabel('Depth Layer', fontsize=12)
    ax[0].set_ylabel('T [K]', fontsize=12)
    ax[0].set_yscale('linear')
    ax[0].set_xlim(1, 50)
    ax[0].minorticks_on()  
    ax[0].tick_params(axis='both', which='major', direction='in', length=6, width=1.2, colors='black', labelcolor='black', labelsize=10)
    ax[0].tick_params(axis='both', which='minor', length=3, width=0.8, colors='gray')
    ax[0].text(0.05, 0.80, info_text,transform=ax[0].transAxes, ha='left', va='bottom',fontsize=10,fontfamily='monospace',linespacing=1.5)
    ax[0].legend(loc='upper right', framealpha=0.95, edgecolor='gray', frameon=False)
    

    y_true_ne = targets[idx, :, idx_ne].numpy()
    y_pred_ne = predictions[idx, :, idx_ne].numpy()
    valid_mask_ne = np.isfinite(y_true_ne) & np.isfinite(y_pred_ne)
    
    ax[1].plot(depth[valid_mask_ne], y_true_ne[valid_mask_ne], color=COLORS['true'], linewidth=2.5, linestyle='--', label='TLUSTY', alpha=0.9)
    ax[1].plot(depth[valid_mask_ne], y_pred_ne[valid_mask_ne], color=COLORS['predicted'], linewidth=2.5, linestyle='--', label='NN', alpha=0.9)
    ax[1].set_xlabel('Depth Layer', fontsize=12)
    ax[1].set_ylabel('nₑ [cm⁻³]', fontsize=12)
    ax[1].set_yscale('log')
    ax[1].set_xlim(1, 50)
    
    ax[1].minorticks_on()
    ax[1].tick_params(axis='both',which='major',direction='in',length=6,width=1.2,colors='black',labelsize=10)
    ax[1].tick_params(axis='both',which='minor',length=3,width=0.8,colors='gray')
    ax[1].legend(loc='upper right', framealpha=0.95, edgecolor='gray', frameon=False)
    
    y_true_rho = targets[idx, :, idx_rho].numpy()
    y_pred_rho = predictions[idx, :, idx_rho].numpy()
    valid_mask_rho = np.isfinite(y_true_rho) & np.isfinite(y_pred_rho)
    
    ax[2].plot(depth[valid_mask_rho], y_true_rho[valid_mask_rho], color=COLORS['true'], linewidth=2.5, linestyle='--', label='TLUSTY', alpha=0.9)
    ax[2].plot(depth[valid_mask_rho], y_pred_rho[valid_mask_rho], color=COLORS['predicted'], linewidth=2.5, linestyle='--', label='NN', alpha=0.9)
    ax[2].set_xlabel('Depth Layer',  fontsize=12)
    ax[2].set_ylabel('ρ [g/cm³]',  fontsize=12)
    ax[2].set_yscale('log')
    ax[2].set_xlim(1, 50)
    
    ax[2].minorticks_on()
    ax[2].tick_params(axis='both',which='major',direction='in',length=6,width=1.2,colors='black',labelcolor='black',labelsize=10)
    ax[2].tick_params(axis='both',which='minor',length=3,width=0.8,colors='gray')
    ax[2].legend(loc='upper right', framealpha=0.95, edgecolor='gray', frameon=False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_{idx+1:02d}_Teff{teff_r:.0f}_logg{logg_r:.1f}.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()



def create_multi_model_comparison(predictions, targets, inputs, indices, output_dir, stats=None):

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.numpy()
    
    indices = list(indices)[:3]
    n_models = len(indices)
    if n_models < 2:
        return
    
    fig = plt.figure(figsize=(16, 4 * n_models))
    depth = np.arange(1, 51)
    idx_T, idx_ne, idx_rho = _get_var_indices(stats)
    
    # ============================================
    # Model 1
    # ============================================
    idx_1 = indices[0]
    teff_1 = inputs[idx_1, 0].item()
    logg_1 = inputs[idx_1, 1].item()
    mh_1 = inputs[idx_1, 2].item()
    teff_r_1, logg_r_1, mh_r_1 = round_to_dataset_grid(teff_1, logg_1, mh_1)
    
    # Model 1 - Temperature
    ax_1_T = plt.subplot(n_models, 3, 1)
    y_true_1_T = targets[idx_1, :, idx_T].numpy()
    y_pred_1_T = predictions[idx_1, :, idx_T].numpy()
    valid_mask_1_T = np.isfinite(y_true_1_T) & np.isfinite(y_pred_1_T)
    ax_1_T.plot(depth[valid_mask_1_T], y_true_1_T[valid_mask_1_T], color=COLORS['true'], linewidth=2, label='True')
    ax_1_T.plot(depth[valid_mask_1_T], y_pred_1_T[valid_mask_1_T], color=COLORS['predicted'], linewidth=2, linestyle='--', label='Predicted')
    ax_1_T.set_xlabel('Depth Layer')
    ax_1_T.set_ylabel('Temperature [K]')
    ax_1_T.set_yscale('linear')
    ax_1_T.text(0.04, 0.90, _format_model_title(teff_1, logg_1, mh_1), transform=ax_1_T.transAxes, ha='left', va='top', fontsize=12)
    ax_1_T.legend(loc='upper right')
    
    # Model 1 - Electron Density
    ax_1_ne = plt.subplot(n_models, 3, 2)
    y_true_1_ne = targets[idx_1, :, idx_ne].numpy()
    y_pred_1_ne = predictions[idx_1, :, idx_ne].numpy()
    valid_mask_1_ne = np.isfinite(y_true_1_ne) & np.isfinite(y_pred_1_ne)
    ax_1_ne.plot(depth[valid_mask_1_ne], y_true_1_ne[valid_mask_1_ne], color=COLORS['true'], linewidth=2)
    ax_1_ne.plot(depth[valid_mask_1_ne], y_pred_1_ne[valid_mask_1_ne], color=COLORS['predicted'], linewidth=2, linestyle='--')
    ax_1_ne.set_xlabel('Depth Layer')
    ax_1_ne.set_ylabel('nₑ [cm⁻³]')
    ax_1_ne.set_yscale('log')
    
    # Model 1 - Density
    ax_1_rho = plt.subplot(n_models, 3, 3)
    y_true_1_rho = targets[idx_1, :, idx_rho].numpy()
    y_pred_1_rho = predictions[idx_1, :, idx_rho].numpy()
    valid_mask_1_rho = np.isfinite(y_true_1_rho) & np.isfinite(y_pred_1_rho)
    ax_1_rho.plot(depth[valid_mask_1_rho], y_true_1_rho[valid_mask_1_rho], color=COLORS['true'], linewidth=2)
    ax_1_rho.plot(depth[valid_mask_1_rho], y_pred_1_rho[valid_mask_1_rho], color=COLORS['predicted'], linewidth=2, linestyle='--')
    ax_1_rho.set_xlabel('Depth Layer')
    ax_1_rho.set_ylabel('ρ [g/cm³]')
    ax_1_rho.set_yscale('log')
    
    # ============================================
    # Model 2
    # ============================================
    idx_2 = indices[1]
    teff_2 = inputs[idx_2, 0].item()
    logg_2 = inputs[idx_2, 1].item()
    mh_2 = inputs[idx_2, 2].item()
    teff_r_2, logg_r_2, mh_r_2 = round_to_dataset_grid(teff_2, logg_2, mh_2)
    
    # Model 2 - Temperature
    ax_2_T = plt.subplot(n_models, 3, 4)
    y_true_2_T = targets[idx_2, :, idx_T].numpy()
    y_pred_2_T = predictions[idx_2, :, idx_T].numpy()
    valid_mask_2_T = np.isfinite(y_true_2_T) & np.isfinite(y_pred_2_T)
    ax_2_T.plot(depth[valid_mask_2_T], y_true_2_T[valid_mask_2_T], color=COLORS['true'], linewidth=2)
    ax_2_T.plot(depth[valid_mask_2_T], y_pred_2_T[valid_mask_2_T], color=COLORS['predicted'], linewidth=2, linestyle='--')
    ax_2_T.set_xlabel('Depth Layer')
    ax_2_T.set_ylabel('Temperature [K]')
    ax_2_T.set_yscale('linear')
    ax_2_T.text(0.04, 0.90, _format_model_title(teff_2, logg_2, mh_2), transform=ax_2_T.transAxes, ha='left', va='top', fontsize=12)
    
    # Model 2 - Electron Density
    ax_2_ne = plt.subplot(n_models, 3, 5)
    y_true_2_ne = targets[idx_2, :, idx_ne].numpy()
    y_pred_2_ne = predictions[idx_2, :, idx_ne].numpy()
    valid_mask_2_ne = np.isfinite(y_true_2_ne) & np.isfinite(y_pred_2_ne)
    ax_2_ne.plot(depth[valid_mask_2_ne], y_true_2_ne[valid_mask_2_ne], color=COLORS['true'], linewidth=2)
    ax_2_ne.plot(depth[valid_mask_2_ne], y_pred_2_ne[valid_mask_2_ne], color=COLORS['predicted'], linewidth=2, linestyle='--')
    ax_2_ne.set_xlabel('Depth Layer')
    ax_2_ne.set_ylabel('nₑ [cm⁻³]')
    ax_2_ne.set_yscale('log')
    
    # Model 2 - Density
    ax_2_rho = plt.subplot(n_models, 3, 6)
    y_true_2_rho = targets[idx_2, :, idx_rho].numpy()
    y_pred_2_rho = predictions[idx_2, :, idx_rho].numpy()
    valid_mask_2_rho = np.isfinite(y_true_2_rho) & np.isfinite(y_pred_2_rho)
    ax_2_rho.plot(depth[valid_mask_2_rho], y_true_2_rho[valid_mask_2_rho], color=COLORS['true'], linewidth=2)
    ax_2_rho.plot(depth[valid_mask_2_rho], y_pred_2_rho[valid_mask_2_rho], color=COLORS['predicted'], linewidth=2, linestyle='--')
    ax_2_rho.set_xlabel('Depth Layer')
    ax_2_rho.set_ylabel('ρ [g/cm³]')
    ax_2_rho.set_yscale('log')
    
    # ============================================
    # Model 3 (可选)
    # ============================================
    if n_models >= 3:
        idx_3 = indices[2]
        teff_3 = inputs[idx_3, 0].item()
        logg_3 = inputs[idx_3, 1].item()
        mh_3 = inputs[idx_3, 2].item()
        teff_r_3, logg_r_3, mh_r_3 = round_to_dataset_grid(teff_3, logg_3, mh_3)
        
        # Model 3 - Temperature
        ax_3_T = plt.subplot(n_models, 3, 7)
        y_true_3_T = targets[idx_3, :, idx_T].numpy()
        y_pred_3_T = predictions[idx_3, :, idx_T].numpy()
        valid_mask_3_T = np.isfinite(y_true_3_T) & np.isfinite(y_pred_3_T)
        ax_3_T.plot(depth[valid_mask_3_T], y_true_3_T[valid_mask_3_T], color=COLORS['true'], linewidth=2)
        ax_3_T.plot(depth[valid_mask_3_T], y_pred_3_T[valid_mask_3_T], color=COLORS['predicted'], linewidth=2, linestyle='--')
        ax_3_T.set_xlabel('Depth Layer')
        ax_3_T.set_ylabel('Temperature [K]')
        ax_3_T.set_yscale('linear')
        ax_3_T.text(0.04, 0.90, _format_model_title(teff_3, logg_3, mh_3), transform=ax_3_T.transAxes, ha='left', va='top', fontsize=12)
        
        # Model 3 - Electron Density
        ax_3_ne = plt.subplot(n_models, 3, 8)
        y_true_3_ne = targets[idx_3, :, idx_ne].numpy()
        y_pred_3_ne = predictions[idx_3, :, idx_ne].numpy()
        valid_mask_3_ne = np.isfinite(y_true_3_ne) & np.isfinite(y_pred_3_ne)
        ax_3_ne.plot(depth[valid_mask_3_ne], y_true_3_ne[valid_mask_3_ne], color=COLORS['true'], linewidth=2)
        ax_3_ne.plot(depth[valid_mask_3_ne], y_pred_3_ne[valid_mask_3_ne], color=COLORS['predicted'], linewidth=2, linestyle='--')
        ax_3_ne.set_xlabel('Depth Layer')
        ax_3_ne.set_ylabel('nₑ [cm⁻³]')
        ax_3_ne.set_yscale('log')
        
        # Model 3 - Density
        ax_3_rho = plt.subplot(n_models, 3, 9)
        y_true_3_rho = targets[idx_3, :, idx_rho].numpy()
        y_pred_3_rho = predictions[idx_3, :, idx_rho].numpy()
        valid_mask_3_rho = np.isfinite(y_true_3_rho) & np.isfinite(y_pred_3_rho)
        ax_3_rho.plot(depth[valid_mask_3_rho], y_true_3_rho[valid_mask_3_rho], color=COLORS['true'], linewidth=2)
        ax_3_rho.plot(depth[valid_mask_3_rho], y_pred_3_rho[valid_mask_3_rho], color=COLORS['predicted'], linewidth=2, linestyle='--')
        ax_3_rho.set_xlabel('Depth Layer')
        ax_3_rho.set_ylabel('ρ [g/cm³]')
        ax_3_rho.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()



def create_scatter_comparison(predictions, targets, output_dir, max_points=5000, stats=None):
    """
    Create scatter plots comparing predictions vs targets (2x2 layout).
    
    Args:
        predictions: [n_models, 50, n_outputs] predicted values
        targets: [n_models, 50, n_outputs] true values
        output_dir: directory to save the plot
        max_points: maximum number of points to plot (for performance)
        stats: optional statistics for labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    idx_T, idx_ne, idx_rho = _get_var_indices(stats)
    
    # --- Temperature ---
    ax_T = axes[0]
    y_true_T = targets[:, :, idx_T].numpy().flatten()
    y_pred_T = predictions[:, :, idx_T].numpy().flatten()
    valid_mask_T = np.isfinite(y_true_T) & np.isfinite(y_pred_T)
    y_true_T = y_true_T[valid_mask_T]
    y_pred_T = y_pred_T[valid_mask_T]
    if len(y_true_T) > max_points:
        indices = np.random.choice(len(y_true_T), max_points, replace=False)
        y_true_T = y_true_T[indices]
        y_pred_T = y_pred_T[indices]
    ax_T.scatter(y_true_T, y_pred_T, alpha=0.9, s=2, c=COLORS['accent'], edgecolors='none')
    min_val_T = min(y_true_T.min(), y_pred_T.min())
    max_val_T = max(y_true_T.max(), y_pred_T.max())
    ax_T.plot([min_val_T, max_val_T], [min_val_T, max_val_T],'k--', linewidth=1.5, alpha=0.7)
    ax_T.set_xlabel('Tlusty', fontweight='bold')
    ax_T.set_ylabel('NN', fontweight='bold')
    ax_T.text(0.05, 0.9, 'T',transform=ax_T.transAxes,fontsize=10,fontweight='bold',
            color='black',           
            alpha=1.0,               
            backgroundcolor='none',  
            bbox=None,              
            horizontalalignment='left',   
            verticalalignment='bottom',   
            rotation=0,             
            linespacing=1.0)  
    # 修改：统一使用 log-log
    ax_T.set_xscale('log')
    ax_T.set_yscale('log')
    
    # --- Electron Density ---
    ax_ne = axes[1]
    y_true_ne = targets[:, :, idx_ne].numpy().flatten()
    y_pred_ne = predictions[:, :, idx_ne].numpy().flatten()
    valid_mask_ne = np.isfinite(y_true_ne) & np.isfinite(y_pred_ne)
    y_true_ne = y_true_ne[valid_mask_ne]
    y_pred_ne = y_pred_ne[valid_mask_ne]
    if len(y_true_ne) > max_points:
        indices = np.random.choice(len(y_true_ne), max_points, replace=False)
        y_true_ne = y_true_ne[indices]
        y_pred_ne = y_pred_ne[indices]
    ax_ne.scatter(y_true_ne, y_pred_ne, alpha=0.9, s=2, c=COLORS['accent'], edgecolors='none')
    min_val_ne = min(y_true_ne.min(), y_pred_ne.min())
    max_val_ne = max(y_true_ne.max(), y_pred_ne.max())
    ax_ne.plot([min_val_ne, max_val_ne], [min_val_ne, max_val_ne],'k--', linewidth=1.5, alpha=0.7)
    ax_ne.set_xlabel('Tlusty', fontweight='bold')
    ax_ne.set_ylabel('NN', fontweight='bold')
    ax_ne.text(0.05, 0.9, 'Electron Density',transform=ax_ne.transAxes,fontsize=10,fontweight='bold',
            color='black',           
            alpha=1.0,               
            backgroundcolor='none',  
            bbox=None,              
            horizontalalignment='left',   
            verticalalignment='bottom',   
            rotation=0,             
            linespacing=1.0)  
    # 修改：统一使用 log-log
    ax_ne.set_xscale('log')
    ax_ne.set_yscale('log')
    
    # --- Density ---
    ax_rho = axes[2]
    y_true_rho = targets[:, :, idx_rho].numpy().flatten()
    y_pred_rho = predictions[:, :, idx_rho].numpy().flatten()
    valid_mask_rho = np.isfinite(y_true_rho) & np.isfinite(y_pred_rho)
    y_true_rho = y_true_rho[valid_mask_rho]
    y_pred_rho = y_pred_rho[valid_mask_rho]
    if len(y_true_rho) > max_points:
        indices = np.random.choice(len(y_true_rho), max_points, replace=False)
        y_true_rho = y_true_rho[indices]
        y_pred_rho = y_pred_rho[indices]
    ax_rho.scatter(y_true_rho, y_pred_rho, alpha=0.9, s=2, c=COLORS['accent'], edgecolors='none')
    min_val_rho = min(y_true_rho.min(), y_pred_rho.min())
    max_val_rho = max(y_true_rho.max(), y_pred_rho.max())
    ax_rho.plot([min_val_rho, max_val_rho], [min_val_rho, max_val_rho],'k--', linewidth=1.5, alpha=0.7)
    ax_rho.set_xlabel('Tlusty', fontweight='bold')
    ax_rho.set_ylabel('NN', fontweight='bold')
    ax_rho.text(0.05, 0.9, 'Density',transform=ax_rho.transAxes,fontsize=10,fontweight='bold',
            color='black',           
            alpha=1.0,               
            backgroundcolor='none',  
            bbox=None,              
            horizontalalignment='left',   
            verticalalignment='bottom',   
            rotation=0,             
            linespacing=1.0)  
    # 修改：统一使用 log-log
    ax_rho.set_xscale('log')
    ax_rho.set_yscale('log')
    
    # Level populations (if available)
    ax = axes[3]
    # tau=0, T=1, ne=2, rho=3, levels 从 4 开始
    # tau 已移除，能级布居数从 rho 之后开始（idx_rho + 1 = 3）
    y_true = targets[:, :, 3:].numpy().flatten()
    y_pred = predictions[:, :, 3:].numpy().flatten()
    
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) > max_points:
        indices = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    ax.scatter(y_true, y_pred, alpha=0.9, s=2, c=COLORS['accent'], edgecolors='none')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Tlusty', fontweight='bold')
    ax.set_ylabel('NN', fontweight='bold')
    ax.text(0.05, 0.9, 'Level Populations',transform=ax.transAxes,fontsize=10,fontweight='bold',
            color='black',           
            alpha=1.0,               
            backgroundcolor='none',  
            bbox=None,              
            horizontalalignment='left',   
            verticalalignment='bottom',   
            rotation=0,             
            linespacing=1.0)             
    # 修改：统一使用 log-log
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # plt.suptitle('Prediction Accuracy (Scatter Plots)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_error_by_depth(predictions, targets, output_dir, stats=None):

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    
    depth = np.arange(1, 51)
    
    idx_T, idx_ne, idx_rho = _get_var_indices(stats)
    
    rmse_T = []
    for d in range(50):
        y_true = targets[:, d, idx_T].numpy()
        y_pred = predictions[:, d, idx_T].numpy()
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid_mask.sum() > 0:
            mse = np.mean((y_true[valid_mask] - y_pred[valid_mask])**2)
            rmse_T.append(np.sqrt(mse))
        else:
            rmse_T.append(np.nan)
    rmse_T = np.array(rmse_T)
    valid_depths_T = np.isfinite(rmse_T)

    ax[0].fill_between(depth[valid_depths_T], 0, rmse_T[valid_depths_T], alpha=0.3, color=COLORS['true'])
    ax[0].plot(depth[valid_depths_T], rmse_T[valid_depths_T], color=COLORS['true'], linewidth=2.5)
    ax[0].set_xlabel('Depth Layer', fontweight='bold')
    ax[0].set_ylabel('RMSE [K]', fontweight='bold')
    ax[0].text(0.05, 0.85, 'Temperature', transform=ax[0].transAxes, fontweight='bold', fontsize=12)
    ax[0].set_xlim(1, 50)
    
    # Electron Density subplot
    rmse_ne = []
    for d in range(50):
        y_true = targets[:, d, idx_ne].numpy()
        y_pred = predictions[:, d, idx_ne].numpy()
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid_mask.sum() > 0:
            mse = np.mean((y_true[valid_mask] - y_pred[valid_mask])**2)
            rmse_ne.append(np.sqrt(mse))
        else:
            rmse_ne.append(np.nan)
    rmse_ne = np.array(rmse_ne)
    valid_depths_ne = np.isfinite(rmse_ne)
    ax[1].fill_between(depth[valid_depths_ne], 0, rmse_ne[valid_depths_ne], alpha=0.3, color=COLORS['true'])
    ax[1].plot(depth[valid_depths_ne], rmse_ne[valid_depths_ne], color=COLORS['true'], linewidth=2.5)
    ax[1].set_xlabel('Depth Layer', fontweight='bold')
    ax[1].set_ylabel('RMSE', fontweight='bold')
    ax[1].text(0.05, 0.85, 'Electron Density', transform=ax[1].transAxes, fontweight='bold', fontsize=12)
    ax[1].set_xlim(1, 50)
    
    rmse_rho = []
    for d in range(50):
        y_true = targets[:, d, idx_rho].numpy()
        y_pred = predictions[:, d, idx_rho].numpy()
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid_mask.sum() > 0:
            mse = np.mean((y_true[valid_mask] - y_pred[valid_mask])**2)
            rmse_rho.append(np.sqrt(mse))
        else:
            rmse_rho.append(np.nan)
    rmse_rho = np.array(rmse_rho)
    valid_depths_rho = np.isfinite(rmse_rho)
    ax[2].fill_between(depth[valid_depths_rho], 0, rmse_rho[valid_depths_rho], alpha=0.3, color=COLORS['true'])
    ax[2].plot(depth[valid_depths_rho], rmse_rho[valid_depths_rho], color=COLORS['true'], linewidth=2.5)
    ax[2].set_xlabel('Depth Layer', fontweight='bold')
    ax[2].set_ylabel('RMSE', fontweight='bold')
    ax[2].text(0.05, 0.85, 'Density', transform=ax[2].transAxes, fontweight='bold', fontsize=12)
    ax[2].set_xlim(1, 50)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_by_depth.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_single_model(predictions, targets, inputs, idx, output_dir):
    create_single_model_plot(predictions, targets, inputs, idx, output_dir)

def plot_scatter(predictions, targets, output_dir):
    create_scatter_comparison(predictions, targets, output_dir)

def plot_error_depth(predictions, targets, output_dir):
    create_error_by_depth(predictions, targets, output_dir)
