import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 config=None, output_cols=None):
        self.device = device

        self.config = {
            'epochs': 1000,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'pct_start': 0.3,
            'scheduler': 'one_cycle',
            'early_stopping_patience': 150,
            'gradient_clip': 5.0,
            'save_dir': './results',
            'log_interval': 1,
            'use_amp': True,
            'compile_model': False,
        }
        if config:
            self.config.update(config)

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_cols = output_cols or []

        # 变量权重向量
        self.variable_weight_vec = None
        variable_weights = self.config.get('variable_weights', {})
        if self.output_cols and variable_weights:
            vw = []
            default_w = variable_weights.get('default', 1.0)
            for col in self.output_cols:
                w = variable_weights.get(col, default_w)
                vw.append(float(w))
            self.variable_weight_vec = torch.tensor(vw, dtype=torch.float32, device=device)

        # 自动混合精度
        self.use_amp = self.config.get('use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # 学习率调度器
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'one_cycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['epochs'],
                steps_per_epoch=len(train_loader),
                pct_start=self.config['pct_start'],
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', self.config['epochs'] // 3),
                gamma=self.config.get('gamma', 0.5)
            )

        # 追踪指标
        self.train_losses = []
        self.val_losses = []
        self.data_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        self.global_step = 0

        os.makedirs(self.config['save_dir'], exist_ok=True)

    def compute_loss(self, y_pred, y_true):
        """MSE+L1 损失，含 NaN 处理与按变量加权"""
        mask = torch.isfinite(y_true)
        diff = torch.where(mask, y_pred - y_true, torch.zeros_like(y_pred))

        if not mask.any():
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        valid_count = mask.sum(dim=(0, 1)).clamp(min=1.0)
        mse_per_col = (diff ** 2).sum(dim=(0, 1)) / valid_count
        l1_per_col = diff.abs().sum(dim=(0, 1)) / valid_count

        if self.variable_weight_vec is not None:
            w = self.variable_weight_vec.to(mse_per_col.device)
            mse_per_col = mse_per_col * w
            l1_per_col = l1_per_col * w

        mse = mse_per_col.mean()
        l1_loss = l1_per_col.mean()

        return mse + 0.1 * l1_loss

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = total_data = 0.0
        n_batches = 0
        n_skipped = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}', leave=False)

        for batch in pbar:
            x = batch['x'].to(self.device, non_blocking=True)
            y_true = batch['y'].to(self.device, non_blocking=True)

            if 'tau' in batch:
                tau = batch['tau'].to(self.device, non_blocking=True)
            else:
                tau = None

            if 'depth' in batch:
                depth = batch['depth'].to(self.device, non_blocking=True)
            else:
                depth = None

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    y_pred = self.model(x, tau=tau, depth=depth)
                    loss = self.compute_loss(y_pred, y_true)

                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue

                self.scaler.scale(loss).backward()

                if self.config['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                y_pred = self.model(x, tau=tau, depth=depth)
                loss = self.compute_loss(y_pred, y_true)

                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue

                loss.backward()

                if self.config['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )

                self.optimizer.step()

            if self.config.get('scheduler') == 'one_cycle':
                self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            total_data += loss.item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        if n_batches == 0:
            print(f"Warning: all {n_skipped} batches skipped due to NaN/Inf loss!")
            return {'total': float('inf'), 'data': float('inf')}

        if n_skipped > 0:
            print(f"Warning: skipped {n_skipped} batches due to NaN/Inf loss")

        return {
            'total': total_loss / n_batches,
            'data': total_data / n_batches,
        }

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = total_data = 0.0
        n_batches = 0
        n_skipped = 0

        for batch in self.val_loader:
            x = batch['x'].to(self.device, non_blocking=True)
            y_true = batch['y'].to(self.device, non_blocking=True)

            if 'tau' in batch:
                tau = batch['tau'].to(self.device, non_blocking=True)
            else:
                tau = None

            if 'depth' in batch:
                depth = batch['depth'].to(self.device, non_blocking=True)
            else:
                depth = None

            if self.use_amp:
                with autocast():
                    y_pred = self.model(x, tau=tau, depth=depth)
                    loss = self.compute_loss(y_pred, y_true)
            else:
                y_pred = self.model(x, tau=tau, depth=depth)
                loss = self.compute_loss(y_pred, y_true)

            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            total_loss += loss.item()
            total_data += loss.item()
            n_batches += 1

        if n_batches == 0:
            print(f"Warning: all validation batches skipped due to NaN/Inf!")
            return {'total': float('inf'), 'data': float('inf')}

        if n_skipped > 0:
            print(f"Warning: skipped {n_skipped} validation batches due to NaN/Inf loss")

        return {
            'total': total_loss / n_batches,
            'data': total_data / n_batches,
        }

    def train(self):
        """主训练循环"""
        print("="*70)
        print("TLUSTY NN Training")
        print("="*70)
        print(f"Device: {self.device}, AMP: {self.use_amp}")
        print(f"Total epochs: {self.config['epochs']}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        print("="*70)

        start_time = time.time()

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()

            if train_metrics['total'] == float('inf'):
                print(f"Warning: epoch {epoch+1} failed, skipping...")
                continue

            val_metrics = self.validate()

            if self.config.get('scheduler') != 'one_cycle':
                self.scheduler.step()

            self.train_losses.append(train_metrics['total'])
            self.val_losses.append(val_metrics['total'])
            self.data_losses.append(train_metrics['data'])
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            if (epoch + 1) % self.config['log_interval'] == 0:
                print(f"Epoch {epoch+1:4d}/{self.config['epochs']} | "
                      f"Train: {train_metrics['total']:.4e} | "
                      f"Val: {val_metrics['total']:.4e} | "
                      f"LR: {self.learning_rates[-1]:.2e}")

            if val_metrics['data'] < self.best_val_loss:
                self.best_val_loss = val_metrics['data']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"Best validation loss at epoch {epoch+1}: {val_metrics['data']:.4e}")
            else:
                self.patience_counter += 1

            if epoch > 50 and self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4e}")
        print(f"Total epochs trained: {self.current_epoch + 1}")
        print(f"Time: {elapsed/3600:.1f}h (avg {elapsed/(self.current_epoch+1):.1f}s/epoch)")
        print("="*70)

        self.plot_history()
        self.plot_loss_distribution()

        return self.best_val_loss

    def plot_loss_distribution(self):
        """绘制损失随 epoch 的散点连线图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = np.arange(1, len(self.train_losses) + 1)

        ax.plot(epochs, self.train_losses, 'o-', color='#2E86AB',
                linewidth=1.5, markersize=3, alpha=0.8, label='Train Loss')
        ax.plot(epochs, self.val_losses, 's-', color='#E94F37',
                linewidth=1.5, markersize=3, alpha=0.8, label='Val Loss')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Loss vs Epoch', fontweight='bold', pad=10)
        ax.set_yscale('log')
        ax.legend(loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='-', color='#E0E0E0')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], 'loss_distribution.pdf'), dpi=300)
        plt.close()

    def save_checkpoint(self, filename, epoch, metrics):
        """保存检查点"""
        path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config,
        }, path)

    def plot_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                                sharex=True,
                                gridspec_kw={'wspace': 0.3})

        axes[0].semilogy(self.train_losses, label='Train')
        axes[0].semilogy(self.val_losses, label='Val')
        axes[0].set_ylabel('Total Loss', fontsize=14)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].tick_params(axis='both', labelsize=11)

        axes[1].semilogy(self.data_losses)
        axes[1].set_ylabel('Data Loss', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].tick_params(axis='both', labelsize=11)

        axes[2].plot(self.learning_rates)
        axes[2].set_xlabel('Epoch', fontsize=14)
        axes[2].set_ylabel('Learning Rate', fontsize=14)
        axes[2].set_yscale('log')
        axes[2].tick_params(axis='both', labelsize=11)

        plt.savefig(os.path.join(self.config['save_dir'], 'training_history.pdf'), dpi=300)
        plt.close()
