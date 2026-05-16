import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

from tlustynn.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CSV_PATH, VARIABLE_WEIGHTS
from tlustynn.data_loader import load_and_preprocess_data, create_data_loaders
from tlustynn.model import create_model
from tlustynn.train import Trainer


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def save_stats(stats, save_dir):
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return float(obj)

    with open(os.path.join(save_dir, 'stats.json'), 'w') as f:
        json.dump(convert(stats), f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='TLUSTY NN Training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs:
        TRAIN_CONFIG['epochs'] = args.epochs

    print("="*70)
    print("TLUSTY NN Training")
    print("="*70)
    print(f"  Data: {CSV_PATH}")
    print(f"  Batch size: {DATA_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Hidden layers: {MODEL_CONFIG['hidden_layers']}")
    print(f"  Activation: {MODEL_CONFIG['activation']}")
    print(f"  Device: {TRAIN_CONFIG['device']}")
    print("="*70)

    set_seed(DATA_CONFIG['random_seed'])

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        if not hasattr(torch, 'compile'):
            TRAIN_CONFIG['compile_model'] = False

    os.makedirs(TRAIN_CONFIG['save_dir'], exist_ok=True)

    log_transform_cols = DATA_CONFIG.get('log_transform_cols')
    df, input_cols, output_cols, stats = load_and_preprocess_data(CSV_PATH, log_transform_cols=log_transform_cols)
    stats['input_cols'] = input_cols
    stats['output_cols'] = output_cols
    save_stats(stats, TRAIN_CONFIG['save_dir'])

    train_loader, val_loader, _ = create_data_loaders(
        df, input_cols, output_cols, stats,
        batch_size=DATA_CONFIG['batch_size'],
        test_ratio=DATA_CONFIG['test_ratio'],
        val_ratio=DATA_CONFIG['val_ratio'],
        random_seed=DATA_CONFIG['random_seed'],
        num_workers=DATA_CONFIG.get('num_workers', 4),
        log_transform_cols=log_transform_cols
    )
    print(f"  {len(train_loader)} train batches, {len(val_loader)} val batches")

    model = create_model(MODEL_CONFIG['model_type'], **{
        'input_dim': MODEL_CONFIG['input_dim'],
        'output_dim': len(output_cols),
        'n_depths': 50,
        'hidden_layers': MODEL_CONFIG['hidden_layers'],
        'activation': MODEL_CONFIG['activation'],
        'dropout': MODEL_CONFIG['dropout'],
        'use_fourier': MODEL_CONFIG['use_fourier'],
        'fourier_dim': MODEL_CONFIG['fourier_dim'],
        'fourier_scale': MODEL_CONFIG['fourier_scale'],
        'use_output_clamp': MODEL_CONFIG.get('use_output_clamp', False),
        'output_clamp_range': MODEL_CONFIG.get('output_clamp_range', 10.0),
        'use_checkpoint': MODEL_CONFIG.get('use_checkpoint', False),
    })

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params/1e6:.2f}M")

    if TRAIN_CONFIG.get('compile_model', False) and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    train_config = TRAIN_CONFIG.copy()
    train_config['variable_weights'] = VARIABLE_WEIGHTS

    trainer = Trainer(model, train_loader, val_loader,
                      device=TRAIN_CONFIG['device'], config=train_config,
                      output_cols=output_cols)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=TRAIN_CONFIG['device'])
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}")

    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_checkpoint('interrupted.pt', trainer.current_epoch, {})

    summary = {
        'best_val_loss': trainer.best_val_loss,
        'final_epoch': trainer.current_epoch,
        'n_params': n_params}
    with open(os.path.join(TRAIN_CONFIG['save_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
