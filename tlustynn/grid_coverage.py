from pathlib import Path

import numpy as np
import pandas as pd


class GridCoverage:
    def __init__(self, csv_path=None):
        if csv_path is None:
            base_dir = Path(__file__).resolve().parent
            csv_path = base_dir / 'hhe.csv'
            if not csv_path.exists():
                csv_path = base_dir / 'checkpoints' / 'grid_coverage.csv'
        df = pd.read_csv(csv_path, usecols=['teff', 'logg', 'log_he_h']).drop_duplicates()
        self.model_info = df.groupby(['teff', 'logg', 'log_he_h']).size().index
        self.available = set(self.model_info)
        self.teff_nodes = np.sort(df['teff'].unique())
        self.logg_nodes = np.sort(df['logg'].unique())
        self.he_nodes = np.sort(df['log_he_h'].unique())
        self.teff_index = {value: i for i, value in enumerate(self.teff_nodes)}
        self.logg_index = {value: i for i, value in enumerate(self.logg_nodes)}
        self.he_index = {value: i for i, value in enumerate(self.he_nodes)}
        self.bounds = {
            'teff': (self.teff_nodes[0], self.teff_nodes[-1]),
            'logg': (self.logg_nodes[0], self.logg_nodes[-1]),
            'log_he_h': (self.he_nodes[0], self.he_nodes[-1]),
        }

    def missing_neighbors(self, teff, logg, log_he_h):
        outside = not (
            self.bounds['teff'][0] <= teff <= self.bounds['teff'][1] and
            self.bounds['logg'][0] <= logg <= self.bounds['logg'][1] and
            self.bounds['log_he_h'][0] <= log_he_h <= self.bounds['log_he_h'][1])
        if outside:
            return 1.0, True

        center = (
            int(np.argmin(np.abs(self.teff_nodes - teff))),
            int(np.argmin(np.abs(self.logg_nodes - logg))),
            int(np.argmin(np.abs(self.he_nodes - log_he_h))),
        )
        missing = 0
        total = 0
        for i in range(center[0] - 1, center[0] + 2):
            for j in range(center[1] - 1, center[1] + 2):
                for k in range(center[2] - 1, center[2] + 2):
                    if not (0 <= i < len(self.teff_nodes) and
                            0 <= j < len(self.logg_nodes) and
                            0 <= k < len(self.he_nodes)):
                        continue
                    total += 1
                    model_id = (self.teff_nodes[i], self.logg_nodes[j], self.he_nodes[k])
                    if model_id not in self.available:
                        missing += 1
        return missing / total if total else 1.0, False

    def warnings(self, inputs):
        messages = []
        for teff, logg, log_he_h in np.atleast_2d(inputs):
            fraction, outside = self.missing_neighbors(teff, logg, log_he_h)
            if outside:
                messages.append(
                    f'Input ({teff}, {logg}, {log_he_h}) lies outside the converged-grid bounds; '
                    'prediction is extrapolation and has no accuracy guarantee.')
            elif fraction > 0:
                messages.append(
                    f'Input ({teff}, {logg}, {log_he_h}) has {fraction:.1%} missing models '
                    'in its local 3x3x3 neighborhood; prediction is near a grid gap.')
        return messages
