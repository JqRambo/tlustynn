"""
High-level API for TLUSTY NN atmosphere prediction.

Usage:
    from tlustynn import predict_atmosphere

    df = predict_atmosphere(teff=10000, logg=3.7, mh=0.0)
    # Returns a pandas DataFrame with columns:
    # teff, logg, mh, tau, T, ne, rho, level_1 ... level_55
"""

import os
import numpy as np
import pandas as pd

from .predict import TlustyPredictor
from .read_write_tlusty import write_tlusty_model7


# Singleton predictor instance (lazy initialization)
_predictor = None


def _get_predictor():
    """Get or create the default TlustyPredictor."""
    global _predictor
    if _predictor is None:
        _predictor = TlustyPredictor()
    return _predictor


def predict_atmosphere(teff, logg, mh, output_dir=None, filename=None, save_7=True):
    """Predict a single stellar atmosphere model and optionally save to CSV / TLUSTY .7.

    The output CSV follows the same column order as ``hhe.csv``:
    ``teff, logg, mh, tau, T, ne, rho, level_1, ..., level_55``.
    Each model contains 50 depth rows.

    The TLUSTY .7 file contains the raw atmospheric structure (tau + 58 parameters
    per depth) in the standard TLUSTY format, exactly like ``FF.7``.

    Parameters
    ----------
    teff : float
        Effective temperature [K].
    logg : float
        Surface gravity (log10 of cm s^-2).
    mh : float
        Metallicity [dex].
    output_dir : str, optional
        Directory where the files will be written. If ``None``, the DataFrame
        is returned but no file is written.
    filename : str, optional
        Explicit file name stem (without extension). If ``None``, the files
        are named ``{teff}_{logg}_{mh}.csv`` and ``{teff}_{logg}_{mh}.7``.
    save_7 : bool, default True
        Whether to also write a TLUSTY-format ``.7`` file when ``output_dir``
        is provided.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 50 rows (one per atmospheric depth) and columns
        matching the original ``hhe.csv`` format.
    dict or None
        Dictionary with keys ``'csv'`` and ``'7'`` containing the absolute
        paths to the saved files if ``output_dir`` is given, otherwise ``None``.
    """
    predictor = _get_predictor()
    result = predictor.predict(teff, logg, mh)
    y_pred = result['prediction'][0]  # [50, n_outputs]

    # Retrieve output column names from training stats
    if predictor.stats and 'output_cols' in predictor.stats:
        output_cols = predictor.stats['output_cols']
    else:
        output_cols = [f'col_{i}' for i in range(y_pred.shape[1])]

    # Build DataFrame from prediction
    df = pd.DataFrame(y_pred, columns=output_cols)

    # Insert tau (physical units) – taken from the average tau profile.
    if predictor.avg_tau_physical is not None:
        if predictor.stats and 'tau' in predictor.stats.get('log_transform_cols', []):
            tau_vals = 10 ** predictor.avg_tau_physical
        else:
            tau_vals = predictor.avg_tau_physical
    else:
        tau_vals = np.zeros(50, dtype=np.float32)
    df.insert(0, 'tau', tau_vals)

    # Insert stellar parameters (replicated for every depth row)
    df.insert(0, 'mh', float(mh))
    df.insert(0, 'logg', float(logg))
    df.insert(0, 'teff', float(teff))

    # Save to file(s) if requested
    filepaths = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"{teff}_{logg}_{mh}"

        csv_name = filename if filename.endswith('.csv') else f"{filename}.csv"
        csv_path = os.path.join(os.path.abspath(output_dir), csv_name)
        df.to_csv(csv_path, index=False)
        filepaths = {'csv': csv_path}

        if save_7:
            # Build TLUSTY .7 format DataFrame (depth_index, tau, T, ne, rho, level_1..level_55)
            df_7 = df[['tau'] + output_cols].copy()
            df_7.insert(0, 'depth_index', range(1, 51))
            n_depth = 50
            n_params = len(output_cols)  # typically 58 = T + ne + rho + 55 levels

            seven_name = filename.replace('.csv', '.7')
            if not seven_name.endswith('.7'):
                seven_name = f"{seven_name}.7"
            seven_path = os.path.join(os.path.abspath(output_dir), seven_name)

            model_data = {
                'n_depth': n_depth,
                'n_params': n_params,
                'dataframe': df_7,
            }
            write_tlusty_model7(seven_path, model_data)
            filepaths['7'] = seven_path

    return df, filepaths


class TlustyAtmosphere:
    """Convenient wrapper around :class:`TlustyPredictor`.

    Example
    -------
    >>> atm = TlustyAtmosphere()
    >>> df, paths = atm.predict(10000, 3.7, 0.0, output_dir='./predictions')
    """

    def __init__(self, checkpoint_path=None, device=None):
        self.predictor = TlustyPredictor(
            checkpoint_path=checkpoint_path, device=device
        )

    def predict(self, teff, logg, mh, output_dir=None, filename=None, save_7=True):
        """Same interface as :func:`predict_atmosphere`."""
        result = self.predictor.predict(teff, logg, mh)
        y_pred = result['prediction'][0]

        if self.predictor.stats and 'output_cols' in self.predictor.stats:
            output_cols = self.predictor.stats['output_cols']
        else:
            output_cols = [f'col_{i}' for i in range(y_pred.shape[1])]

        df = pd.DataFrame(y_pred, columns=output_cols)

        if self.predictor.avg_tau_physical is not None:
            if self.predictor.stats and 'tau' in self.predictor.stats.get('log_transform_cols', []):
                tau_vals = 10 ** self.predictor.avg_tau_physical
            else:
                tau_vals = self.predictor.avg_tau_physical
        else:
            tau_vals = np.zeros(50, dtype=np.float32)
        df.insert(0, 'tau', tau_vals)

        df.insert(0, 'mh', float(mh))
        df.insert(0, 'logg', float(logg))
        df.insert(0, 'teff', float(teff))

        filepaths = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                filename = f"{teff}_{logg}_{mh}"

            csv_name = filename if filename.endswith('.csv') else f"{filename}.csv"
            csv_path = os.path.join(os.path.abspath(output_dir), csv_name)
            df.to_csv(csv_path, index=False)
            filepaths = {'csv': csv_path}

            if save_7:
                df_7 = df[['tau'] + output_cols].copy()
                df_7.insert(0, 'depth_index', range(1, 51))
                n_depth = 50
                n_params = len(output_cols)

                seven_name = filename.replace('.csv', '.7')
                if not seven_name.endswith('.7'):
                    seven_name = f"{seven_name}.7"
                seven_path = os.path.join(os.path.abspath(output_dir), seven_name)

                model_data = {
                    'n_depth': n_depth,
                    'n_params': n_params,
                    'dataframe': df_7,
                }
                write_tlusty_model7(seven_path, model_data)
                filepaths['7'] = seven_path

        return df, filepaths
