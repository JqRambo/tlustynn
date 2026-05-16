"""
TLUSTY NN – Physics-Informed Neural Network for stellar atmosphere modelling.

Quick start
-----------
>>> from tlustynn import predict_atmosphere
>>> df, paths = predict_atmosphere(10000, 3.7, 0.0, output_dir='./predictions')
>>> print(df.head())
"""

from ._version import __version__
from .api import predict_atmosphere, TlustyAtmosphere

__all__ = [
    "__version__",
    "predict_atmosphere",
    "TlustyAtmosphere",
]
