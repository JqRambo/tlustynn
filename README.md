# Teddy Is All You Need


## TLUSTY NN

Neural Network for fast prediction of TLUSTY stellar atmosphere models.

Given three stellar parameters — effective temperature (Teff), surface gravity (logg) and helium abundance (log(n_He / n_H)) — the network predicts the full 50-layer atmospheric structure (temperature, electron density, mass density, and 55 level populations) in a fraction of a second.

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/JqRambo/tlusty-nn.git
cd tlusty-nn
```

### Important Notes

Please note that this code has not been peer-reviewed.  

> If you intend to use it, please contact the author, Dr. Qi Jia (jq.physics@hotmail.com).  

> The pretrained model weight (`best_model.pt`, ~1 GB) exceeds GitHub's normal file-size limit (100 MB).
  Please reach out to Dr. Jia Qi (jq.physics@hotmail.com) to request access.


### Install into your Python environment

```bash
pip install .
```

After installation the package is available in `site-packages` and can be imported from anywhere:

```python
import tlustynn
```

---

## 🚀 Quick Start

### Predict a single atmosphere model

```python
from tlustynn import predict_atmosphere

# Predict and save as CSV
df, csv_path = predict_atmosphere(
    teff=10000,   # Effective temperature [K]
    logg=3.7,     # Surface gravity [log10(cm/s^2)]
    mh=0.0,       # Metallicity [dex]
    output_dir="./predictions",
    output_format='csv'   # Save as CSV file
)

print(f"CSV saved to: {csv_path}")   # → .../predictions/10000_3.7_0.0.csv
print(f"DataFrame shape: {df.shape}")  # (50, 58) → 50 depths × 58 parameters
```

The default file names follow the format **`{teff}_{logg}_{mh}.csv`** and **`{teff}_{logg}_{mh}.7`**.

# Predict and save as TLUSTY .7 format (fort.7)

```python
df, seven_path = predict_atmosphere(
    teff=10000,
    logg=3.7,
    mh=0.0,
    output_dir="./predictions",
    output_format='7'     # Save as .7 file
)

print(f".7 file saved to: {seven_path}")  # → .../predictions/10000_3.7_0.0.7
```

### Create TLUSTY input file (.5 format)

```python
from tlustynn import create_ff_model

# Generate a TLUSTY input model file (fort.5 format)
create_ff_model(
    output_dir='/path/to/workdir',
    teff=10000,
    logg=3.7,
    mh=0.0,
    lte_flag='F',
    ltgray_flag='F',
    nstmode='nst',
    frequency=2000,
    natoms_num=8
)
```


##  Output formats

### CSV format

The output CSV follows exactly the same column order as the original `hhe.csv` training data:

| Column | Description |
|--------|-------------|
| `teff` | Effective temperature [K] (replicated for all 50 rows) |
| `logg` | Surface gravity (replicated) |
| `mh`   | Metallicity [dex] (replicated) |
| `tau`  | Optical depth (average profile in physical units) |
| `T`    | Temperature [K] |
| `ne`   | Electron number density [cm⁻³] |
| `rho`  | Mass density [g/cm³] |
| `level_1` … `level_55` | Level populations |

Each file contains **50 rows**, one per atmospheric depth layer.

### TLUSTY `.7` format

The `.7` file is a plain-text model atmosphere in the standard TLUSTY `fort.7` format (identical to `FF.7`):

```
   50   58
 3.339578E-07 5.245317E-07 8.231387E-07 ...
 ...
   1.812165E+04   8.381594E+10   1.910841E-13   7.939990E+05 ...
```

- **Line 1**: `n_depth` (50) and `n_params` (58)
- **Next lines**: `tau` values, 6 per line
- **Remaining lines**: for each depth, the 58 parameters (`T`, `ne`, `rho`, `level_1` … `level_55`), 6 per line

---


## 🔬 Training your own model

If you have the full `hhe.csv` dataset (~2.5 GB), you can re-train or fine-tune the network:

```bash
# Put hhe.csv in the working directory (or edit tlustynn/config.py::CSV_PATH)
python scripts/train.py --epochs 1500
```

Trained checkpoints will be saved to `./checkpoints/` by default.

---

## 📁 Repository structure

```
tlusty-nn/
├── tlustynn/                 # Main Python package
│   ├── __init__.py
│   ├── api.py                # User-facing predict_atmosphere() API
│   ├── model.py              # TLUSTYNN network definition
│   ├── predict.py            # TlustyPredictor (model loading & inference)
│   ├── data_loader.py        # Dataset & preprocessing
│   ├── physics.py            # Physics-informed loss constraints
│   ├── train.py              # Trainer class
│   ├── utils.py              # Plotting utilities
│   ├── read_write_tlusty.py  # TLUSTY fort.7 I/O helpers
│   └── checkpoints/          # Pretrained weights (best_model.pt, stats.json, ...)
├── scripts/
│   ├── train.py              # Training entry point
│   └── evaluate.py           # Evaluation & plotting entry point
├── tests/
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📋 Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, Pandas, scikit-learn, Matplotlib, tqdm

All dependencies are listed in `requirements.txt` and will be installed automatically with `pip install`.
---

## 🎯 Applicability Range

The neural network model is trained and validated within the following stellar parameter ranges:

| Parameter | Symbol | Range | Units |
|-----------|--------|-------|-------|
| Effective temperature | `Teff` | 10,000 – 100,000 | K |
| Surface gravity | `logg` | 1.5 – 9.0 | log10(cm/s²) |
| Helium abundance | `log(n_He / n_H)` | -4.0 – 0.0 | dex |

### Notes

- **Extrapolation warning**: Predictions made outside the above ranges may be physically inaccurate or unreliable. The network has not been trained on data beyond these bounds.
- **Metallicity**: Currently, the model assumes solar metallicity (`mh = 0.0` in the API). Support for varying metallicity will be added in future versions.
- **Intended use**: This model is designed for rapid prototyping, parameter space exploration, and applications where TLUSTY runtime is prohibitive. For final scientific results requiring high precision, please validate against full TLUSTY calculations.



## Acknowledgements

Thanks to Dr. Jiao Li, Dr. Jiadong Lee, Dr. Mingjie Jian, Dr. Yangping Luo, Dr. Chenyu He, Dr. Xiaodian Chen, Dr. Zhihong He and Dr. Qian Cui for their assistance with this project.
