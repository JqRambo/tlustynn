# TLUSTY NN

Physics-Informed Neural Network (PINN) for fast prediction of TLUSTY stellar atmosphere models.

Given three stellar parameters — **effective temperature (Teff)**, **surface gravity (logg)** and **metallicity ([M/H])** — the network predicts the full 50-layer atmospheric structure (temperature, electron density, mass density and 55 level populations) in a fraction of a second.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tlusty-nn.git
cd tlusty-nn
```

> **Note on large files**
> The pretrained model weight (`best_model.pt`, ~1 GB) exceeds GitHub's normal file-size limit (100 MB). You have three options:
> 1. **Git LFS** – if the repository is configured with Git LFS, run `git lfs pull` after cloning.
> 2. **Release asset** – download `best_model.pt` from the GitHub Releases page and place it under `tlustynn/checkpoints/`.
> 3. **Manual copy** – if you already have the file locally, copy it to `tlustynn/checkpoints/best_model.pt`.

### 2. Install into your Python environment

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

# Predict and automatically save to CSV + TLUSTY .7 file
df, paths = predict_atmosphere(
    teff=10000,   # Effective temperature [K]
    logg=3.7,     # Surface gravity [log10(cm/s^2)]
    mh=0.0,       # Metallicity [dex]
    output_dir="./predictions"
)

print(f"CSV saved to: {paths['csv']}")    # → .../predictions/10000_3.7_0.0.csv
print(f".7  saved to: {paths['7']}")      # → .../predictions/10000_3.7_0.0.7
```

The default file names follow the format **`{teff}_{logg}_{mh}.csv`** and **`{teff}_{logg}_{mh}.7`**.

### Disable the .7 output

```python
df, paths = predict_atmosphere(
    teff=27000, logg=3.9, mh=-1.0,
    output_dir="./my_models",
    save_7=False          # only CSV will be written
)
```

### Custom file name

```python
df, paths = predict_atmosphere(
    teff=27000, logg=3.9, mh=-1.0,
    output_dir="./my_models",
    filename="model_A"     # produces model_A.csv and model_A.7
)
```

### Get a DataFrame without saving

```python
df, _ = predict_atmosphere(10000, 3.7, 0.0)
print(df.shape)   # (50, 62)  → 50 depth layers × 62 columns
print(df.columns.tolist())
```

### Use the object-oriented API

```python
from tlustynn import TlustyAtmosphere

atm = TlustyAtmosphere()
df, paths = atm.predict(10000, 3.7, 0.0, output_dir="./predictions")
```

---

## 📄 Output formats

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

## 📜 License

MIT
