# Image Data Synthesis (BTS)

A pipeline for synthesizing tabular biological/imaging data using deep generative models (VAE, CVAE, GAN, normalizing flows). Includes preprocessing, training, generation, and evaluation tools.

---

## Repository Overview

```
run_augmentation.py          # Main entry point — edit config here and run
Experiments_new.py           # Experiment functions (ApplyExperiment, PilotExperiment, TransferExperiment)
preprocess_utils.py          # Feature filtering, Yeo-Johnson, z-score, inverse transforms
helper_training_new.py       # Orchestration: builds models, loaders, calls training loops
helper_models_new.py         # PyTorch model architectures (AE, VAE, CVAE, MultiCVAE, GAN)
helper_train_new.py          # Epoch-level training loops and normalizing flow architectures
helper_utils_new.py          # Utilities: label creation, sampling, augmentation, CSV saving
corr_5.4.ipynb               # Correlation fidelity evaluation notebook
evaluation_pipeline_4.28.ipynb  # Comprehensive multi-metric evaluation notebook
```

---

## Requirements

Install dependencies before running:

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn tqdm tensorboardX umap-learn
```

`umap-learn` is optional — the UMAP plot in `evaluation_pipeline_4.28.ipynb` is skipped if not installed.

---

## Quickstart: Generating Synthetic Data

### 1. Prepare your input data

Place your CSV file in the `raw-data` repository directory. The file should have:
- Numeric feature columns
- An optional `groups` column (binary 0/1) for conditional generation (CVAE)
- An optional `groups2` column for multi-conditional generation (MultiCVAE)

Columns whose names start with `groups` are automatically excluded from feature processing.

### 2. Configure `run_augmentation.py`

Open [run_augmentation.py](run_augmentation.py) and set the parameters at the top of the file:

```python
input_filename = "your_data_file"   # CSV filename without .csv extension

# Preprocessing
apply_log    = False                 # Apply log2(x+1) transform
apply_yj     = True                  # Apply Yeo-Johnson transform
apply_zscale = True                  # Apply z-score normalization
skewness_threshold = 0.5             # Apply YJ only to columns with |skewness| > threshold

# Feature filtering
var_filter_method = "nzv"            # "none", "variance", or "nzv"

# Generation
new_size  = [1000]                   # Number of synthetic samples to generate
model     = "CVAE1-1"                # Model type (see Model Options below)
epoch     = 1000                     # Max training epochs
early_stop_num = 30                  # Stop after N epochs with no improvement

# Optimizer
batch_frac    = 0.1                  # Batch size as fraction of training data
learning_rate = 0.0005
```

### 3. Run

```bash
python run_augmentation.py
```

### 4. Outputs

Three files are written to the same directory as the input:

| File | Description |
|------|-------------|
| `{dataname}_preprocessed.csv` | Preprocessed version of the input data |
| `{dataname}_{model}_generated.csv` | Generated synthetic samples |
| `{dataname}_{model}_loss.csv` | Training loss history |

---

## Model Options

Pass any of the following as the `model` argument. The suffix (e.g. `1-1`) sets the KL weight.

| Model string | Architecture | Notes |
|---|---|---|
| `"AE"` | Autoencoder | Basic reconstruction |
| `"VAE1-1"` | Variational AE | KL weight = 1 |
| `"CVAE1-1"` | Conditional VAE | Requires `groups` column |
| `"MultiCVAE1-1"` | Multi-conditional VAE | Requires `groups` and `groups2` columns |
| `"GAN"` | Generative Adversarial Network | |
| `"WGAN"` | Wasserstein GAN | |
| `"WGANGP"` | WGAN with gradient penalty | |
| `"maf"` | Masked Autoregressive Flow | |
| `"realnvp"` | RealNVP coupling flow | |
| `"glow"` | Glow (invertible 1×1 conv) flow | |
| `"maf-split"` | MAF with split autoregressive layers | |
| `"maf-split-glow"` | MAF-split + Glow mixing | |

---

## Preprocessing Options

| Parameter | Effect |
|---|---|
| `apply_log=True` | `log2(x+1)` per column (skips columns with values ≤ -1) |
| `apply_yj=True` | Yeo-Johnson power transform per column |
| `apply_zscale=True` | Z-score normalization per column |
| `skewness_threshold` | With `apply_yj`, only transform columns whose skewness exceeds this value |
| `var_filter_method="nzv"` | Remove near-zero-variance features (caret-style rule) |
| `var_filter_method="variance"` | Remove features below a variance threshold |
| `var_filter_method="none"` | Keep all features |

---

## Advanced: Running Experiments Programmatically

Import and call `ApplyExperiment` directly from Python or a notebook:

```python
from Experiments_new import ApplyExperiment

ApplyExperiment(
    path        = "./data/",          # directory containing input CSV
    dataname    = "my_dataset",       # CSV filename without .csv
    preprocess  = True,
    apply_log   = False,
    apply_yj    = True,
    apply_zscale= True,
    skewness_threshold = 0.5,
    var_filter_method  = "nzv",
    new_size    = [1000],
    model       = "VAE1-1",
    batch_frac  = 0.1,
    learning_rate = 0.0005,
    epoch       = 500,
    early_stop_num = 30,
)
```

### Transfer learning

Use `TransferExperiment` to pre-train on a source dataset and fine-tune on a target:

```python
from Experiments_new import TransferExperiment

TransferExperiment(
    pilot_size  = [50, 100],
    fromname    = "source_dataset",
    toname      = "target_dataset",
    fromsize    = 1000,
    model       = "VAE1-1",
    ...
)
```

### Pilot studies

Use `PilotExperiment` to evaluate performance across multiple pilot sample sizes (5 random draws each):

```python
from Experiments_new import PilotExperiment

PilotExperiment(
    dataname    = "my_dataset",
    pilot_size  = [50, 100, 200],
    model       = "VAE1-1",
    ...
)
```

---

## Evaluation Notebooks

After generating synthetic data, evaluate quality with the provided notebooks.

### Correlation fidelity — `corr_5.4.ipynb`

Measures how well the synthetic data preserves pairwise Pearson correlation structure.

**Before running:** Update these variables in the first code cell:

```python
REAL_PATH = "your_real_data.csv"
SYN_PATH  = "your_generated_data.csv"
```

**Outputs (displayed in-notebook):**
- Global and stratified correlation agreement rates
- False-correlation rate (neutral pairs incorrectly correlated in synthetic)
- Bivariate accuracy scores (binned joint distribution comparison)

### Comprehensive evaluation — `evaluation_pipeline_4.28.ipynb`

Side-by-side comparison of two generators against real data across multiple metrics.

**Before running:** Update the config block (Chunk 2):

```python
SHARED_REAL_FILE = "your_real_data.csv"
GEN_1_FILE       = "generator_1_output.csv"
GEN_2_FILE       = "generator_2_output.csv"
DATA_DIR         = "./your/data/directory/"
N_BOOTSTRAP      = 300
```

**Outputs (displayed in-notebook):**
- Summary statistics table (mean, SD, skewness, kurtosis per feature)
- Marginal distribution metrics: Wasserstein-1 distance, KS statistic (bootstrap violin plots)
- UMAP 2D embeddings of real vs. generated samples
- Sample structure: kNN mixing score, silhouette score (bootstrap violin plots)
- Correlation difference heatmaps and scalar metrics (Frobenius norm, Mantel correlation)

---

## Data Flow

```
Input CSV
    │
    ▼
preprocess_utils.py   ← feature filtering (NZV / variance)
                      ← Yeo-Johnson + z-score transforms
    │
    ▼
helper_training_new.py  ← model instantiation, DataLoader, optimizer
    │
    ├── helper_models_new.py   (model architectures)
    └── helper_train_new.py    (training loops)
    │
    ▼
helper_utils_new.py   ← sample generation, inverse transform, CSV saving
    │
    ▼
*_generated.csv
    │
    ▼
Evaluation notebooks (corr_5.4.ipynb, evaluation_pipeline_4.28.ipynb)
```
