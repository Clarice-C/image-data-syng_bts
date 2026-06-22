# run_augmentation.py
# All files are in the same directory, so direct import is possible
from Experiments_new import ApplyExperiment
from preprocess_utils import DEFAULT_SKIP_PREFIXES
import os

# Get the directory where this script is located (all files are here)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Use absolute path to avoid path issues on Windows
input_path = current_dir  # Current directory (absolute path)

# Set input filename (without .csv extension)
input_filename = "processed_train_136_full_model_data"

# First condition column name in the raw CSV (passed through the whole pipeline as condition1)
# Example: condition1 = "luminal-like"  or  condition1 = "groups"
# Set to None for unconditional training (no condition column)
condition1 = "groups"

# Preprocessing: raw data is read from path; options below control transform (params fit in-memory, no pre-existing CSV needed)
apply_log = False       # True: apply log2(x+1)
apply_yj = "all"       # False, "selected" (YJ only when skewness > skewness_threshold), or "all"
apply_zscale = True     # True: apply z-score (after YJ if apply_yj, else on raw)
skewness_threshold = 1.0   # used when apply_yj == "selected"

# Variance filter (Python implementation; condition1 is skipped automatically)
var_filter_method = "nzv"   # "none" | "nzv" | "variance"
var_threshold = 0.01         # used when var_filter_method == "variance"
nzv_freq_cut = 95.0 / 5.0    # used when var_filter_method == "nzv"
nzv_unique_cut = 10.0        # used when var_filter_method == "nzv"
skip_columns = None          # extra columns to skip besides condition1, e.g. ["patient_id"]
skip_prefixes = DEFAULT_SKIP_PREFIXES  # optional prefix skip (e.g. legacy groups2 columns)

# Output path: preprocessed data and generated data saved under input_path
# Preprocessed CSV: {dataname}_preprocessed.csv; generated: {dataname}_{model}_generated.csv

# Loss figure handling: "save" (auto-save to loss_figure/, no popup), "show" (popup), "none" (skip)
loss_figure_mode = "save"

print(f"Input file path: {os.path.join(input_path, input_filename + '.csv')}")
print(f"Output directory: {os.path.abspath(input_path)}")
print("Starting data augmentation...")

ApplyExperiment(
    path=input_path + os.sep,   # Current directory (using os.sep for cross-platform compatibility)
    dataname=input_filename,      # CSV filename (without .csv)
    condition1=condition1,
    apply_log=apply_log,
    apply_yj=apply_yj,
    apply_zscale=apply_zscale,
    skewness_threshold=skewness_threshold,
    var_filter_method=var_filter_method,
    var_threshold=var_threshold,
    nzv_freq_cut=nzv_freq_cut,
    nzv_unique_cut=nzv_unique_cut,
    skip_columns=skip_columns,
    skip_prefixes=skip_prefixes,
    new_size=[1000],            # Generate 1000 new samples
    model="CVAE1-1",                # Model type: CVAE (conditional on condition1)
                                # Alternative options: "MAF", "realnvp", "glow"
    batch_frac=0.1,
    learning_rate=0.0005,       # Learning rate
    epoch=1000,                 # Maximum training epochs
    early_stop_num=30,          # Early stopping patience
    off_aug=None,
    AE_head_num=2,
    Gaussian_head_num=9,
    pre_model=None,
    save_model=None,
    loss_figure_mode=loss_figure_mode,
)

print("Data augmentation completed!")
print(f"Output file saved at: {os.path.abspath(input_path)}")