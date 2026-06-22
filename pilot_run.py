# pilot_run.py
# Pilot experiments: small-sample draws + generation (preprocessing aligned with run_augmentation.py)
from Experiments_new import PilotExperiment
from preprocess_utils import DEFAULT_SKIP_PREFIXES
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = current_dir
output_subdir = "pilot_output"
output_path = os.path.join(current_dir, output_subdir)

# Input CSV filename (without .csv extension); file must be in the same folder as this script
input_filename = "processed_train_136_full_model_data"

# First condition column in the raw CSV; None for unconditional training
condition1 = "groups"

# Preprocessing (same options as run_augmentation.py)
apply_log = False
apply_yj = "all"
apply_zscale = True
skewness_threshold = 1.0

var_filter_method = "nzv"
var_threshold = 0.01
nzv_freq_cut = 95.0 / 5.0
nzv_unique_cut = 10.0
skip_columns = None
skip_prefixes = DEFAULT_SKIP_PREFIXES

# Pilot-specific settings
pilot_size = [100, 150, 200]   # absolute sample counts per draw (per group if two-group data)
num_draws = 30                # random draws per pilot size
new_size = [1000]            # generated samples per run; None -> legacy 5 * full dataset size

# Model and training
model = "CVAE1-0.5"
batch_frac = 0.1
learning_rate = 0.0005
epoch = 1000
early_stop_num = 30
off_aug = None
AE_head_num = 2
Gaussian_head_num = 9
pre_model = None
save_model = None

# Loss figure handling: "save" (auto-save to pilot_output/loss_figure/, no popup), "show", "none"
loss_figure_mode = "none"

print(f"Input file: {os.path.join(input_path, input_filename + '.csv')}")
print(f"Output directory: {os.path.abspath(output_path)}")
print("Starting pilot experiments...")

PilotExperiment(
    dataname=input_filename,
    pilot_size=pilot_size,
    model=model,
    batch_frac=batch_frac,
    learning_rate=learning_rate,
    epoch=epoch,
    early_stop_num=early_stop_num,
    off_aug=off_aug,
    AE_head_num=AE_head_num,
    Gaussian_head_num=Gaussian_head_num,
    pre_model=pre_model,
    input_path=input_path + os.sep,
    output_path=output_path + os.sep,
    condition1=condition1,
    new_size=new_size,
    num_draws=num_draws,
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
    save_model=save_model,
    loss_figure_mode=loss_figure_mode,
)

print("Pilot experiments completed!")
print(f"Results saved under: {os.path.abspath(output_path)}")
