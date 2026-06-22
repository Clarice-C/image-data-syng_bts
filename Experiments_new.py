#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:09:21 2022

@author: yunhui, xinyi
"""

# %% Import libraries
import torch
import pandas as pd
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from helper_utils_new import *
from helper_training_new import *
import re

sns.set()

import importlib.resources as pkg_resources
from preprocess_utils import DEFAULT_SKIP_PREFIXES

# %% Define pilot experiments functions
def _pilot_ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _pilot_normalize_dir(path):
    path = os.path.normpath(path)
    if not path.endswith(os.sep):
        path = path + os.sep
    return path


def PilotExperiment(
    dataname,
    pilot_size,
    model,
    batch_frac,
    learning_rate,
    epoch,
    early_stop_num=30,
    off_aug=None,
    AE_head_num=2,
    Gaussian_head_num=9,
    pre_model=None,
    input_path=None,
    output_path=None,
    condition1=None,
    new_size=None,
    num_draws=5,
    preprocess=None,
    apply_log=False,
    apply_yj=False,
    apply_zscale=False,
    skewness_threshold=1.0,
    var_filter_method="none",
    var_threshold=0.01,
    nzv_freq_cut=95.0 / 5.0,
    nzv_unique_cut=10.0,
    skip_columns=None,
    skip_prefixes=DEFAULT_SKIP_PREFIXES,
    save_model=None,
    use_scheduler=False,
    step_size=10,
    gamma=0.5,
    loss_figure_mode="save",  # "show" | "save" | "none"
):
    r"""
    Pilot experiments: preprocess full data (same options as ApplyExperiment when
    input_path/output_path are set), draw small subsets per pilot_size, train, generate.

    When input_path and output_path are both None, uses legacy ../RealData/ paths and
    log2-only preprocessing (for TransferExperiment compatibility).
    """
    use_legacy = input_path is None and output_path is None

    if isinstance(pilot_size, int):
        pilot_size = [pilot_size]

    if use_legacy:
        read_path = "../RealData/" + dataname + ".csv"
        df = pd.read_csv(read_path, header=0)
        dat_pd = df
        data_pd = dat_pd.select_dtypes(include=np.number)
        oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)
        oridata = preprocessinglog2(oridata)
        n_samples = oridata.shape[0]
        cond1 = dat_pd["groups"] if "groups" in dat_pd.columns else None
        groups2 = dat_pd["groups2"] if "groups2" in dat_pd.columns else None
        feat_cols = None
        colnames = None
        preprocess_params_df = None
        condition1 = condition1 or "groups"
        print("1. Read data (legacy), path is " + read_path)
    else:
        input_path = _pilot_normalize_dir(input_path)
        output_path = _pilot_normalize_dir(output_path)
        read_path = input_path + dataname + ".csv"
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"Input CSV not found: {read_path}")
        dat_pd = pd.read_csv(read_path, header=0)

        effective_skip = list(skip_columns) if skip_columns else []
        if condition1 and condition1 not in effective_skip:
            effective_skip.append(condition1)
        skip_columns_prep = effective_skip if effective_skip else None

        if condition1:
            if condition1 not in dat_pd.columns:
                raise ValueError(
                    f"condition1 column '{condition1}' not found in {read_path}. "
                    f"Available columns: {list(dat_pd.columns)}"
                )
            cond1 = dat_pd[condition1]
            print(f"   Condition1: column '{condition1}'")
        else:
            cond1 = None
            print("   Condition1: not configured (unconditional training)")

        groups2 = dat_pd["groups2"] if "groups2" in dat_pd.columns else None

        from preprocess_utils import filter_feature_columns, select_passthrough_columns
        feat_cols, filtered_df = filter_feature_columns(
            dat_pd,
            filter_method=var_filter_method,
            variance_threshold=var_threshold,
            nzv_freq_cut=nzv_freq_cut,
            nzv_unique_cut=nzv_unique_cut,
            skip_columns=skip_columns_prep,
            skip_prefixes=skip_prefixes,
        )
        print(f"   Var filter ({var_filter_method}): {len(feat_cols)} numeric features kept")
        passthrough_cols = select_passthrough_columns(
            dat_pd, skip_columns=skip_columns_prep, skip_prefixes=skip_prefixes
        )
        oridata = torch.from_numpy(filtered_df.to_numpy()).to(torch.float32)
        colnames = filtered_df.columns
        preprocess_params_df = None

        if preprocess is not None:
            if preprocess == "log":
                apply_log = True
            elif preprocess == "yj":
                apply_yj = "all"
                apply_zscale = True
            else:
                raise ValueError(f"preprocess must be None, 'log', or 'yj', got: {preprocess}")

        if apply_log:
            oridata = preprocessinglog2(oridata)
            print("   Applied log2(x+1)")
        if apply_yj in ("selected", "all") or apply_zscale:
            from preprocess_utils import fit_transform_for_augmentation
            X_raw = oridata.numpy()
            X_prep, preprocess_params_df = fit_transform_for_augmentation(
                X_raw,
                feat_cols,
                yj_mode=apply_yj if apply_yj in ("selected", "all") else False,
                zscore=apply_zscale,
                skewness_threshold=skewness_threshold,
            )
            oridata = torch.from_numpy(X_prep).to(torch.float32)
            print("   Applied YJ/zscore preprocessing (fit in-memory)")

        preprocessed_path = os.path.join(
            os.path.normpath(output_path.rstrip(os.sep)),
            dataname + "_preprocessed.csv",
        )
        prep_df = pd.DataFrame(oridata.numpy(), columns=feat_cols)
        for c in passthrough_cols:
            prep_df[c] = dat_pd[c].values
        _pilot_ensure_dir(os.path.dirname(preprocessed_path))
        prep_df.to_csv(preprocessed_path, index=False)
        print("   Saved preprocessed data:", preprocessed_path)
        n_samples = oridata.shape[0]
        print("1. Read data, path is " + read_path)

        _pilot_ensure_dir(os.path.join(output_path.rstrip(os.sep), "generated"))
        _pilot_ensure_dir(os.path.join(output_path.rstrip(os.sep), "loss"))

    if model.startswith("MultiCVAE"):
        match = re.match(r"MultiCVAE(\d)([-+])(\d+)", model)
        if match:
            modelname = "MultiCVAE"
            kl_weight = int(match.group(3))
        else:
            modelname = "MultiCVAE"
            kl_weight = 1
    elif len(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)) > 1:
        kl_weight = int(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[4])
        modelname = re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[1]
    else:
        modelname = model
        kl_weight = 1

    print("2. Determine the model is " + model + " with kl-weight = " + str(kl_weight))
    print("   Parsed modelname: " + modelname)

    if not use_legacy:
        if cond1 is not None and groups2 is not None:
            if modelname == "CVAE":
                modelname = "MultiCVAE"
            print(f"Using multi-conditional labels ({condition1} + groups2)")
            orilabels, oriblurlabels = create_multi_conditional_labels(
                n_samples=n_samples, groups=cond1, groups2=groups2
            )
        elif modelname == "MultiCVAE":
            if cond1 is None or groups2 is None:
                print("Warning: MultiCVAE requires condition1 and groups2 columns!")
                orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=cond1)
            else:
                orilabels, oriblurlabels = create_multi_conditional_labels(
                    n_samples=n_samples, groups=cond1, groups2=groups2
                )
        else:
            if cond1 is not None:
                print(f"Using single conditional labels (condition1 = '{condition1}')")
            else:
                print("No conditional labels available")
            orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=cond1)
    else:
        orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=cond1)

    pilot_new_size = new_size
    if pilot_new_size is None:
        repli = 5
        if (len(torch.unique(orilabels)) > 1) and (
            int(sum(orilabels == 0)) != int(sum(orilabels == 1))
        ):
            pilot_new_size = [int(sum(orilabels == 0)), int(sum(orilabels == 1)), repli]
        else:
            pilot_new_size = [repli * n_samples]
    elif modelname == "MultiCVAE":
        num_groups = 2
        if groups2 is not None:
            groups2_unique = np.unique(np.array(groups2))
            num_groups2 = len(groups2_unique)
        else:
            num_groups2 = 1
        num_combinations = num_groups * num_groups2
        if isinstance(pilot_new_size, list) and len(pilot_new_size) == 1:
            total_samples = pilot_new_size[0]
            base_per_combo = total_samples // num_combinations
            rem = total_samples % num_combinations
            pilot_new_size = [base_per_combo] * num_combinations
            for i in range(rem):
                pilot_new_size[i] += 1
            pilot_new_size.append(1)
            print(
                f"MultiCVAE: Distributing {total_samples} samples across "
                f"{num_combinations} combinations: {pilot_new_size[:-1]}"
            )

    model_tag = "batch" + str(batch_frac).replace(".", "") + "_" + model

    if use_legacy:
        if epoch is not None:
            num_epochs = epoch
            early_stop = False
            epoch_info = str(epoch)
            model_tag = "epoch" + epoch_info + "_" + model_tag
        else:
            num_epochs = 1000
            early_stop = True
            epoch_info = "early_stop"
            model_tag = "epochES_" + model_tag
    else:
        num_epochs = epoch
        if early_stop_num is not None:
            early_stop = True
            epoch_info = "early_stop"
            model_tag = "epochES_" + model_tag
        else:
            early_stop = False
            epoch_info = str(epoch)
            model_tag = "epoch" + epoch_info + "_" + model_tag

    if off_aug == "AE_head":
        AE_head = True
        Gaussian_head = False
        off_aug_info = off_aug
    elif off_aug == "Gaussian_head":
        Gaussian_head = True
        AE_head = False
        off_aug_info = off_aug
    else:
        AE_head = False
        Gaussian_head = False
        off_aug_info = "No"

    print(
        "3. Training parameters: epoch = "
        + epoch_info
        + " off_aug = "
        + off_aug_info
        + " learning rate = "
        + str(learning_rate)
        + " batch_frac = "
        + str(batch_frac)
        + " num_draws = "
        + str(num_draws)
    )

    if pre_model is not None:
        model_tag = model_tag + "_transfrom" + re.search(r"from([A-Z]+)_", pre_model).group(1)

    random_seed = 123

    print("4. Pilot experiments start ...")
    for n_pilot in pilot_size:
        for rand_pilot in range(1, num_draws + 1):
            print(
                f"Training for data={dataname}, model={model_tag}, "
                f"pilot size={n_pilot}, draw {rand_pilot}/{num_draws}"
            )

            rawdata, rawlabels, rawblurlabels = draw_pilot(
                dataset=oridata,
                labels=orilabels,
                blurlabels=oriblurlabels,
                n_pilot=n_pilot,
                seednum=rand_pilot,
            )

            if (modelname != "CVAE") and (modelname != "MultiCVAE") and (
                torch.unique(rawlabels).shape[0] > 1
            ):
                rawdata = torch.cat((rawdata, rawblurlabels), dim=1)

            prefix = dataname + "_"
            if Gaussian_head:
                prefix = dataname + "_Gaussianhead_"
            elif AE_head:
                prefix = dataname + "_AEhead_"

            stem = prefix + model_tag + "_" + str(n_pilot) + "_Draw" + str(rand_pilot)

            if use_legacy:
                savepathnew = "../GeneratedData/" + stem + ".csv"
                losspath = "../Loss/" + stem + ".csv"
                savepath = "../ReconsData/" + stem + ".csv"
                savepathextend = "../ExtendData/" + stem + ".csv"
            else:
                savepathnew = os.path.join(
                    output_path.rstrip(os.sep), "generated", stem + "_generated.csv"
                )
                losspath = os.path.join(
                    output_path.rstrip(os.sep), "loss", stem + "_loss.csv"
                )
                savepath = None
                savepathextend = None

            if use_legacy:
                run_loss_figure_dir = None
            else:
                run_loss_figure_dir = os.path.join(
                    output_path.rstrip(os.sep), "loss_figure", stem
                )
            run_loss_figure_prefix = ""

            if Gaussian_head:
                rawdata, rawlabels = Gaussian_aug(
                    rawdata, rawlabels, multiplier=[Gaussian_head_num]
                )
                print("Gaussian head is added.")

            if AE_head:
                print("AE reconstruction head is added, reconstruction starting ...")
                feed_data, feed_labels = training_iter(
                    iter_times=AE_head_num,
                    savepathextend=savepathextend,
                    rawdata=rawdata,
                    rawlabels=rawlabels,
                    random_seed=random_seed,
                    modelname="AE",
                    num_epochs=1000,
                    batch_size=round(rawdata.shape[0] * 0.1),
                    learning_rate=0.0005,
                    early_stop=False,
                    early_stop_num=30,
                    kl_weight=1,
                    loss_fn="MSE",
                    replace=True,
                    saveextend=False,
                    plot=False,
                    loss_figure_mode=loss_figure_mode,
                    loss_figure_dir=run_loss_figure_dir,
                    loss_figure_prefix="aehead",
                )
                rawdata = feed_data
                rawlabels = feed_labels
                print("Reconstruction finish, AE head is added.")

            if "GAN" in modelname:
                log_dict = training_GANs(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    rawlabels=rawlabels,
                    batch_size=round(rawdata.shape[0] * batch_frac),
                    random_seed=random_seed,
                    modelname=modelname,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    new_size=pilot_new_size,
                    early_stop=early_stop,
                    early_stop_num=early_stop_num,
                    pre_model=pre_model,
                    save_model=save_model,
                    save_new=True,
                    plot=False,
                    loss_figure_mode=loss_figure_mode,
                    loss_figure_dir=run_loss_figure_dir,
                    loss_figure_prefix=run_loss_figure_prefix,
                )
                log_pd = pd.DataFrame(
                    {
                        "discriminator": log_dict["train_discriminator_loss_per_batch"],
                        "generator": log_dict["train_generator_loss_per_batch"],
                    }
                )
            elif "AE" in modelname:
                log_dict = training_AEs(
                    savepath=savepath,
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    rawlabels=rawlabels,
                    colnames=colnames,
                    preprocess=preprocess,
                    preprocess_params_df=preprocess_params_df,
                    feat_cols=feat_cols,
                    batch_size=round(rawdata.shape[0] * batch_frac),
                    random_seed=random_seed,
                    modelname=modelname,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    kl_weight=kl_weight,
                    early_stop=early_stop,
                    early_stop_num=early_stop_num,
                    pre_model=pre_model,
                    save_model=save_model,
                    loss_fn="MSE",
                    save_recons=False,
                    new_size=pilot_new_size,
                    save_new=True,
                    plot=False,
                    use_scheduler=use_scheduler,
                    step_size=step_size,
                    gamma=gamma,
                    condition1_col=condition1,
                    loss_figure_mode=loss_figure_mode,
                    loss_figure_dir=run_loss_figure_dir,
                    loss_figure_prefix=run_loss_figure_prefix,
                )
                log_pd = pd.DataFrame(
                    {
                        "kl": log_dict["train_kl_loss_per_batch"],
                        "recons": log_dict["train_reconstruction_loss_per_batch"],
                    }
                )
            elif "maf" in modelname or "realnvp" in modelname or "glow" in modelname:
                flow_rawlabels = rawlabels
                if cond1 is not None and groups2 is not None:
                    print(f"Using conditional Flow with {condition1} and groups2")
                elif cond1 is not None:
                    print(f"Using conditional Flow with condition1 = '{condition1}'")
                log_dict, _ = training_flows(
                    savepathnew=savepathnew,
                    rawdata=rawdata,
                    batch_frac=batch_frac,
                    valid_batch_frac=0.3,
                    random_seed=random_seed,
                    modelname=modelname,
                    num_blocks=5,
                    num_epoches=num_epochs,
                    learning_rate=learning_rate,
                    new_size=pilot_new_size,
                    num_hidden=145,
                    early_stop=early_stop,
                    early_stop_num=early_stop_num,
                    pre_model=pre_model,
                    save_model=save_model,
                    plot=False,
                    rawlabels=flow_rawlabels,
                    preprocess=preprocess,
                    preprocess_params_df=preprocess_params_df,
                    feat_cols=feat_cols,
                    loss_figure_mode=loss_figure_mode,
                    loss_figure_dir=run_loss_figure_dir,
                    loss_figure_prefix=run_loss_figure_prefix,
                )
                log_pd = pd.DataFrame({"loss": log_dict["train_loss_per_batch"]})
            else:
                print("wait for other models")
                continue

            losspath = os.path.normpath(losspath)
            _pilot_ensure_dir(os.path.dirname(losspath))
            log_pd.to_csv(Path(losspath), index=False)
            if run_loss_figure_dir and loss_figure_mode == "save":
                print(f"   Loss figures saved under: {run_loss_figure_dir}")
            print(f"Finished pilot size={n_pilot}, draw={rand_pilot}")

# %% Define application of experiment
def ApplyExperiment(
    path,
    dataname,
    preprocess=None,
    new_size=None,
    model=None,
    batch_frac=None,
    learning_rate=None,
    epoch=None,
    validation_rate = None, # Control whether separate the dataset in-function
    early_stop_num=None,
    off_aug=None,
    AE_head_num=2,
    Gaussian_head_num=9,
    pre_model=None,
    save_model=None,
    use_scheduler = False,
    step_size = 10,
    gamma = 0.5,
    apply_log=False,
    apply_yj=False,
    apply_zscale=False,
    skewness_threshold=1.0,
    var_filter_method="none",   # "none" | "nzv" | "variance"
    var_threshold=0.01,         # used when var_filter_method == "variance"
    nzv_freq_cut=95.0 / 5.0,    # used when var_filter_method == "nzv"
    nzv_unique_cut=10.0,        # used when var_filter_method == "nzv"
    skip_columns=None,          # optional manual skip list, e.g. ["id"]
    skip_prefixes=DEFAULT_SKIP_PREFIXES,  # shared skip prefixes for all preprocessing
    condition1=None,            # CSV column name for first condition; None = unconditional
    loss_figure_mode="save",    # "show" | "save" | "none" for training loss figures
):
    r"""
        This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP
        given data, model, batch_size, learning_rate, epoch, off_aug and pre_model
        and generate new samples with size specified by the users.

    Parameters
    ----------
    path : string
                              path for reading real data and saving new data
    dataname : string
                    pure data name without .csv. Eg: BRCASubtypeSel_train
    preprocess : str or None
                      Deprecated. Use apply_log, apply_yj, apply_zscale instead. None/"log"/"yj" for backward compat.
    apply_log : bool
                      If True, apply log2(x+1) to raw data.
    apply_yj : bool or str
                      False = no YJ; "selected" = YJ only when skewness > skewness_threshold; "all" = YJ for all columns.
    apply_zscale : bool
                      If True, apply z-score (after YJ if apply_yj, else on raw).
    skewness_threshold : float
                      Used when apply_yj == "selected". Default 1.0.
    new_size : int
             the number of generated samples. If CVAE is called, the group sample size will be new_size/2.
    model : string
                              name of the model to be trained
    batch_frac : float
                    batch fraction
    learning_rate : float
              learning rate
    epoch : int
            choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
    early_stop_num : int
            if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == None.
    off_aug : string (AE_head or Gaussian_head or None)
                      choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
    AE_head_num : int
                  how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
    Gaussian_head_num : int
         how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
    pre_model : string
                      transfer learning input model. If pre_model == None, no transfer learning
    save_model : string
                    if the trained model should be saved, specify the path and name of the saved model
    use_scheduler : bool
                    turn on/off scheduler for training
    step_size : int
                    step size for scheduler
    gamma : float
                    gamma for scheduler
    condition1 : str or None
                    CSV column name for the first condition (e.g. "groups", "luminal-like").
                    Automatically skipped during preprocessing. None = unconditional training.
    """

    read_path = path + dataname + ".csv"
    # just use an if statement for datasets that are already built in
    if dataname == "BRCASubtypeSel" and not os.path.exists(path):
        with pkg_resources.open_text(
            "syng_bts.Case.BRCASubtype", "BRCASubtypeSel.csv"
        ) as data_file:
            df = pd.read_csv(data_file)
    else:
        df = pd.read_csv(read_path, header=0)
    dat_pd = df

    # condition1: resolve column, auto-skip in preprocessing
    effective_skip = list(skip_columns) if skip_columns else []
    if condition1 and condition1 not in effective_skip:
        effective_skip.append(condition1)
    skip_columns_prep = effective_skip if effective_skip else None

    if condition1:
        if condition1 not in dat_pd.columns:
            raise ValueError(
                f"condition1 column '{condition1}' not found in {read_path}. "
                f"Available columns: {list(dat_pd.columns)}"
            )
        cond1 = dat_pd[condition1]
        print(f"   Condition1: column '{condition1}'")
    else:
        cond1 = None
        print("   Condition1: not configured (unconditional training)")

    # condition2 (groups2) — reserved; will become condition2 parameter later
    groups2 = dat_pd["groups2"] if "groups2" in dat_pd.columns else None

    from preprocess_utils import filter_feature_columns, select_passthrough_columns
    feat_cols, filtered_df = filter_feature_columns(
        dat_pd,
        filter_method=var_filter_method,
        variance_threshold=var_threshold,
        nzv_freq_cut=nzv_freq_cut,
        nzv_unique_cut=nzv_unique_cut,
        skip_columns=skip_columns_prep,
        skip_prefixes=skip_prefixes,
    )
    print(
        f"   Var filter ({var_filter_method}): {len(feat_cols)} numeric features kept"
    )
    passthrough_cols = select_passthrough_columns(
        dat_pd, skip_columns=skip_columns_prep, skip_prefixes=skip_prefixes
    )
    oridata = torch.from_numpy(filtered_df.to_numpy()).to(torch.float32)
    colnames = filtered_df.columns
    preprocess_params_df = None
    # Backward compat: map preprocess to apply_* when apply_* not explicitly set
    if preprocess is not None:
        if preprocess == "log":
            apply_log = True
        elif preprocess == "yj":
            apply_yj = "all"
            apply_zscale = True
        else:
            raise ValueError(f"preprocess must be None, 'log', or 'yj', got: {preprocess}")
    # Preprocessing: log
    if apply_log:
        oridata = preprocessinglog2(oridata)
        print("   Applied log2(x+1)")
    # Preprocessing: YJ and/or zscore (fit in-memory, no pre-existing params file)
    if apply_yj in ("selected", "all") or apply_zscale:
        from preprocess_utils import fit_transform_for_augmentation
        X_raw = oridata.numpy()
        X_prep, preprocess_params_df = fit_transform_for_augmentation(
            X_raw, feat_cols,
            yj_mode=apply_yj if apply_yj in ("selected", "all") else False,
            zscore=apply_zscale,
            skewness_threshold=skewness_threshold,
        )
        oridata = torch.from_numpy(X_prep).to(torch.float32)
        print("   Applied YJ/zscore preprocessing (fit in-memory)")

    # Save preprocessed data CSV (after var filter + optional log/YJ/zscore)
    preprocessed_path = os.path.join(
        os.path.normpath(path.rstrip(os.sep)),
        dataname + "_preprocessed.csv",
    )
    prep_df = pd.DataFrame(oridata.numpy(), columns=feat_cols)
    for c in passthrough_cols:
        prep_df[c] = dat_pd[c].values
    prep_df.to_csv(preprocessed_path, index=False)
    print("   Saved preprocessed data:", preprocessed_path)
    n_samples = oridata.shape[0]

    # valdata = None
    # valgroups = None
    # if validation_rate is not None and 0 <= validation_rate < 1:
    #     val_size = int(n_samples * validation_rate)
    #     train_size = n_samples - val_size

    #     generator = torch.Generator().manual_seed(0)
    #     oridata, valdata = torch.utils.data.random_split(oridata, [train_size, val_size], generator=generator)

    #     if groups is not None:
    #         groups_tensor = torch.tensor(groups.values)
    #         origroups, valgroups = torch.utils.data.random_split(groups_tensor, [train_size, val_size], generator=generator)
        # else:
        #     origroups = valgroups = None
    # else:
        # train_data = oridata
        # val_data = None
        # train_groups = groups
        # val_groups = None

    print("1. Read data, path is " + read_path)

    # get model name and kl_weight if modelname is some autoencoder
    # First check for MultiCVAE specifically
    if model.startswith("MultiCVAE"):
        # Extract MultiCVAE and kl_weight (e.g., "MultiCVAE1-5" -> "MultiCVAE", 5)
        match = re.match(r"MultiCVAE(\d)([-+])(\d+)", model)
        if match:
            modelname = "MultiCVAE"
            kl_weight = int(match.group(3))
        else:
            modelname = "MultiCVAE"
            kl_weight = 1
    elif len(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)) > 1:
        kl_weight = int(re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[4])
        modelname = re.split(r"([A-Z]+)(\d)([-+])(\d+)", model)[1]
    else:
        modelname = model
        kl_weight = 1

    print("2. Determine the model is " + model + " with kl-weight = " + str(kl_weight))
    print("   Parsed modelname: " + modelname)
    
    # Single condition (condition1) or multi (condition1 + groups2 legacy)
    if cond1 is not None and groups2 is not None:
        if modelname == "CVAE":
            modelname = "MultiCVAE"
            print(f"Using multi-conditional labels ({condition1} + groups2)")
        elif modelname == "MultiCVAE":
            print(f"Using multi-conditional labels ({condition1} + groups2)")
        else:
            print(f"Using multi-conditional labels ({condition1} + groups2)")
        orilabels, oriblurlabels = create_multi_conditional_labels(
            n_samples=n_samples, groups=cond1, groups2=groups2
        )
        combo_df = pd.DataFrame({condition1: cond1, "groups2": groups2})
        print("Condition1 and groups2 combination distribution:")
        print(pd.crosstab(combo_df[condition1], combo_df["groups2"], margins=True))
    elif modelname == "MultiCVAE":
        if cond1 is None or groups2 is None:
            print("Warning: MultiCVAE requires condition1 and groups2 columns!")
            print("Falling back to single conditional CVAE.")
            orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=cond1)
        else:
            print(f"Using multi-conditional labels ({condition1} + groups2)")
            orilabels, oriblurlabels = create_multi_conditional_labels(
                n_samples=n_samples, groups=cond1, groups2=groups2
            )
            combo_df = pd.DataFrame({condition1: cond1, "groups2": groups2})
            print("Condition1 and groups2 combination distribution:")
            print(pd.crosstab(combo_df[condition1], combo_df["groups2"], margins=True))
    else:
        if cond1 is not None:
            print(f"Using single conditional labels (condition1 = '{condition1}')")
        else:
            print("No conditional labels available")
        orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=cond1)

    rawdata = oridata
    rawlabels = orilabels
    
    # Adjust new_size for MultiCVAE if needed
    if modelname == "MultiCVAE":
        # Determine number of combinations dynamically based on groups and groups2
        num_groups = 2  # groups: 0 or 1
        if groups2 is not None:
            groups2_unique = np.unique(np.array(groups2))
            num_groups2 = len(groups2_unique)
        else:
            num_groups2 = 1  # Default to 1 if groups2 not provided
        num_combinations = num_groups * num_groups2
        
        # If new_size is a single integer, distribute it across all combinations
        if isinstance(new_size, list) and len(new_size) == 1:
            # Distribute evenly across all combinations: (groups, groups2)
            total_samples = new_size[0]
            base_per_combo = total_samples // num_combinations
            rem = total_samples % num_combinations
            new_size = [base_per_combo] * num_combinations
            for i in range(rem):
                new_size[i] += 1
            new_size.append(1)  # Add replicate number
            print(f"MultiCVAE: Distributing {total_samples} samples across {num_combinations} combinations: {new_size[:-1]}")
        elif isinstance(new_size, list) and len(new_size) == num_combinations + 1:
            # Already in correct format: [n1, n2, ..., n_combinations, replicate]
            print(f"MultiCVAE: Using specified sample sizes per combination: {new_size[:-1]}, replicate: {new_size[-1]}")
        else:
            print(f"Warning: new_size format for MultiCVAE should be list of length 1 or {num_combinations + 1}. Got: {new_size}")

    # decide batch fraction in file name
    model = "batch" + str(batch_frac).replace(".", "") + "_" + model

    # decide epoch
    num_epochs = epoch
    if early_stop_num is not None:
        early_stop = True
        epoch_info = "early_stop"
        model = "epochES_" + model
    else:
        early_stop = False
        epoch_info = str(epoch)
        model = "epoch" + epoch_info + "_" + model

    # decide offline augmentation
    if off_aug == "AE_head":
        AE_head = True
        Gaussian_head = False
        off_aug_info = off_aug
    elif off_aug == "Gaussian_head":
        Gaussian_head = True
        AE_head = False
        off_aug_info = off_aug
    else:
        AE_head = False
        Gaussian_head = False
        off_aug_info = "No"

    print(
        "3. Determine the training parameters are epoch = "
        + epoch_info
        + " off_aug = "
        + off_aug_info
        + " learing rate = "
        + str(learning_rate)
        + " batch_frac = "
        + str(batch_frac)
    )

    if pre_model is not None:
        model = model + "_transfrom" + re.search(r"from([A-Z]+)_", pre_model).group(1)

    # hyperparameters
    random_seed = 123

    savepath = path + dataname + "_" + model + "_recons.csv"
    savepathnew = path + dataname + "_" + model + "_generated.csv"
    losspath = path + dataname + "_" + model + "_loss.csv"
    apply_loss_figure_dir = os.path.join(os.path.normpath(path.rstrip(os.sep)), "loss_figure")
    apply_loss_figure_prefix = dataname + "_" + model

    if Gaussian_head:
        rawdata, rawlabels = Gaussian_aug(
            rawdata, rawlabels, multiplier=[Gaussian_head_num]
        )
        savepath = path + dataname + "_Gaussianhead_" + model + "_recons.csv"
        savepathnew = path + dataname + "_Gaussianhead_" + model + "_generated.csv"
        losspath = path + dataname + "_Gaussianhead_" + model + "_loss.csv"
        apply_loss_figure_prefix = dataname + "_Gaussianhead_" + model
        print("Gaussian head is added.")

    if AE_head:
        savepathextend = path + dataname + "_AEhead_" + model + "_extend.csv"
        savepath = path + dataname + "_AEhead_" + model + "_recons.csv"
        savepathnew = path + dataname + "_AEhead_" + model + "_generated.csv"
        losspath = path + dataname + "_AEhead_" + model + "_loss.csv"
        apply_loss_figure_prefix = dataname + "_AEhead_" + model
        print("AE reconstruction head is added, reconstruction starting ...")
        feed_data, feed_labels = training_iter(
            iter_times=AE_head_num,  # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
            savepathextend=savepathextend,  # save path of the extended dataset
            rawdata=rawdata,  # pilot data
            rawlabels=rawlabels,  # pilot labels
            random_seed=random_seed,
            modelname="AE",  # choose from AE, VAE
            num_epochs=1000,  # maximum number of epochs if early stop is not triggered, default value for AEhead is 1000
            batch_size=round(
                rawdata.shape[0] * 0.1
            ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
            learning_rate=0.0005,  # learning rate, default value for AEhead is 0.0005
            early_stop=False,  # AEhead by default does not utilize early stopping rule
            early_stop_num=30,  # won't take effect since early_stop == False
            kl_weight=1,  # only take effect if model name is VAE, default value is 2
            loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
            replace=True,  # whether to replace the failure features in each reconstruction
            saveextend=False,  # whether to save the extended dataset, if true, savepathextend must be provided
            plot=False,
            loss_figure_mode=loss_figure_mode,
            loss_figure_dir=apply_loss_figure_dir,
            loss_figure_prefix=apply_loss_figure_prefix + "_aehead",
        )  # whether or not plot the heatmap of extended data

        rawdata = feed_data
        rawlabels = feed_labels
        print("AEhead added.")

    print("3. Training starts ......")
    # Training
    if "GAN" in modelname:
        log_dict = training_GANs(
            savepathnew=savepathnew,  # path to save newly generated samples
            rawdata=rawdata,  # raw data matrix with samples in row, features in column
            rawlabels=rawlabels,  # labels for each sample, n_samples * 1, will not be used in AE or VAE
            batch_size=round(
                rawdata.shape[0] * batch_frac
            ),  # batch size, note rawdata.shape[0] = n_pilot if no AE_head
            random_seed=random_seed,
            modelname=modelname,  # choose from "GAN","WGAN","WGANGP"
            num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
            learning_rate=learning_rate,
            new_size=new_size,  # how many new samples you want to generate
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,  # save model for transfer learning, specify the path if want to save model
            save_new=True,  # whether to save the newly generated samples
            plot=False,
            loss_figure_mode=loss_figure_mode,
            loss_figure_dir=apply_loss_figure_dir,
            loss_figure_prefix=apply_loss_figure_prefix,
        )  # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones

        print("GAN model training finished.")

        log_pd = pd.DataFrame(
            {
                "discriminator": log_dict["train_discriminator_loss_per_batch"],
                "generator": log_dict["train_generator_loss_per_batch"],
            }
        )
        # create directory if not exists
        losspath = os.path.normpath(losspath)  # Normalize path
        directory = os.path.dirname(losspath)  # Get directory part
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)
        log_pd.to_csv(Path(losspath), index=False)

    elif "AE" in modelname:
        log_dict = training_AEs(
            savepath=savepath,  # path to save reconstructed samples
            savepathnew=savepathnew,  # path to save newly generated samples
            rawdata=rawdata,  # raw data tensor with samples in row, features in column
            rawlabels=rawlabels,  # abels for each sample, n_samples * 1, will not be used in AE or VAE
            colnames = colnames,  # colnames saved
            preprocess=preprocess,  # None, "log", or "yj" (for clamp when saving)
            preprocess_params_df=preprocess_params_df,
            feat_cols=feat_cols,
            batch_size=round(rawdata.shape[0] * batch_frac),  # batch size
            random_seed=random_seed,
            modelname=modelname,  # choose from "VAE", "AE"
            num_epochs=num_epochs,  # maximum number of epochs if early stop is not triggered
            learning_rate=learning_rate,
            kl_weight=kl_weight,  # only take effect if model name is VAE, default value is
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,  # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,  # save model for transfer learning, specify the path if want to save model
            loss_fn="MSE",  # only choose WMSE if you know the weights, ow. choose MSE by default
            save_recons=False,  # whether save reconstructed data, if True, savepath must be provided
            new_size=new_size,  # how many new samples you want to generate
            save_new=True,  # whether save new samples, if True, savepathnew must be provided
            plot=False,
            use_scheduler = use_scheduler,
            step_size = step_size,
            gamma = gamma,
            condition1_col=condition1,
            loss_figure_mode=loss_figure_mode,
            loss_figure_dir=apply_loss_figure_dir,
            loss_figure_prefix=apply_loss_figure_prefix,
        )  # whether plot reconstructed samples' heatmap

        print("VAEs model training finished.")
        if loss_figure_mode == "save":
            print(f"   Loss figures saved under: {apply_loss_figure_dir}")
        log_pd = pd.DataFrame(
            {
                "kl": log_dict["train_kl_loss_per_batch"],
                "recons": log_dict["train_reconstruction_loss_per_batch"],
            }
        )
        # create directory if not exists
        losspath = os.path.normpath(losspath)  # Normalize path
        directory = os.path.dirname(losspath)  # Get directory part
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)
        log_pd.to_csv(Path(losspath), index=False)
    elif "maf" in modelname:
        flow_rawlabels = None
        if cond1 is not None and groups2 is not None:
            flow_rawlabels, _ = create_multi_conditional_labels(
                n_samples=n_samples, groups=cond1, groups2=groups2
            )
            print(f"Using conditional MAF with {condition1} and groups2")
        elif cond1 is not None:
            flow_rawlabels, _ = create_labels(n_samples=n_samples, groups=cond1)
            print(f"Using conditional MAF with condition1 = '{condition1}'")
        log_dict, best_model = training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=145,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
            rawlabels=flow_rawlabels,  # Add conditional labels
            preprocess=preprocess,
            preprocess_params_df=preprocess_params_df,
            feat_cols=feat_cols,
            loss_figure_mode=loss_figure_mode,
            loss_figure_dir=apply_loss_figure_dir,
            loss_figure_prefix=apply_loss_figure_prefix,
        )
        print("MAF model training for one pilot size one draw finished.")
        log_pd = pd.DataFrame({
            "loss": log_dict["train_loss_per_batch"],
        })
        # create directory if not exists
        losspath = os.path.normpath(losspath)  # Normalize path
        directory = os.path.dirname(losspath)  # Get directory part
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)
        log_pd.to_csv(Path(losspath), index=False)
    elif "realnvp" in modelname:
        flow_rawlabels = None
        if cond1 is not None and groups2 is not None:
            flow_rawlabels, _ = create_multi_conditional_labels(
                n_samples=n_samples, groups=cond1, groups2=groups2
            )
            print(f"Using conditional RealNVP with {condition1} and groups2")
        elif cond1 is not None:
            flow_rawlabels, _ = create_labels(n_samples=n_samples, groups=cond1)
            print(f"Using conditional RealNVP with condition1 = '{condition1}'")
        log_dict, best_model = training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=145,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
            rawlabels=flow_rawlabels,  # Add conditional labels
            preprocess=preprocess,
            preprocess_params_df=preprocess_params_df,
            feat_cols=feat_cols,
            loss_figure_mode=loss_figure_mode,
            loss_figure_dir=apply_loss_figure_dir,
            loss_figure_prefix=apply_loss_figure_prefix,
        )
        print("RealNVP model training for one pilot size one draw finished.")
        log_pd = pd.DataFrame({
            "loss": log_dict["train_loss_per_batch"],
        })
        # create directory if not exists
        losspath = os.path.normpath(losspath)  # Normalize path
        directory = os.path.dirname(losspath)  # Get directory part
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)
        log_pd.to_csv(Path(losspath), index=False)

    elif "glow" in modelname:
        flow_rawlabels = None
        if cond1 is not None and groups2 is not None:
            flow_rawlabels, _ = create_multi_conditional_labels(
                n_samples=n_samples, groups=cond1, groups2=groups2
            )
            print(f"Using conditional Glow with {condition1} and groups2")
        elif cond1 is not None:
            flow_rawlabels, _ = create_labels(n_samples=n_samples, groups=cond1)
            print(f"Using conditional Glow with condition1 = '{condition1}'")
        log_dict, best_model = training_flows(
            savepathnew=savepathnew,
            rawdata=rawdata,
            batch_frac=batch_frac,
            valid_batch_frac=0.3,
            random_seed=random_seed,
            modelname=modelname,
            num_blocks=5,
            num_epoches=num_epochs,
            learning_rate=learning_rate,
            new_size=new_size,
            num_hidden=145,
            early_stop=early_stop,  # whether use early stopping rule
            early_stop_num=early_stop_num,
            # stop training if loss does not improve for early_stop_num epochs
            pre_model=pre_model,  # load pre-trained model from transfer learning
            save_model=save_model,
            plot=False,
            rawlabels=flow_rawlabels,  # Add conditional labels
            preprocess=preprocess,
            preprocess_params_df=preprocess_params_df,
            feat_cols=feat_cols,
            loss_figure_mode=loss_figure_mode,
            loss_figure_dir=apply_loss_figure_dir,
            loss_figure_prefix=apply_loss_figure_prefix,
        )
        print("Glow model training for one pilot size one draw finished.")
        log_pd = pd.DataFrame({
            "loss": log_dict["train_loss_per_batch"],
        })
        # create directory if not exists
        losspath = os.path.normpath(losspath)  # Normalize path
        directory = os.path.dirname(losspath)  # Get directory part
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)
        log_pd.to_csv(Path(losspath), index=False)

    else:
        print("wait for other models")


# %% Define transfer learing
def TransferExperiment(
    pilot_size,
    fromname,
    toname,
    fromsize,
    model,
    new_size=500,
    preprocess="log",  # None, "log", or "yj"
    epoch=None,
    batch_frac=0.1,
    learning_rate=0.0005,
    off_aug=None,
):
    """
    This function runs transfer learning using VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP.
    The model will be first trained on the pre-training dataset, and then the trained model will be saved, and the fine-tuning dataset will be trained on the save model.
    The fine tuning model training can be pilot experiments or apply experiments depending on the input of the pilot_size.
    Make sure data files for pre_model training and fine tuning model training are in the folder Transfer/.

    Parameters
    ----------
    pilot_size : int
                    if None, the fine tuning model will be apply experiment and new_size will take effect
                    otherwise, the fine tuning model will be trained using pilot experiments
    fromname : string
                name of the pretraining dataset
    toname : string
                name of the fine tuning dataset
    fromsize : int
                number of samples when pre-training the model
    new_size : int
                if apply experiment, this will be the sample size of generated samples
    preprocess : str or None
                None, "log", or "yj"
    model : string
                name of the model to be trained
    batch_frac : float
                batch fraction
    learning_rate : float
              learning rate
    epoch : int
            choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
    off_aug : string (AE_head or Gaussian_head or None)
            choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
    """

    path = "../Transfer/"
    save_model = "../Transfer/" + toname + "_from" + fromname + "_" + model + ".pt"
    ApplyExperiment(
        path=path,
        dataname=fromname,
        preprocess=preprocess,
        new_size=[fromsize],
        model=model,
        batch_frac=batch_frac,
        learning_rate=learning_rate,
        epoch=epoch,
        early_stop_num=30,
        off_aug=off_aug,
        AE_head_num=2,
        Gaussian_head_num=9,
        pre_model=None,
        save_model=save_model,
    )

    # training toname using pre-model
    pre_model = "../Transfer/" + toname + "_from" + fromname + "_" + model + ".pt"
    if pilot_size is not None:
        PilotExperiment(
            dataname=toname,
            pilot_size=pilot_size,
            model=model,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            pre_model=pre_model,
            epoch=epoch,
            off_aug=off_aug,
            early_stop_num=30,
            AE_head_num=2,
            Gaussian_head_num=9,
        )
    else:
        ApplyExperiment(
            path=path,
            dataname=toname,
            preprocess=preprocess,
            new_size=[new_size],
            model=model,
            batch_frac=batch_frac,
            learning_rate=learning_rate,
            epoch=epoch,
            early_stop_num=30,
            off_aug=off_aug,
            AE_head_num=2,
            Gaussian_head_num=9,
            pre_model=pre_model,
            save_model=None,
        )