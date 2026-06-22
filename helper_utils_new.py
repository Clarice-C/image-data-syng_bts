#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:22:40 2022

@author: yunhui, xinyi
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import torch.nn.functional as F


def preprocessinglog2(dataset):
    # log2 pre-processing of count data
    # If a column contains values <= -1, skip log transformation for that entire column
    # Otherwise, apply log2(x+1) transformation (no clipping needed if all values > -1)
    
    dataset_result = dataset.clone()  # Create a copy to avoid modifying original
    
    # Check each column for values <= -1
    n_features = dataset.shape[1]
    skip_cols = []
    
    for col_idx in range(n_features):
        col_data = dataset[:, col_idx]
        if torch.any(col_data <= -1):
            # This column has values <= -1, skip log transformation
            skip_cols.append(col_idx)
        else:
            # This column is safe for log transformation (all values > -1)
            dataset_result[:, col_idx] = torch.log2(col_data + 1)
    
    if len(skip_cols) > 0:
        print(f"Warning: {len(skip_cols)} columns skipped log2 transformation due to values <= -1: column indices {skip_cols}")
    
    return dataset_result


def set_all_seeds(seed):
    # set random seed
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_labels(n_samples, groups=None):
    # create binary labels and blurry labels for training two-group data
    set_all_seeds(10)  # randomness only for blur labels generation.
    if groups is None:
        labels = torch.zeros([n_samples, 1])
        blurlabels = labels
    else:
        base = groups[0]
        labels = torch.zeros([n_samples, 1]).to(torch.float32)
        labels[groups != base, 0] = 1
        blurlabels = torch.zeros([n_samples, 1]).to(torch.float32)
        blurlabels[groups != base, 0] = (10 - 9) * torch.rand(sum(groups != base)) + 9
        blurlabels[groups == base, 0] = (1 - 0) * torch.rand(sum(groups == base)) + 0
    return labels, blurlabels
    
def create_labels_mul(n_samples, groups=None):
    set_all_seeds(10)

    if groups is None:
        labels = torch.zeros([n_samples, 1], dtype=torch.float32)
        blurlabels = labels.clone()
        return labels, blurlabels

    groups_cat = groups.astype("category")
    codes = groups_cat.cat.codes
    group_tensor = torch.from_numpy(codes.copy().values)
    labels = group_tensor.float().unsqueeze(1) 
    blurlabels = labels + torch.rand_like(labels)
    return labels, blurlabels

def create_multi_conditional_labels(n_samples, groups=None, groups2=None):
    """
    Create multi-conditional labels for MultiCVAE.
    Combines groups (0/1) and groups2 (dynamically detected unique values) into a one-hot vector:
    y = [onehot(groups), onehot(groups2)]
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    groups : array-like, optional
        First condition variable (0 or 1)
    groups2 : array-like, optional
        Second condition variable (can be 0/1, or 0/1/2, or other values - automatically detected)
    
    Returns:
    --------
    labels : torch.Tensor
        Multi-conditional labels of shape (n_samples, 2 + groups2_dim)
        First 2 dims: one-hot encoding of groups
        Last groups2_dim dims: one-hot encoding of groups2
    blurlabels : torch.Tensor
        Blurred version of labels (for training stability)
    """
    set_all_seeds(10)
    
    # Determine groups2 dimension dynamically
    groups2_dim = 2  # Default to 2 (0 and 1)
    groups2_unique = None
    if groups2 is not None:
        groups2_array = np.array(groups2)
        groups2_unique = np.unique(groups2_array)
        groups2_dim = len(groups2_unique)
        # Ensure groups2_unique is sorted
        groups2_unique = np.sort(groups2_unique)
    
    total_dim = 2 + groups2_dim  # 2 for groups + groups2_dim for groups2
    
    if groups is None and groups2 is None:
        # No conditions provided, return zeros
        labels = torch.zeros([n_samples, total_dim], dtype=torch.float32)
        blurlabels = labels.clone()
        return labels, blurlabels
    
    labels = torch.zeros([n_samples, total_dim], dtype=torch.float32)
    
    # Encode groups (0/1) -> one-hot(2)
    if groups is not None:
        groups = np.array(groups)
        # One-hot encode groups: [1,0] for 0, [0,1] for 1
        labels[groups == 0, 0] = 1.0  # groups=0 -> [1,0,...]
        labels[groups == 1, 1] = 1.0  # groups=1 -> [0,1,...]
    else:
        # If groups not provided, set first dim to 1 (default to group 0)
        labels[:, 0] = 1.0
    
    # Encode groups2 dynamically based on unique values
    if groups2 is not None:
        groups2 = np.array(groups2)
        # One-hot encode groups2 based on unique values
        for idx, val in enumerate(groups2_unique):
            labels[groups2 == val, 2 + idx] = 1.0
    else:
        # If groups2 not provided, set first groups2 dim to 1 (default to groups2=0)
        labels[:, 2] = 1.0
    
    # Create blurred labels (add small random noise)
    blurlabels = labels.clone()
    noise = torch.rand_like(blurlabels) * 0.1  # Small noise
    blurlabels = blurlabels + noise
    # Normalize to ensure it's still a valid probability distribution
    blurlabels = blurlabels / (blurlabels.sum(dim=1, keepdim=True) + 1e-8)
    
    return labels, blurlabels

def draw_pilot(dataset, labels, blurlabels, n_pilot, seednum):
    # draw pilot datasets
    set_all_seeds(
        seednum
    )  # each draw has its own seednum, so guaranteed that 25 replicated sets are not the same
    n_samples = dataset.shape[0]
    if torch.unique(labels).shape[0] == 1:
        shuffled_indices = torch.randperm(n_samples)
        pilot_indices = shuffled_indices[-n_pilot:]
        rawdata = dataset[pilot_indices, :]
        rawlabels = labels[pilot_indices, :]
        rawblurlabels = blurlabels[pilot_indices, :]
    else:
        base = labels[0, :]
        n_pilot_1 = n_pilot
        n_pilot_2 = n_pilot
        n_samples_1 = sum(labels[:, 0] == base)
        n_samples_2 = sum(labels[:, 0] != base)
        dataset_1 = dataset[labels[:, 0] == base, :]
        dataset_2 = dataset[labels[:, 0] != base, :]
        labels_1 = labels[labels[:, 0] == base, :]
        labels_2 = labels[labels[:, 0] != base, :]
        blurlabels_1 = blurlabels[labels[:, 0] == base, :]
        blurlabels_2 = blurlabels[labels[:, 0] != base, :]
        shuffled_indices_1 = torch.randperm(n_samples_1)
        pilot_indices_1 = shuffled_indices_1[-n_pilot_1:]
        rawdata_1 = dataset_1[pilot_indices_1, :]
        rawlabels_1 = labels_1[pilot_indices_1, :]
        rawblurlabels_1 = blurlabels_1[pilot_indices_1, :]
        shuffled_indices_2 = torch.randperm(n_samples_2)
        pilot_indices_2 = shuffled_indices_2[-n_pilot_2:]
        rawdata_2 = dataset_2[pilot_indices_2, :]
        rawlabels_2 = labels_2[pilot_indices_2, :]
        rawblurlabels_2 = blurlabels_2[pilot_indices_2, :]
        rawdata = torch.cat((rawdata_1, rawdata_2), dim=0)
        rawlabels = torch.cat((rawlabels_1, rawlabels_2), dim=0)
        rawblurlabels = torch.cat((rawblurlabels_1, rawblurlabels_2), dim=0)
    return rawdata, rawlabels, rawblurlabels


def Gaussian_aug(rawdata, rawlabels, multiplier):
    # Gaussian augmentation
    # This function performs offline augmentation by adding gaussian noise to the
    # log2 counts, rawdata is the data generated from draw_pilot(), so does rawlabels,
    # multiplier specifies the number of samples for each kind of label, must be a list if
    # unique labels > 1. This function generates rawdata and rawlabels again but with
    # gaussian augmented data with size multiplier*n_rawdata

    oriraw = rawdata
    orirawlabels = rawlabels
    for all_mult in multiplier:
        for mult in list(range(all_mult)):
            rawdata = torch.cat(
                (
                    rawdata,
                    oriraw
                    + torch.normal(
                        mean=0, std=1, size=(oriraw.shape[0], oriraw.shape[1])
                    ),
                ),
                dim=0,
            )
            rawlabels = torch.cat((rawlabels, orirawlabels), dim=0)

    return rawdata, rawlabels


def plot_training_loss(
    minibatch_losses, num_epochs, averaging_iterations=100, custom_label=""
):
    # Ensure minibatch_losses is a numpy array
    if not isinstance(minibatch_losses, np.ndarray):
        minibatch_losses = np.array(minibatch_losses)
    
    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        range(len(minibatch_losses)),
        (minibatch_losses),
        label=f"Minibatch Loss{custom_label}",
    )
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")

    if len(minibatch_losses) < 1001:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    # Handle NaN and Inf values to avoid plotting errors
    loss_slice = np.array(minibatch_losses[num_losses:])  # Ensure it is a numpy array
    # Filter out NaN and Inf values
    finite_mask = np.isfinite(loss_slice)
    valid_losses = loss_slice[finite_mask]
    
    if len(valid_losses) > 0:
        max_loss = np.max(valid_losses)
        if np.isfinite(max_loss) and max_loss > 0:
            ax1.set_ylim([0, max_loss * 1.5])
        else:
            # Use default range if all values are invalid
            ax1.set_ylim([0, 1])
    else:
        # Use default range if no valid values
        ax1.set_ylim([0, 1])

    ax1.plot(
        np.convolve(
            minibatch_losses,
            np.ones(
                averaging_iterations,
            )
            / averaging_iterations,
            mode="valid",
        ),
        label=f"Running Average{custom_label}",
    )
    ax1.legend()

    ###################
    # Set second x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


def plot_recons_samples(
    savepath, model, modelname, data_loader, n_features, plot=False
):
    # plot reconstructed samples heatmap and save reconstructed samples as .csv file

    orig_all = torch.zeros([1, n_features])
    decoded_all = torch.zeros([1, n_features])
    labels = torch.zeros(0, dtype=torch.long)
    multi_labels_list = []  # For MultiCVAE

    for batch_idx, (features, lab) in enumerate(data_loader):
        if modelname == "MultiCVAE":
            # For MultiCVAE, lab is already a 5-dim one-hot vector
            multi_labels_list.append(lab)
            # Extract groups for compatibility (not used for CVAE logic)
            labels_batch = lab[:, :2].argmax(dim=1)  # groups: 0 or 1
        else:
            # For regular CVAE, lab is 1-dim
            labels_batch = torch.argmax(lab, dim=1) if lab.dim() > 1 else lab.squeeze(1).long()
        labels = torch.cat((labels, labels_batch), dim=0)

        with torch.no_grad():
            if modelname == "CVAE" or modelname == "MultiCVAE":
                encoded, z_mean, z_log_var, decoded_images = model(features, lab)
            elif modelname == "VAE":
                encoded, z_mean, z_log_var, decoded_images = model(features)
            else:
                encoded, decoded_images = model(features)

        orig_all = torch.cat((orig_all, features), dim=0)
        decoded_all = torch.cat((decoded_all, decoded_images), dim=0)

    orig_all = orig_all[1:]
    decoded_all = decoded_all[1:]

    if modelname == "CVAE":
        labels = labels.unsqueeze(1).float()  # shape: (N,1)
        orig_all = torch.cat((orig_all, labels), dim=1)
        decoded_all = torch.cat((decoded_all, labels), dim=1)
    elif modelname == "MultiCVAE":
        # For MultiCVAE, extract groups and groups2 from multi-conditional labels
        if multi_labels_list:
            all_multi_labels = torch.cat(multi_labels_list, dim=0)
            # Extract groups (first 2 dims) and groups2 (last 3 dims)
            groups_col = all_multi_labels[:, :2].argmax(dim=1).unsqueeze(1).float()  # groups: 0 or 1
            groups2_col = all_multi_labels[:, 2:].argmax(dim=1).unsqueeze(1).float()  # groups2: 0, 1, or 2
            orig_all = torch.cat((orig_all, groups_col, groups2_col), dim=1)
            decoded_all = torch.cat((decoded_all, groups_col, groups2_col), dim=1)
    if plot:
        sns.heatmap(
            torch.cat((orig_all, decoded_all), dim=0).detach().numpy(), cmap="YlGnBu"
        )
        plt.show()

    if savepath is not None:
        # Use os.path for correct path handling on Windows and Linux
        savepath = str(savepath)
        savepath = os.path.normpath(savepath)
        # Get directory part
        directory = os.path.dirname(savepath)
        # Create directory if not empty and does not exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print("Directory created: " + directory)

        file_path = savepath
        # used to be savepath instead of file_path
        np.savetxt(
            file_path,
            torch.cat((orig_all, decoded_all), dim=0).detach().numpy(),
            delimiter=",",
        )
    else:
        return torch.cat((orig_all, decoded_all), dim=0).detach(), labels


# def plot_latent_space_with_labels(num_classes, data_loader, encoding_fn):
#     d = {i:[] for i in range(num_classes)}

#     with torch.no_grad():
#         for i, (features,targets) in enumerate(data_loader):
#             embedding = encoding_fn(features)
#             for i in range(num_classes):
#                 if i in targets:
#                     mask = targets == i
#                     d[i].append(embedding[mask].numpy())

#     colors = list(mcolors.TABLEAU_COLORS.items())
#     for i in range(num_classes):
#         d[i] = np.concatenate(d[i])
#         plt.scatter(
#             d[i][:, 0], d[i][:, 1],
#             color=colors[i][1],
#             label=f'{i}',
#             alpha=0.5)

#     plt.legend()


def plot_new_samples(
    model, modelname, savepathnew, latent_size, num_images, plot=False, colnames = None, rawlabels=None,
    preprocess=None, preprocess_params_path=None, preprocess_params_df=None, feat_cols=None
):
    # plot new samples heatmap and save new samples as .csv file
    
    # Initialize new_images to avoid UnboundLocalError
    new_images = None

    with torch.no_grad():
        ##########################
        ###### RANDOM SAMPLE #####
        ##########################
        # Check if num_images is a list or integer
        if isinstance(num_images, list):
            if len(num_images) == 0:
                raise ValueError("num_images cannot be an empty list")
            elif len(num_images) == 1:
                # Single integer in list, convert to int
                num_images_int = num_images[0]
                rand_features = torch.randn(num_images_int, latent_size)
                if modelname == "CVAE":
                    num_classes = model.num_classes
                    base = num_images_int // num_classes
                    rem = num_images_int % num_classes
                    counts = [base]*num_classes
                    for i in range(rem):
                        counts[i] += 1
                    labels_list = []
                    for class_id, n_c in enumerate(counts):
                        ids = torch.full((n_c,), fill_value=class_id, dtype=torch.float32)
                        labels_list.append(ids)
                    one_group_labels = torch.cat(labels_list)
                    labels = one_group_labels.unsqueeze(1)  # shape = [N, 1]
                    
                    rand_features = torch.cat((rand_features, labels), dim=1)
                    new_images = model.decoder(rand_features)
                    new_images = torch.cat((new_images, labels), dim=1)
                elif modelname == "MultiCVAE":
                    # MultiCVAE: Generate samples for all combinations of (groups, groups2)
                    # groups: 0 or 1 (2 options)
                    # groups2: dynamically determined from condition_dim
                    condition_dim = model.condition_dim  # Should be 2 + groups2_dim
                    groups2_dim = condition_dim - 2  # groups2 dimension
                    num_groups = 2  # groups: 0 or 1
                    num_combinations = num_groups * groups2_dim
                    
                    base = num_images_int // num_combinations
                    rem = num_images_int % num_combinations
                    counts = [base] * num_combinations
                    for i in range(rem):
                        counts[i] += 1
                    
                    # Create multi-conditional labels for each combination
                    labels_list = []
                    # Dynamically generate all combinations: (groups, groups2)
                    combinations = [(g, g2) for g in range(num_groups) for g2 in range(groups2_dim)]
                    for combo_idx, (g, g2) in enumerate(combinations):
                        n_c = counts[combo_idx]
                        if n_c > 0:
                            # Create one-hot encoding: [onehot(groups), onehot(groups2)]
                            combo_labels = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                            combo_labels[:, g] = 1.0  # groups one-hot (position 0 or 1)
                            combo_labels[:, 2 + g2] = 1.0  # groups2 one-hot (position 2, 3, ...)
                            labels_list.append(combo_labels)
                    
                    if labels_list:
                        labels = torch.cat(labels_list, dim=0)  # shape = [N, condition_dim]
                    else:
                        labels = torch.zeros((num_images_int, condition_dim), dtype=torch.float32)
                    
                    rand_features = torch.cat((rand_features, labels), dim=1)
                    new_images = model.decoder(rand_features)
                    # Append both groups and groups2 columns for saving
                    groups_col = labels[:, :2].argmax(dim=1).unsqueeze(1).float()  # Extract groups (0 or 1)
                    groups2_col = labels[:, 2:].argmax(dim=1).unsqueeze(1).float()  # Extract groups2 (0, 1, ...)
                    new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                elif modelname == "AE":
                    new_images = model.decoder(rand_features)
                elif modelname == "VAE":
                    new_images = model.decoder(rand_features)
                elif modelname == "GANs":
                    new_images = model.generator(rand_features)
                elif modelname == "glow":
                    if rawlabels is not None:
                        # For conditional Flow, generate labels for all combinations
                        condition_dim = rawlabels.shape[1] if rawlabels.dim() > 1 else 1
                        
                        # Dynamically determine combinations based on condition_dim
                        # condition_dim = 2 (groups) + groups2_dim
                        if condition_dim >= 2:
                            groups2_dim = condition_dim - 2
                            
                            # Generate all possible combinations
                            # groups: 0 or 1 (2 values)
                            # groups2: 0 to groups2_dim-1 (groups2_dim values)
                            combinations = [(g, g2) for g in [0, 1] for g2 in range(groups2_dim)]
                            num_combinations = len(combinations)
                        else:
                            # Single condition (only groups)
                            num_combinations = 2
                            combinations = [(0,), (1,)]
                        
                        base = num_images_int // num_combinations
                        rem = num_images_int % num_combinations
                        counts = [base] * num_combinations
                        for i in range(rem):
                            counts[i] += 1
                        
                        # Create conditional labels for each combination
                        cond_labels_list = []
                        for combo_idx, combo in enumerate(combinations):
                            n_c = counts[combo_idx]
                            if n_c > 0:
                                if condition_dim >= 2:
                                    g, g2 = combo
                                    combo_cond = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                                    combo_cond[:, g] = 1.0  # groups one-hot
                                    combo_cond[:, 2 + g2] = 1.0  # groups2 one-hot
                                else:
                                    # Single condition (condition_dim=1): use value 0 or 1 directly
                                    combo_cond = torch.zeros((n_c, 1), dtype=torch.float32)
                                    combo_cond[:, 0] = float(combo[0])
                                cond_labels_list.append(combo_cond)
                        
                        if cond_labels_list:
                            cond_labels = torch.cat(cond_labels_list, dim=0)
                        else:
                            cond_labels = torch.zeros((num_images_int, condition_dim), dtype=torch.float32)
                        
                        new_images = model.sample(num_images_int, cond_inputs=cond_labels)
                        # Append groups and groups2 columns
                        if condition_dim >= 2:
                            groups_col = cond_labels[:, :2].argmax(dim=1).unsqueeze(1).float()
                            groups2_col = cond_labels[:, 2:].argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                        else:
                            groups_col = cond_labels.argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col), dim=1)
                    else:
                        new_images = model.sample(num_images_int)
                elif modelname == "realnvp":
                    if rawlabels is not None:
                        # Same logic as glow
                        condition_dim = rawlabels.shape[1] if rawlabels.dim() > 1 else 1
                        
                        # Dynamically determine combinations based on condition_dim
                        # condition_dim = 2 (groups) + groups2_dim
                        if condition_dim >= 2:
                            groups2_dim = condition_dim - 2
                            
                            # Generate all possible combinations
                            # groups: 0 or 1 (2 values)
                            # groups2: 0 to groups2_dim-1 (groups2_dim values)
                            combinations = [(g, g2) for g in [0, 1] for g2 in range(groups2_dim)]
                            num_combinations = len(combinations)
                        else:
                            # Single condition (only groups)
                            num_combinations = 2
                            combinations = [(0,), (1,)]
                        
                        base = num_images_int // num_combinations
                        rem = num_images_int % num_combinations
                        counts = [base] * num_combinations
                        for i in range(rem):
                            counts[i] += 1
                        
                        cond_labels_list = []
                        for combo_idx, combo in enumerate(combinations):
                            n_c = counts[combo_idx]
                            if n_c > 0:
                                if condition_dim >= 2:
                                    g, g2 = combo
                                    combo_cond = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                                    combo_cond[:, g] = 1.0  # groups one-hot
                                    combo_cond[:, 2 + g2] = 1.0  # groups2 one-hot
                                else:
                                    # Single condition (condition_dim=1): use value 0 or 1 directly
                                    combo_cond = torch.zeros((n_c, 1), dtype=torch.float32)
                                    combo_cond[:, 0] = float(combo[0])
                                cond_labels_list.append(combo_cond)
                        
                        if cond_labels_list:
                            cond_labels = torch.cat(cond_labels_list, dim=0)
                        else:
                            cond_labels = torch.zeros((num_images_int, condition_dim), dtype=torch.float32)
                        
                        new_images = model.sample(num_images_int, cond_inputs=cond_labels)
                        # Append groups and groups2 columns
                        if condition_dim >= 2:
                            groups_col = cond_labels[:, :2].argmax(dim=1).unsqueeze(1).float()
                            groups2_col = cond_labels[:, 2:].argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                        else:
                            groups_col = cond_labels.argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col), dim=1)
                    else:
                        new_images = model.sample(num_images_int)
                elif modelname == "maf":
                    if rawlabels is not None:
                        # Same logic as glow
                        condition_dim = rawlabels.shape[1] if rawlabels.dim() > 1 else 1
                        
                        # Dynamically determine combinations based on condition_dim
                        # condition_dim = 2 (groups) + groups2_dim
                        if condition_dim >= 2:
                            groups2_dim = condition_dim - 2
                            
                            # Generate all possible combinations
                            # groups: 0 or 1 (2 values)
                            # groups2: 0 to groups2_dim-1 (groups2_dim values)
                            combinations = [(g, g2) for g in [0, 1] for g2 in range(groups2_dim)]
                            num_combinations = len(combinations)
                        else:
                            # Single condition (only groups)
                            num_combinations = 2
                            combinations = [(0,), (1,)]
                        
                        base = num_images_int // num_combinations
                        rem = num_images_int % num_combinations
                        counts = [base] * num_combinations
                        for i in range(rem):
                            counts[i] += 1
                        
                        cond_labels_list = []
                        for combo_idx, combo in enumerate(combinations):
                            n_c = counts[combo_idx]
                            if n_c > 0:
                                if condition_dim >= 2:
                                    g, g2 = combo
                                    combo_cond = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                                    combo_cond[:, g] = 1.0  # groups one-hot
                                    combo_cond[:, 2 + g2] = 1.0  # groups2 one-hot
                                else:
                                    # Single condition (condition_dim=1): use value 0 or 1 directly
                                    combo_cond = torch.zeros((n_c, 1), dtype=torch.float32)
                                    combo_cond[:, 0] = float(combo[0])
                                cond_labels_list.append(combo_cond)
                        
                        if cond_labels_list:
                            cond_labels = torch.cat(cond_labels_list, dim=0)
                        else:
                            cond_labels = torch.zeros((num_images_int, condition_dim), dtype=torch.float32)
                        
                        new_images = model.sample(num_images_int, cond_inputs=cond_labels)
                        # Append groups and groups2 columns
                        if condition_dim >= 2:
                            groups_col = cond_labels[:, :2].argmax(dim=1).unsqueeze(1).float()
                            groups2_col = cond_labels[:, 2:].argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                        else:
                            groups_col = cond_labels.argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col), dim=1)
                    else:
                        new_images = model.sample(num_images_int)
                else:
                    # For other models when num_images is a single integer
                    new_images = model.decoder(rand_features) if hasattr(model, 'decoder') else model(rand_features)
            elif len(num_images) > 1:
                # if new_size = num_images = [n_for_0, n_for_1, ... , n_for_(num_classes-1), replicate]
                counts = num_images[:-1]
                repli = num_images[-1]
                num_images_repe = sum(counts)
                num_images_total = num_images_repe * repli
                rand_features = torch.randn(num_images_total, latent_size)
                
                # Debug: print modelname to verify
                print(f"DEBUG: In plot_new_samples, modelname='{modelname}', type={type(modelname)}, len(num_images)={len(num_images)}")
                print(f"DEBUG: modelname == 'MultiCVAE': {modelname == 'MultiCVAE'}")
                print(f"DEBUG: modelname == 'CVAE': {modelname == 'CVAE'}")
                
                if modelname == "CVAE":
                    num_classes = model.num_classes
                    if len(num_images) != num_classes + 1:
                        raise ValueError("num_images should have length num_classes+1")

                    labels_list = []
                    for class_id, n_c in enumerate(counts):
                        ids = torch.full((n_c,), fill_value=class_id, dtype=torch.float32)
                        labels_list.append(ids)
                    one_group_labels = torch.cat(labels_list)
                    labels = one_group_labels.repeat(repli).unsqueeze(1)  # shape = [N, 1]

                    rand_features = torch.cat((rand_features, labels), dim=1)
                    new_images = model.decoder(rand_features)
                    new_images = torch.cat((new_images, labels), dim=1)
                elif modelname == "MultiCVAE":
                    # MultiCVAE: num_images should have length 7 (6 combinations + 1 replicate)
                    print(f"DEBUG: Entering MultiCVAE branch, len(num_images)={len(num_images)}, counts={counts}, repli={repli}")
                    condition_dim = model.condition_dim  # Should be 5
                    num_combinations = 6
                    if len(num_images) != num_combinations + 1:
                        raise ValueError(f"num_images should have length {num_combinations + 1} for MultiCVAE (6 combinations + 1 replicate), got {len(num_images)}")
                    print(f"DEBUG: MultiCVAE validation passed, condition_dim={condition_dim}")

                    labels_list = []
                    combinations = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]  # (groups, groups2)
                    print(f"DEBUG: Creating labels for {len(combinations)} combinations")
                    for combo_idx, (g, g2) in enumerate(combinations):
                        n_c = counts[combo_idx]
                        print(f"DEBUG: Combination {combo_idx}: (groups={g}, groups2={g2}), n_c={n_c}")
                        if n_c > 0:
                            # Create one-hot encoding: [onehot(groups), onehot(groups2)]
                            combo_labels = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                            combo_labels[:, g] = 1.0  # groups one-hot
                            combo_labels[:, 2 + g2] = 1.0  # groups2 one-hot
                            labels_list.append(combo_labels)
                    
                    print(f"DEBUG: Created {len(labels_list)} label tensors")
                    if labels_list:
                        one_cycle_labels = torch.cat(labels_list, dim=0)  # shape = [sum(counts), 5]
                        print(f"DEBUG: one_cycle_labels shape: {one_cycle_labels.shape}")
                        labels = one_cycle_labels.repeat(repli, 1)  # shape = [N, 5]
                        print(f"DEBUG: labels shape after repeat: {labels.shape}")
                    else:
                        labels = torch.zeros((num_images_total, condition_dim), dtype=torch.float32)
                        print(f"DEBUG: Using zero labels, shape: {labels.shape}")

                    print(f"DEBUG: Concatenating rand_features and labels, rand_features shape: {rand_features.shape}, labels shape: {labels.shape}")
                    rand_features = torch.cat((rand_features, labels), dim=1)
                    print(f"DEBUG: Calling model.decoder, input shape: {rand_features.shape}")
                    new_images = model.decoder(rand_features)
                    print(f"DEBUG: Decoded images shape: {new_images.shape}")
                    # Append both groups and groups2 columns for saving
                    groups_col = labels[:, :2].argmax(dim=1).unsqueeze(1).float()
                    groups2_col = labels[:, 2:].argmax(dim=1).unsqueeze(1).float()
                    print(f"DEBUG: groups_col shape: {groups_col.shape}, groups2_col shape: {groups2_col.shape}")
                    new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                    print(f"DEBUG: Final new_images shape: {new_images.shape}")
                elif modelname == "AE":
                    new_images = model.decoder(rand_features)
                elif modelname == "VAE":
                    new_images = model.decoder(rand_features)
                elif modelname == "GANs":
                    new_images = model.generator(rand_features)
                elif modelname == "glow":
                    if rawlabels is not None:
                        # For conditional Flow with specified counts
                        condition_dim = rawlabels.shape[1] if rawlabels.dim() > 1 else 1
                        
                        # Dynamically determine combinations based on condition_dim
                        # condition_dim = 2 (groups) + groups2_dim
                        if condition_dim >= 2:
                            groups2_dim = condition_dim - 2
                            
                            # Generate all possible combinations
                            # groups: 0 or 1 (2 values)
                            # groups2: 0 to groups2_dim-1 (groups2_dim values)
                            combinations = [(g, g2) for g in [0, 1] for g2 in range(groups2_dim)]
                            num_combinations = len(combinations)
                        else:
                            # Single condition (only groups)
                            num_combinations = 2
                            combinations = [(0,), (1,)]
                        
                        if len(num_images) != num_combinations + 1:
                            raise ValueError(f"num_images should have length {num_combinations + 1} for conditional Flow ({num_combinations} combinations + 1 replicate)")
                        
                        cond_labels_list = []
                        for combo_idx, combo in enumerate(combinations):
                            n_c = counts[combo_idx]
                            if n_c > 0:
                                if condition_dim >= 2:
                                    g, g2 = combo
                                    combo_cond = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                                    combo_cond[:, g] = 1.0  # groups one-hot
                                    combo_cond[:, 2 + g2] = 1.0  # groups2 one-hot
                                else:
                                    # Single condition (condition_dim=1): use value 0 or 1 directly
                                    combo_cond = torch.zeros((n_c, 1), dtype=torch.float32)
                                    combo_cond[:, 0] = float(combo[0])
                                cond_labels_list.append(combo_cond)
                        
                        if cond_labels_list:
                            one_cycle_cond = torch.cat(cond_labels_list, dim=0)
                            cond_labels = one_cycle_cond.repeat(repli, 1)
                        else:
                            cond_labels = torch.zeros((num_images_total, condition_dim), dtype=torch.float32)
                        
                        new_images = model.sample(num_images_total, cond_inputs=cond_labels)
                        # Append groups and groups2 columns
                        if condition_dim >= 2:
                            groups_col = cond_labels[:, :2].argmax(dim=1).unsqueeze(1).float()
                            groups2_col = cond_labels[:, 2:].argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                        else:
                            groups_col = cond_labels.argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col), dim=1)
                    else:
                        new_images = model.sample(num_images_total)
                elif modelname == "realnvp":
                    if rawlabels is not None:
                        # Same logic as glow
                        condition_dim = rawlabels.shape[1] if rawlabels.dim() > 1 else 1
                        
                        # Dynamically determine combinations based on condition_dim
                        # condition_dim = 2 (groups) + groups2_dim
                        if condition_dim >= 2:
                            groups2_dim = condition_dim - 2
                            
                            # Generate all possible combinations
                            # groups: 0 or 1 (2 values)
                            # groups2: 0 to groups2_dim-1 (groups2_dim values)
                            combinations = [(g, g2) for g in [0, 1] for g2 in range(groups2_dim)]
                            num_combinations = len(combinations)
                        else:
                            # Single condition (only groups)
                            num_combinations = 2
                            combinations = [(0,), (1,)]
                        
                        if len(num_images) != num_combinations + 1:
                            raise ValueError(f"num_images should have length {num_combinations + 1} for conditional Flow ({num_combinations} combinations + 1 replicate)")
                        
                        cond_labels_list = []
                        for combo_idx, combo in enumerate(combinations):
                            n_c = counts[combo_idx]
                            if n_c > 0:
                                if condition_dim >= 2:
                                    g, g2 = combo
                                    combo_cond = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                                    combo_cond[:, g] = 1.0  # groups one-hot
                                    combo_cond[:, 2 + g2] = 1.0  # groups2 one-hot
                                else:
                                    # Single condition (condition_dim=1): use value 0 or 1 directly
                                    combo_cond = torch.zeros((n_c, 1), dtype=torch.float32)
                                    combo_cond[:, 0] = float(combo[0])
                                cond_labels_list.append(combo_cond)
                        
                        if cond_labels_list:
                            one_cycle_cond = torch.cat(cond_labels_list, dim=0)
                            cond_labels = one_cycle_cond.repeat(repli, 1)
                        else:
                            cond_labels = torch.zeros((num_images_total, condition_dim), dtype=torch.float32)
                        
                        new_images = model.sample(num_images_total, cond_inputs=cond_labels)
                        # Append groups and groups2 columns
                        if condition_dim >= 2:
                            groups_col = cond_labels[:, :2].argmax(dim=1).unsqueeze(1).float()
                            groups2_col = cond_labels[:, 2:].argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                        else:
                            groups_col = cond_labels.argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col), dim=1)
                    else:
                        new_images = model.sample(num_images_total)
                elif modelname == "maf":
                    if rawlabels is not None:
                        # Same logic as glow
                        condition_dim = rawlabels.shape[1] if rawlabels.dim() > 1 else 1
                        
                        # Dynamically determine combinations based on condition_dim
                        # condition_dim = 2 (groups) + groups2_dim
                        if condition_dim >= 2:
                            groups2_dim = condition_dim - 2
                            
                            # Generate all possible combinations
                            # groups: 0 or 1 (2 values)
                            # groups2: 0 to groups2_dim-1 (groups2_dim values)
                            combinations = [(g, g2) for g in [0, 1] for g2 in range(groups2_dim)]
                            num_combinations = len(combinations)
                        else:
                            # Single condition (only groups)
                            num_combinations = 2
                            combinations = [(0,), (1,)]
                        
                        if len(num_images) != num_combinations + 1:
                            raise ValueError(f"num_images should have length {num_combinations + 1} for conditional Flow ({num_combinations} combinations + 1 replicate)")
                        
                        cond_labels_list = []
                        for combo_idx, combo in enumerate(combinations):
                            n_c = counts[combo_idx]
                            if n_c > 0:
                                if condition_dim >= 2:
                                    g, g2 = combo
                                    combo_cond = torch.zeros((n_c, condition_dim), dtype=torch.float32)
                                    combo_cond[:, g] = 1.0  # groups one-hot
                                    combo_cond[:, 2 + g2] = 1.0  # groups2 one-hot
                                else:
                                    # Single condition (condition_dim=1): use value 0 or 1 directly
                                    combo_cond = torch.zeros((n_c, 1), dtype=torch.float32)
                                    combo_cond[:, 0] = float(combo[0])
                                cond_labels_list.append(combo_cond)
                        
                        if cond_labels_list:
                            one_cycle_cond = torch.cat(cond_labels_list, dim=0)
                            cond_labels = one_cycle_cond.repeat(repli, 1)
                        else:
                            cond_labels = torch.zeros((num_images_total, condition_dim), dtype=torch.float32)
                        
                        new_images = model.sample(num_images_total, cond_inputs=cond_labels)
                        # Append groups and groups2 columns
                        if condition_dim >= 2:
                            groups_col = cond_labels[:, :2].argmax(dim=1).unsqueeze(1).float()
                            groups2_col = cond_labels[:, 2:].argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col, groups2_col), dim=1)
                        else:
                            groups_col = cond_labels.argmax(dim=1).unsqueeze(1).float()
                            new_images = torch.cat((new_images, groups_col), dim=1)
                    else:
                        new_images = model.sample(num_images_total)
                else:
                    # For other models when num_images is a list with multiple elements
                    new_images = model.decoder(rand_features) if hasattr(model, 'decoder') else model(rand_features)
        else:
            # num_images is not a list (shouldn't happen, but handle it)
            raise ValueError(f"num_images must be a list, got {type(num_images)}")

        ##########################
        ### VISUALIZATION
        ##########################
        # Check if new_images was defined
        if new_images is None:
            raise ValueError(f"new_images was not defined. modelname={modelname}, num_images={num_images}, type={type(num_images)}")
        
        # last column of saved data is the labels: 0 for MXF, 20 for PMFH
        # either generated for VAE or setted for CVAE
        if plot:
            sns.heatmap(new_images.detach().numpy(), cmap="YlGnBu")
            plt.show()

        if savepathnew is not None:
            # Apply non-negativity clamp when using YJ/zscore preprocessing (params in-memory or from file)
            params_df = preprocess_params_df
            if params_df is None and preprocess == "yj" and preprocess_params_path and feat_cols is not None:
                from preprocess_utils import load_preprocess_params
                params_df = load_preprocess_params(preprocess_params_path)
            if params_df is not None and feat_cols is not None:
                from preprocess_utils import clamp_for_nonneg
                n_feat = len(feat_cols)
                X_feat = new_images[:, :n_feat].detach().numpy()
                X_clamped = clamp_for_nonneg(X_feat, params_df, feat_cols)
                new_images = new_images.clone()
                new_images[:, :n_feat] = torch.from_numpy(X_clamped).to(new_images.dtype)
            # Use os.path for correct path handling on Windows and Linux
            savepathnew = str(savepathnew)
            # Normalize path (handle ./ and ../ etc.)
            savepathnew = os.path.normpath(savepathnew)
            # Get directory part
            directory = os.path.dirname(savepathnew)
            # Create directory if not empty and does not exist
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print("Directory created: " + directory)
            # Save file
            np.savetxt(savepathnew, new_images.detach().numpy(), delimiter=",")
        else:
            return new_images


def plot_multiple_training_losses(
    losses_list, num_epochs, averaging_iterations=100, custom_labels_list=None
):
    for i, _ in enumerate(losses_list):
        if not len(losses_list[i]) == len(losses_list[0]):
            raise ValueError(
                "All loss tensors need to have the same number of elements."
            )

    if custom_labels_list is None:
        custom_labels_list = [str(i) for i, _ in enumerate(custom_labels_list)]

    iter_per_epoch = len(losses_list[0]) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    for i, minibatch_loss_tensor in enumerate(losses_list):
        ax1.plot(
            range(len(minibatch_loss_tensor)),
            (minibatch_loss_tensor),
            label=f"Minibatch Loss{custom_labels_list[i]}",
        )
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")

        ax1.plot(
            np.convolve(
                minibatch_loss_tensor,
                np.ones(
                    averaging_iterations,
                )
                / averaging_iterations,
                mode="valid",
            ),
            color="black",
        )

    if len(losses_list[0]) < 1000:
        num_losses = len(losses_list[0]) // 2
    else:
        num_losses = 1000
    # Handle NaN and Inf values to avoid plotting errors
    maxes = []
    for i, _ in enumerate(losses_list):
        # Ensure it is a numpy array
        loss_array = np.array(losses_list[i]) if not isinstance(losses_list[i], np.ndarray) else losses_list[i]
        loss_slice = np.array(loss_array[num_losses:])
        # Filter out NaN and Inf values
        finite_mask = np.isfinite(loss_slice)
        valid_losses = loss_slice[finite_mask]
        if len(valid_losses) > 0:
            max_val = np.max(valid_losses)
            if np.isfinite(max_val):
                maxes.append(max_val)
    
    if len(maxes) > 0:
        max_loss = np.max(maxes)
        if np.isfinite(max_loss) and max_loss > 0:
            ax1.set_ylim([0, max_loss * 1.5])
        else:
            ax1.set_ylim([0, 1])
    else:
        ax1.set_ylim([0, 1])
    ax1.legend()

    ###################
    # Set second x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
