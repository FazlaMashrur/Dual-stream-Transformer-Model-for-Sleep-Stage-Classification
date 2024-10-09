# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:38:40 2024

@author: yzhao
"""

import os

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold

from utils.preprocessing import reshape_sleep_data


def slice_data(data, sleep_scores, seq_len):
    n = len(data)
    n_to_crop = n % seq_len
    n_new_seq = (n - n_to_crop) // seq_len
    if n_to_crop != 0:
        data = np.concatenate([data[:-n_to_crop], data[-seq_len:]], axis=0)
        sleep_scores = np.concatenate(
            [sleep_scores[:-n_to_crop], sleep_scores[-seq_len:]], axis=0
        )
        n_new_seq += 1

    assert (n - n_to_crop) % seq_len == 0

    data = data.reshape(
        (n_new_seq, seq_len, data.shape[1], data.shape[2], data.shape[3])
    )
    sleep_scores = sleep_scores.reshape((n_new_seq, seq_len, sleep_scores.shape[1]))
    return [data, sleep_scores]


def prepare_data(mat_file, seq_len=64, augment=False, upsampling_scale=10):
    mat = loadmat(mat_file)
    eeg, emg, sleep_scores = reshape_sleep_data(mat)
    sleep_scores_len = len(sleep_scores)
    eeg_len = len(eeg)

    if sleep_scores_len >= eeg_len:
        sleep_scores = sleep_scores[:eeg_len]
    else:
        eeg = eeg[:sleep_scores_len]
        emg = emg[:sleep_scores_len]

    # standardize
    eeg_standardized = (eeg - np.mean(eeg)) / np.std(eeg)
    emg_standardized = (emg - np.mean(emg)) / np.std(emg)

    sleep_scores_reshaped = sleep_scores[:, np.newaxis]
    eeg_reshaped = eeg_standardized[:, np.newaxis, :]
    emg_reshaped = emg_standardized[:, np.newaxis, :]
    data = np.stack((eeg_reshaped, emg_reshaped), axis=1)
    sliced_data, sliced_sleep_scores = slice_data(data, sleep_scores_reshaped, seq_len)

    if augment:
        transition_indices = np.flatnonzero(np.diff(sleep_scores))
        REM_transition_indices = transition_indices[
            sleep_scores[transition_indices] == 2
        ]
        for transition_ind in REM_transition_indices:
            REM_sampling_range = np.arange(
                max(0, transition_ind - seq_len + 2),
                min(transition_ind + 1, sleep_scores_len - seq_len),
            )
            REM_sampling_start_inds = np.random.choice(
                REM_sampling_range, size=upsampling_scale, replace=False
            )
            REM_sampling_start_inds = np.expand_dims(REM_sampling_start_inds, axis=-1)
            REM_sampling_range = REM_sampling_start_inds + np.arange(seq_len)
            augmented_sample = data[REM_sampling_range]
            augmented_sleep_scores = sleep_scores_reshaped[REM_sampling_range]
            sliced_data = np.concatenate([sliced_data, augmented_sample], axis=0)
            sliced_sleep_scores = np.concatenate(
                [sliced_sleep_scores, augmented_sleep_scores], axis=0
            )

    return sliced_data, sliced_sleep_scores


def write_data(
    data_path,
    save_path,
    on_hold_list=[],
    fold=1,
    seq_len=64,
    augment=False,
    upsampling_scale=10,
):
    mat_list = []
    for file in os.listdir(data_path):
        if not file.endswith(".mat") or file in on_hold_list:
            continue

        mat_file = os.path.join(data_path, file)
        mat = loadmat(mat_file)
        eeg, emg, sleep_scores = reshape_sleep_data(mat)
        if np.isnan(sleep_scores).any() or np.any(sleep_scores == -1):
            continue

        mat_list.append(file)

    mat_list = sorted(mat_list)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(kf.split(mat_list))
    train_indices, val_indices = fold_indices[fold - 1]  # Label of fold start from one

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    train_file_list = []
    val_file_list = []
    for train_ind in train_indices:
        train_file_name = mat_list[train_ind]
        print(train_file_name)
        train_file_list.append(train_file_name)
        train_mat_file = os.path.join(data_path, train_file_name)
        sliced_data, sliced_sleep_scores = prepare_data(
            train_mat_file,
            seq_len=seq_len,
            augment=augment,
            upsampling_scale=upsampling_scale,
        )
        train_data.append(sliced_data)
        train_labels.append(sliced_sleep_scores)

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, f"train_trace{fold}.npy"), train_data)
    np.save(os.path.join(save_path, f"train_label{fold}.npy"), train_labels)
    print("saved train.")

    for val_ind in val_indices:
        val_file_name = mat_list[val_ind]
        print(val_file_name)
        val_file_list.append(val_file_name)
        val_mat_file = os.path.join(data_path, val_file_name)
        sliced_data, sliced_sleep_scores = prepare_data(val_mat_file)
        val_data.append(sliced_data)
        val_labels.append(sliced_sleep_scores)

    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    np.save(os.path.join(save_path, f"val_trace{fold}.npy"), val_data)
    np.save(os.path.join(save_path, f"val_label{fold}.npy"), val_labels)
    return train_file_list, val_file_list


# %%
if __name__ == "__main__":
    seq_len = 64  # don't change
    fold = 1  # don't change
    data_path = "C:/Users/yzhao/python_projects/time_series/data"  # path to the preprocessed data, ie., the .mat files
    save_path = f"C:/Users/yzhao/python_projects/time_series/sdreamer_data/n_seq_{seq_len}/fold_{fold}"  # where you want to save the train and val data
    # exclude files if needed
    on_hold_list = set(
        [
            "aud_392.mat",
            "aud_420.mat",
            "chr2_590_freq.mat",
            "sal_600.mat",
        ]
    )

    train_file_list, val_file_list = write_data(
        data_path, save_path, on_hold_list, fold=fold
    )
    with open(os.path.join(save_path, "train_val_split.txt"), "w") as outfile1:
        outfile1.write(
            "\n".join(
                ["## train_list"]
                + train_file_list
                + ["\n"]
                + ["## val_list"]
                + val_file_list
            )
        )
