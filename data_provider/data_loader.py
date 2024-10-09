import os
import logging
from glob import glob

import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

# from torchvision import transforms, datasets
# from pathlib import Path
# from torch.utils import data
# import matplotlib.pyplot as plt


def file2tensor(file_path, norm=False, isLabel=False):
    data = torch.from_numpy(np.load(file_path))
    if not isLabel and norm:
        mean, std = torch.mean(data, dim=0), torch.std(data, dim=0)
        data = (data - mean) / std
    return data
    # return data if not isLabel else data.unsqueeze(1)


def filter_func(data_list, label):
    return list(map(lambda tensor: tensor[torch.where(label[:, 0] >= 0)], data_list))


# For One-to-one Classification
class Epoch_Loader(Dataset):
    def __init__(
        self,
        root_path="data/raw_data/",
        data_path="data/dst_data/epoch/",
        isEval=False,
        fold=1,
        n_sequences=1,
        useNorm=False,
    ):
        self.root_path = root_path
        self.dst_path = "{}fold_{}/".format(data_path, fold)

        if not os.path.exists(self.dst_path):
            print(
                ">>>>>>>>Starting Processing and Splitting Raw Data Fold{}<<<<<<<<<<<<<<<<<<<".format(
                    fold
                )
            )
            os.makedirs(self.dst_path)
            os.system("rm -rf {}".format(self.dst_path))
            trace_files = sorted(glob(self.root_path + "*data.npy"))
            label_files = sorted(glob(self.root_path + "*label.npy"))

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_idxs = list(kf.split(trace_files))
            train_idxs, val_idxs = fold_idxs[fold - 1]  # Label of fold start from one

            print("Train_idxs: ", train_idxs)
            print("Val_idxs: ", val_idxs)

            train_traces = torch.cat(
                [file2tensor(trace_files[idx]) for idx in train_idxs], dim=0
            ).float()
            train_norm = torch.cat(
                [file2tensor(trace_files[idx], norm=True) for idx in train_idxs], dim=0
            ).float()
            train_labels = torch.cat(
                [file2tensor(label_files[idx], isLabel=True) for idx in train_idxs],
                dim=0,
            )

            val_traces = torch.cat(
                [file2tensor(trace_files[idx]) for idx in val_idxs], dim=0
            ).float()
            val_norm = torch.cat(
                [file2tensor(trace_files[idx], norm=True) for idx in val_idxs], dim=0
            ).float()
            val_labels = torch.cat(
                [file2tensor(label_files[idx], isLabel=True) for idx in val_idxs], dim=0
            )

            train_traces, train_norm, self.train_labels = filter_func(
                [train_traces, train_norm, train_labels], train_labels
            )
            val_traces, val_norm, self.val_labels = filter_func(
                [val_traces, val_norm, val_labels], val_labels
            )
            # concat the normalized trace to channel dimmension
            self.train_traces = torch.cat([train_traces, train_norm], dim=2)
            self.val_traces = torch.cat([val_traces, val_norm], dim=2)
            # after concat [N, 2, 2, 512]  the thrid dim include pre-normed and normed data
            np.save(
                "{}train_trace{}.npy".format(self.dst_path, fold), self.train_traces
            )
            np.save(
                "{}train_label{}.npy".format(self.dst_path, fold), self.train_labels
            )
            np.save("{}val_trace{}.npy".format(self.dst_path, fold), self.val_traces)
            np.save("{}val_label{}.npy".format(self.dst_path, fold), self.val_labels)

        else:
            print(">>>>>>>>>Loading Existing Fold{}<<<<<<<<<<<<<<<<<<<<<<".format(fold))
            self.train_traces = torch.from_numpy(
                np.load(
                    "{}train_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.train_labels = torch.from_numpy(
                np.load(
                    "{}train_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_traces = torch.from_numpy(
                np.load(
                    "{}val_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_labels = torch.from_numpy(
                np.load(
                    "{}val_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )

        self.traces, self.labels = (
            (self.train_traces, self.train_labels)
            if not isEval
            else (self.val_traces, self.val_labels)
        )
        self.traces = self.traces[:, :, :1] if not useNorm else self.traces[:, :, -1:]

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]
        return trace, label


def filter_func_visualize(data_list, label):
    print(label.shape)
    return list(map(lambda tensor: tensor[torch.where(label[:, 0] >= 0)], data_list))


def build_traces_list(traces_list, length):
    ret = []
    stamp = []
    for i in range(len(traces_list)):
        item = traces_list[i]
        cur = 0
        while cur + length <= item.size(0):
            ret.append(item[cur : cur + length].unsqueeze(0))
            stamp.append([i, cur])
            cur += length
    ret = torch.cat(ret, dim=0)
    return ret, stamp


class Item_Loader(Dataset):
    def __init__(
        self,
        root_path="raw_data/",
        data_path="dst_data/epoch/",
        isEval=False,
        fold=1,
        n_sequences=1,
        useNorm=False,
    ):
        self.root_path = root_path
        self.dst_path = "{}fold_{}/visualize/".format(data_path, fold)

        if not os.path.exists(self.dst_path):
            print(
                ">>>>>>>>Starting Processing and Splitting Raw Data Fold{}<<<<<<<<<<<<<<<<<<<".format(
                    fold
                )
            )
            logging.getLogger("logger").info(
                f">>>>>>>>Starting Processing and Splitting Raw Data Fold{fold}<<<<<<<<<<<<<<<<<<<"
            )
            os.makedirs(self.dst_path)
            # os.system("rm -rf {}".format(self.dst_dir))
            trace_files = sorted(glob(self.root_path + "*data.npy"))
            label_files = sorted(glob(self.root_path + "*label.npy"))

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_idxs = list(kf.split(trace_files))
            train_idxs, val_idxs = fold_idxs[fold - 1]  # Label of fold start from one
            print("Train_idxs: ", train_idxs)
            print("Val_idxs: ", val_idxs)
            logging.getLogger("logger").info("Train_idxs: ", train_idxs)
            logging.getLogger("logger").info("Val_idxs: ", val_idxs)

            val_traces = [file2tensor(trace_files[idx]).float() for idx in val_idxs]
            val_norm = [
                file2tensor(trace_files[idx], norm=True).float() for idx in val_idxs
            ]
            val_labels = [
                file2tensor(label_files[idx], isLabel=True).float() for idx in val_idxs
            ]

            self.val_traces, val_stamp = build_traces_list(val_traces, 320)
            self.val_norm, _ = build_traces_list(val_norm, 320)
            self.val_labels, _ = build_traces_list(val_labels, 320)

            for i in range(len(val_stamp)):
                val_stamp[i][0] = val_idxs[val_stamp[i][0]]
            self.val_stamp = torch.tensor(val_stamp)

            # val_traces, val_norm, self.val_labels = filter_func_visualize([val_traces, val_norm, val_labels], val_labels)
            # concat the normalized trace to channel dimmension
            # self.val_traces = torch.cat([val_traces, val_norm], dim=2)
            # after concat [N, 2, 2, 512]  the thrid dim include pre-normed and normed data
            # np.save('{}train_trace{}.npy'.format(self.dst_path, fold), self.train_traces)
            # np.save('{}train_label{}.npy'.format(self.dst_path, fold), self.train_labels)
            np.save("{}val_trace{}.npy".format(self.dst_path, fold), self.val_traces)
            np.save("{}val_norm{}.npy".format(self.dst_path, fold), self.val_norm)
            np.save("{}val_label{}.npy".format(self.dst_path, fold), self.val_labels)
            np.save("{}val_stamp{}.npy".format(self.dst_path, fold), self.val_stamp)

        else:
            print(">>>>>>>>>Loading Existing Fold{}<<<<<<<<<<<<<<<<<<<<<<".format(fold))
            logging.getLogger("logger").info(
                f">>>>>>>>>Loading Existing Fold{fold}<<<<<<<<<<<<<<<<<<<<<<"
            )
            # self.train_traces = torch.from_numpy(np.load('{}train_trace{}.npy'.format(self.dst_path, fold), allow_pickle=True))
            # self.train_labels = torch.from_numpy(np.load('{}train_label{}.npy'.format(self.dst_path, fold), allow_pickle=True))
            self.val_traces = torch.from_numpy(
                np.load(
                    "{}val_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_labels = torch.from_numpy(
                np.load(
                    "{}val_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_norm = torch.from_numpy(
                np.load(
                    "{}val_norm{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_stamp = torch.from_numpy(
                np.load(
                    "{}val_stamp{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )

        self.traces, self.traces_norm, self.labels, self.stamp = (
            self.val_traces,
            self.val_norm,
            self.val_labels,
            self.val_stamp,
        )
        # self.traces = self.traces[:,:,:1] if not useNorm else self.traces[:,:,-1:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        trace_no_norm = self.traces[idx]
        trace = self.traces_norm[idx]
        label = self.labels[idx]
        stamp = self.stamp[idx]
        return trace_no_norm, trace, label, stamp


# dummy_data = Epoch_Loader(isEval=True, fold=1, useNorm=True)


def SeqFile2tensor(file_path, norm=False, isLabel=False):
    data = torch.from_numpy(np.load(file_path))
    if not isLabel and norm:
        mean, std = torch.mean(data, dim=0), torch.std(data, dim=0)
        data = (data - mean) / std
    return data if not isLabel else data.unsqueeze(1)


def Seq_filter_func(data_list, label_list):
    return list(
        map(
            lambda tensor, label: tensor[torch.where(label[:, 0] >= 0)],
            data_list,
            label_list,
        )
    )


def slice_trace(trace, norm, label, n_sequences):
    n = len(trace)
    n_to_crop = n % n_sequences
    n_new_seq = (n - n_to_crop) // n_sequences
    if n_to_crop != 0:
        trace = torch.cat([trace[:-n_to_crop], trace[-n_sequences:]], dim=0)
        norm = torch.cat([norm[:-n_to_crop], norm[-n_sequences:]], dim=0)
        label = torch.cat([label[:-n_to_crop], label[-n_sequences:]], dim=0)
        # trace = trace[:-n_to_crop]
        # norm = norm[:-n_to_crop]
        # label = label[:-n_to_crop]
        n_new_seq += 1

    assert (n - n_to_crop) % n_sequences == 0

    trace = trace.reshape(
        (n_new_seq, n_sequences, trace.shape[1], trace.shape[2], trace.shape[3])
    )
    norm = norm.reshape(
        (n_new_seq, n_sequences, norm.shape[1], norm.shape[2], norm.shape[3])
    )
    label = label.reshape((n_new_seq, n_sequences, label.shape[1]))
    return [trace, norm, label]


def slice_trace_wNE(trace, ne, norm, norm_ne, label, n_sequences):
    n = len(trace)
    n_to_crop = n % n_sequences
    if n_to_crop != 0:
        trace = trace[:-n_to_crop]
        ne = ne[:-n_to_crop]
        norm = norm[:-n_to_crop]
        norm_ne = norm_ne[:-n_to_crop]
        label = label[:-n_to_crop]
    assert (n - n_to_crop) % n_sequences == 0
    n_new_seq = (n - n_to_crop) // n_sequences
    trace = trace.reshape(
        (n_new_seq, n_sequences, trace.shape[1], trace.shape[2], trace.shape[3])
    )
    ne = ne.reshape((n_new_seq, n_sequences, ne.shape[1], ne.shape[2], ne.shape[3]))
    norm = norm.reshape(
        (n_new_seq, n_sequences, norm.shape[1], norm.shape[2], norm.shape[3])
    )
    norm_ne = norm_ne.reshape(
        (n_new_seq, n_sequences, norm_ne.shape[1], norm_ne.shape[2], norm_ne.shape[3])
    )
    label = label.reshape((n_new_seq, n_sequences, label.shape[1]))
    return [trace, ne, norm, norm_ne, label]


def Seq_slice_func(trace_list, norm_list, label_list, n_sequences):
    return list(
        map(
            lambda trace, norm, label: slice_trace(trace, norm, label, n_sequences),
            trace_list,
            norm_list,
            label_list,
        )
    )


def Seq_slice_func_wNE(
    trace_list, ne_list, norm_list, norm_ne_list, label_list, n_sequences
):
    return list(
        map(
            lambda trace, ne, norm, norm_ne, label: slice_trace_wNE(
                trace, ne, norm, norm_ne, label, n_sequences
            ),
            trace_list,
            ne_list,
            norm_list,
            norm_ne_list,
            label_list,
        )
    )


class Seq_Loader(Dataset):
    def __init__(
        self,
        root_path="raw_data/",
        data_path="dst_data/seq/",
        isEval=False,
        fold=1,
        n_sequences=1,
        useNorm=False,
    ):
        self.root_path = root_path
        self.dst_path = "{}n_seq_{}/fold_{}/".format(data_path, n_sequences, fold)

        if not os.path.exists(self.dst_path):
            print(
                ">>>>>>>>Starting Processing and Splitting Raw Data Fold{}<<<<<<<<<<<<<<<<<<<".format(
                    fold
                )
            )
            logging.getLogger("logger").info(
                f">>>>>>>>Starting Processing and Splitting Raw Data Fold{fold}<<<<<<<<<<<<<<<<<<<"
            )
            os.makedirs(self.dst_path)
            # os.system("rm -rf {}".format(self.dst_path))
            trace_files = sorted(glob(self.root_path + "*data.npy"))
            label_files = sorted(glob(self.root_path + "*label.npy"))

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_idxs = list(kf.split(trace_files))
            train_idxs, val_idxs = fold_idxs[fold - 1]  # Label of fold start from one

            train_traces_list = [file2tensor(trace_files[idx]) for idx in train_idxs]
            train_norm_list = [
                file2tensor(trace_files[idx], norm=True) for idx in train_idxs
            ]
            train_labels_list = [
                file2tensor(label_files[idx], isLabel=True) for idx in train_idxs
            ]

            train_traces_list = Seq_filter_func(train_traces_list, train_labels_list)
            train_norm_list = Seq_filter_func(train_norm_list, train_labels_list)
            train_labels_list = Seq_filter_func(train_labels_list, train_labels_list)
            train_data = Seq_slice_func(
                train_traces_list, train_norm_list, train_labels_list, n_sequences
            )
            train_traces = torch.cat([i[0] for i in train_data], dim=0).float()
            train_norm = torch.cat([i[1] for i in train_data], dim=0).float()
            self.train_labels = torch.cat([i[2] for i in train_data], dim=0)

            val_traces_list = [file2tensor(trace_files[idx]) for idx in val_idxs]
            val_norm_list = [
                file2tensor(trace_files[idx], norm=True) for idx in val_idxs
            ]
            val_labels_list = [
                file2tensor(label_files[idx], isLabel=True) for idx in val_idxs
            ]

            val_traces_list = Seq_filter_func(val_traces_list, val_labels_list)
            val_norm_list = Seq_filter_func(val_norm_list, val_labels_list)
            val_labels_list = Seq_filter_func(val_labels_list, val_labels_list)
            val_data = Seq_slice_func(
                val_traces_list, val_norm_list, val_labels_list, n_sequences
            )
            val_traces = torch.cat([i[0] for i in val_data], dim=0).float()
            val_norm = torch.cat([i[1] for i in val_data], dim=0).float()
            self.val_labels = torch.cat([i[2] for i in val_data], dim=0)

            # concat the normalized trace to channel dimmension
            self.train_traces = torch.cat([train_traces, train_norm], dim=3)
            self.val_traces = torch.cat([val_traces, val_norm], dim=3)

            # after concat -> [N, n_sequence, 2, 2, 512]. the 4th dim include pre-normed and normed data
            # print(self.dst_path)
            np.save(
                "{}train_trace{}.npy".format(self.dst_path, fold), self.train_traces
            )
            np.save(
                "{}train_label{}.npy".format(self.dst_path, fold), self.train_labels
            )
            np.save("{}val_trace{}.npy".format(self.dst_path, fold), self.val_traces)
            np.save("{}val_label{}.npy".format(self.dst_path, fold), self.val_labels)

        else:
            print(">>>>>>>>>Loading Existing Fold{}<<<<<<<<<<<<<<<<<<<<<<".format(fold))
            logging.getLogger("logger").info(
                f">>>>>>>>>Loading Existing Fold{fold}<<<<<<<<<<<<<<<<<<<<<<"
            )
            self.train_traces = torch.from_numpy(
                np.load(
                    "{}train_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.train_labels = torch.from_numpy(
                np.load(
                    "{}train_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_traces = torch.from_numpy(
                np.load(
                    "{}val_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_labels = torch.from_numpy(
                np.load(
                    "{}val_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )

        self.traces, self.labels = (
            (self.train_traces, self.train_labels)
            if not isEval
            else (self.val_traces, self.val_labels)
        )
        self.traces = (
            self.traces[:, :, :, :1] if not useNorm else self.traces[:, :, :, -1:]
        )
        # i=0

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]
        return trace, label


def file2tensor_wNE(file_path, ne_file_path, norm=False, isLabel=False):
    data = torch.from_numpy(np.load(file_path))
    data_NE = torch.from_numpy(np.load(ne_file_path))
    if not isLabel and norm:
        mean, std = torch.mean(data, dim=0), torch.std(data, dim=0)
        data = (data - mean) / std
    data = data[: data_NE.size(0)]
    return data if not isLabel else data.unsqueeze(1)


# For One-to-one Classification
class Epoch_Loader_NE(Dataset):
    def __init__(
        self,
        root_path="data/raw_data_wNE/",
        data_path="data/dst_data_wNE/epoch/",
        isEval=False,
        fold=1,
        n_sequences=1,
        useNorm=False,
    ):
        self.root_path = root_path
        self.dst_path = "{}fold_{}/".format(data_path, fold)

        if not os.path.exists(self.dst_path):
            print(
                ">>>>>>>>Starting Processing and Splitting Raw Data Fold{}<<<<<<<<<<<<<<<<<<<".format(
                    fold
                )
            )
            os.makedirs(self.dst_path)
            # os.system("rm -rf {}".format(self.dst_path))
            trace_files = sorted(glob(self.root_path + "*data.npy"))
            ne_files = sorted(glob(self.root_path + "*NE.npy"))
            label_files = sorted(glob(self.root_path + "*label.npy"))

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_idxs = list(kf.split(trace_files))
            train_idxs, val_idxs = fold_idxs[fold - 1]  # Label of fold start from one

            print("Train_idxs: ", train_idxs)
            print("Val_idxs: ", val_idxs)

            train_traces = torch.cat(
                [
                    file2tensor_wNE(trace_files[idx], ne_files[idx])
                    for idx in train_idxs
                ],
                dim=0,
            ).float()
            train_ne = torch.cat(
                [file2tensor_wNE(ne_files[idx], ne_files[idx]) for idx in train_idxs],
                dim=0,
            ).float()
            train_norm = torch.cat(
                [
                    file2tensor_wNE(trace_files[idx], ne_files[idx], norm=True)
                    for idx in train_idxs
                ],
                dim=0,
            ).float()
            trace_norm_ne = torch.cat(
                [
                    file2tensor_wNE(ne_files[idx], ne_files[idx], norm=True)
                    for idx in train_idxs
                ],
                dim=0,
            ).float()
            train_labels = torch.cat(
                [
                    file2tensor_wNE(label_files[idx], ne_files[idx], isLabel=True)
                    for idx in train_idxs
                ],
                dim=0,
            )

            val_traces = torch.cat(
                [file2tensor_wNE(trace_files[idx], ne_files[idx]) for idx in val_idxs],
                dim=0,
            ).float()
            val_ne = torch.cat(
                [file2tensor_wNE(ne_files[idx], ne_files[idx]) for idx in val_idxs],
                dim=0,
            ).float()
            val_norm = torch.cat(
                [
                    file2tensor_wNE(trace_files[idx], ne_files[idx], norm=True)
                    for idx in val_idxs
                ],
                dim=0,
            ).float()
            val_norm_ne = torch.cat(
                [
                    file2tensor_wNE(ne_files[idx], ne_files[idx], norm=True)
                    for idx in val_idxs
                ],
                dim=0,
            ).float()
            val_labels = torch.cat(
                [
                    file2tensor_wNE(label_files[idx], ne_files[idx], isLabel=True)
                    for idx in val_idxs
                ],
                dim=0,
            )

            (
                train_traces,
                train_norm,
                train_ne,
                trace_norm_ne,
                self.train_labels,
            ) = filter_func(
                [train_traces, train_norm, train_ne, trace_norm_ne, train_labels],
                train_labels,
            )
            val_traces, val_norm, val_ne, val_norm_ne, self.val_labels = filter_func(
                [val_traces, val_norm, val_ne, val_norm_ne, val_labels], val_labels
            )
            # concat the normalized trace to channel dimmension
            self.train_traces = torch.cat([train_traces, train_norm], dim=2)
            self.train_ne = torch.cat([train_ne, trace_norm_ne], dim=2)
            self.val_traces = torch.cat([val_traces, val_norm], dim=2)
            self.val_ne = torch.cat([val_ne, val_norm_ne], dim=2)
            # after concat [N, 2, 2, 512]  the thrid dim include pre-normed and normed data
            np.save(
                "{}train_trace{}.npy".format(self.dst_path, fold), self.train_traces
            )
            np.save("{}train_ne{}.npy".format(self.dst_path, fold), self.train_ne)
            np.save(
                "{}train_label{}.npy".format(self.dst_path, fold), self.train_labels
            )
            np.save("{}val_trace{}.npy".format(self.dst_path, fold), self.val_traces)
            np.save("{}val_ne{}.npy".format(self.dst_path, fold), self.val_ne)
            np.save("{}val_label{}.npy".format(self.dst_path, fold), self.val_labels)

        else:
            print(">>>>>>>>>Loading Existing Fold{}<<<<<<<<<<<<<<<<<<<<<<".format(fold))
            self.train_traces = torch.from_numpy(
                np.load(
                    "{}train_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.train_ne = torch.from_numpy(
                np.load(
                    "{}train_ne{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.train_labels = torch.from_numpy(
                np.load(
                    "{}train_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_traces = torch.from_numpy(
                np.load(
                    "{}val_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_ne = torch.from_numpy(
                np.load("{}val_ne{}.npy".format(self.dst_path, fold), allow_pickle=True)
            )
            self.val_labels = torch.from_numpy(
                np.load(
                    "{}val_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )

        self.traces, self.ne, self.labels = (
            (self.train_traces, self.train_ne, self.train_labels)
            if not isEval
            else (self.val_traces, self.val_ne, self.val_labels)
        )
        self.traces = self.traces[:, :, :1] if not useNorm else self.traces[:, :, -1:]
        self.ne = self.ne[:, :, :1] if not useNorm else self.ne[:, :, -1:]

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        ne = self.ne[idx]
        label = self.labels[idx]
        return trace, ne, label


class Seq_Loader_NE(Dataset):
    def __init__(
        self,
        root_path="data/raw_data_wNE/",
        data_path="data/dst_data_wNE/seq/",
        isEval=False,
        fold=1,
        n_sequences=16,
        useNorm=False,
    ):
        self.root_path = root_path
        self.dst_path = "{}n_seq_{}/fold_{}/".format(data_path, n_sequences, fold)
        i = 0
        if not os.path.exists(self.dst_path):
            print(
                ">>>>>>>>Starting Processing and Splitting Raw Data Fold{}<<<<<<<<<<<<<<<<<<<".format(
                    fold
                )
            )
            os.makedirs(self.dst_path)
            # os.system("rm -rf {}".format(self.dst_path))
            trace_files = sorted(glob(self.root_path + "*data.npy"))
            ne_files = sorted(glob(self.root_path + "*NE.npy"))
            label_files = sorted(glob(self.root_path + "*label.npy"))

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_idxs = list(kf.split(trace_files))
            train_idxs, val_idxs = fold_idxs[fold - 1]  # Label of fold start from one
            print("Train_idxs: ", train_idxs)
            print("Val_idxs: ", val_idxs)

            train_traces_list = [
                file2tensor_wNE(trace_files[idx], ne_files[idx]) for idx in train_idxs
            ]
            trace_ne_list = [
                file2tensor_wNE(ne_files[idx], ne_files[idx]) for idx in train_idxs
            ]
            train_norm_list = [
                file2tensor_wNE(trace_files[idx], ne_files[idx], norm=True)
                for idx in train_idxs
            ]
            train_norm_ne_list = [
                file2tensor_wNE(ne_files[idx], ne_files[idx], norm=True)
                for idx in train_idxs
            ]
            train_labels_list = [
                file2tensor_wNE(label_files[idx], ne_files[idx], isLabel=True)
                for idx in train_idxs
            ]

            train_traces_list = Seq_filter_func(train_traces_list, train_labels_list)
            trace_ne_list = Seq_filter_func(trace_ne_list, train_labels_list)
            train_norm_list = Seq_filter_func(train_norm_list, train_labels_list)
            train_norm_ne_list = Seq_filter_func(train_norm_ne_list, train_labels_list)
            train_labels_list = Seq_filter_func(train_labels_list, train_labels_list)
            train_data = Seq_slice_func_wNE(
                train_traces_list,
                trace_ne_list,
                train_norm_list,
                train_norm_ne_list,
                train_labels_list,
                n_sequences,
            )
            train_traces = torch.cat([i[0] for i in train_data], dim=0).float()
            train_ne = torch.cat([i[1] for i in train_data], dim=0).float()
            train_norm = torch.cat([i[2] for i in train_data], dim=0).float()
            train_norm_ne = torch.cat([i[3] for i in train_data], dim=0).float()
            self.train_labels = torch.cat([i[4] for i in train_data], dim=0)

            val_traces_list = [
                file2tensor_wNE(trace_files[idx], ne_files[idx]) for idx in val_idxs
            ]
            val_ne_list = [
                file2tensor_wNE(ne_files[idx], ne_files[idx]) for idx in val_idxs
            ]
            val_norm_list = [
                file2tensor_wNE(trace_files[idx], ne_files[idx], norm=True)
                for idx in val_idxs
            ]
            val_norm_ne_list = [
                file2tensor_wNE(ne_files[idx], ne_files[idx], norm=True)
                for idx in val_idxs
            ]
            val_labels_list = [
                file2tensor_wNE(label_files[idx], ne_files[idx], isLabel=True)
                for idx in val_idxs
            ]

            val_traces_list = Seq_filter_func(val_traces_list, val_labels_list)
            val_ne_list = Seq_filter_func(val_ne_list, val_labels_list)
            val_norm_list = Seq_filter_func(val_norm_list, val_labels_list)
            val_norm_ne_list = Seq_filter_func(val_norm_ne_list, val_labels_list)
            val_labels_list = Seq_filter_func(val_labels_list, val_labels_list)
            val_data = Seq_slice_func_wNE(
                val_traces_list,
                val_ne_list,
                val_norm_list,
                val_norm_ne_list,
                val_labels_list,
                n_sequences,
            )
            val_traces = torch.cat([i[0] for i in val_data], dim=0).float()
            val_ne = torch.cat([i[1] for i in val_data], dim=0).float()
            val_norm = torch.cat([i[2] for i in val_data], dim=0).float()
            val_norm_ne = torch.cat([i[3] for i in val_data], dim=0).float()
            self.val_labels = torch.cat([i[4] for i in val_data], dim=0)

            # concat the normalized trace to channel dimmension
            self.train_traces = torch.cat([train_traces, train_norm], dim=3)
            self.val_traces = torch.cat([val_traces, val_norm], dim=3)
            self.train_ne = torch.cat([train_ne, train_norm_ne], dim=3)
            self.val_ne = torch.cat([val_ne, val_norm_ne], dim=3)

            # after concat [N, 2, 2, 512]  the thrid dim include pre-normed and normed data
            # print(self.dst_path)
            np.save(
                "{}train_trace{}.npy".format(self.dst_path, fold), self.train_traces
            )
            np.save("{}train_ne{}.npy".format(self.dst_path, fold), self.train_ne)
            np.save(
                "{}train_label{}.npy".format(self.dst_path, fold), self.train_labels
            )
            np.save("{}val_trace{}.npy".format(self.dst_path, fold), self.val_traces)
            np.save("{}val_ne{}.npy".format(self.dst_path, fold), self.val_ne)
            np.save("{}val_label{}.npy".format(self.dst_path, fold), self.val_labels)

        else:
            print(">>>>>>>>>Loading Existing Fold{}<<<<<<<<<<<<<<<<<<<<<<".format(fold))
            self.train_traces = torch.from_numpy(
                np.load(
                    "{}train_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.train_ne = torch.from_numpy(
                np.load(
                    "{}train_ne{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.train_labels = torch.from_numpy(
                np.load(
                    "{}train_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_traces = torch.from_numpy(
                np.load(
                    "{}val_trace{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )
            self.val_ne = torch.from_numpy(
                np.load("{}val_ne{}.npy".format(self.dst_path, fold), allow_pickle=True)
            )
            self.val_labels = torch.from_numpy(
                np.load(
                    "{}val_label{}.npy".format(self.dst_path, fold), allow_pickle=True
                )
            )

        self.traces, self.ne, self.labels = (
            (self.train_traces, self.train_ne, self.train_labels)
            if not isEval
            else (self.val_traces, self.val_ne, self.val_labels)
        )
        self.traces = (
            self.traces[:, :, :, :1] if not useNorm else self.traces[:, :, :, -1:]
        )
        self.ne = self.ne[:, :, :, :1] if not useNorm else self.ne[:, :, :, -1:]

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        ne = self.ne[idx]
        label = self.labels[idx]
        return trace, ne, label
