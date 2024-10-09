from numpy import genfromtxt
import numpy as np
import os

if not os.path.exists("/Users/yaoyuan/Documents/neurofluids/output_processed"):
    os.mkdir("/Users/yaoyuan/Documents/neurofluids/output_processed")

with open("/Users/yaoyuan/Documents/neurofluids/output/data_list.txt", "r") as f:
    names = f.readlines()
names = [name.strip() for name in names]

for name in names:
    print(name)
    data = genfromtxt(
        f"/Users/yaoyuan/Documents/neurofluids/output/{name}_data.csv", delimiter=","
    )
    print(data.shape)
    print(data[2][-1])
    assert data[2][0] == 0.0
    label = genfromtxt(
        f"/Users/yaoyuan/Documents/neurofluids/output/{name}_score.csv", delimiter=","
    )
    print(label.shape)

    len_data = data.shape[1]
    total_seconds = label.shape[1]
    assert data[2][-1] < total_seconds + 100
    assert data[2][-1] > total_seconds - 100

    all_epoch_data = []
    all_epoch_label = []

    pointer = 0
    for i in range(total_seconds):
        while data[2][pointer] < i:
            pointer += 1
            if pointer + 512 > len_data:
                break
        if pointer + 512 > len_data:
            break
        # print(i, pointer, data[2][pointer], data[2][pointer-1])
        if (
            data[2][pointer] - i > i - data[2][pointer - 1]
        ):  # if previous time stamp is closer
            pointer -= 1
        pointer = max(pointer, 0)

        epoch_data = data[
            :2, None, pointer : pointer + 512
        ]  # (2, 1, 512): 2 traces(EMG and EEG); 1 channel; 512 sequence length
        if np.max(label[:, i]) == 0:
            epoch_label = -1  # -1 for unknown
        else:
            epoch_label = np.argmax(label[:, i])  # 0 for wake, 1 for sws, 2 for REM
        all_epoch_data.append(epoch_data)
        all_epoch_label.append(epoch_label)

    all_epoch_data = np.array(all_epoch_data)
    all_epoch_label = np.array(all_epoch_label)

    print(all_epoch_data.shape)
    print(all_epoch_label.shape)

    np.save(
        f"/Users/yaoyuan/Documents/neurofluids/output_processed/{name}_data.npy",
        all_epoch_data,
    )  # (N, 2, 1, 512)
    np.save(
        f"/Users/yaoyuan/Documents/neurofluids/output_processed/{name}_label.npy",
        all_epoch_label,
    )  # (N, 1)
