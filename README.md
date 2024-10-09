## Prerequisite
Please first make sure that you have set up a environment with all the packages listed in requirements.txt installed. If possible, use the version specified for each corresponding package. For pytorch, install the cuda version that's right for your gpu. If you train the model with cuda and with the default config (recommended), you may need GPU memory of 16G or more. You may run into GPU out of memory issue with smaller memory. 

## Training
Follow the three steps listed below, in order, to train a sDREAMER model on your data.

### 1. Data files
Each data file must be a .mat file which contains the following field: 1. `eeg` (2D array), 2. `emg` (2D array), 3. `eeg_frequency` (2D array), and 4) `sleep_scores` (2D array). `emg` is assumed to be sampled at the same frequency as `eeg`, so that's why only `eeg_frequency` is needed. They are assumed to have the same duration. Both `eeg` and `emg` must have the shape of [1, *t* $\times$ *f*], where *t* is the duration of recording in seconds, and *f* is the sampling frequency. `eeg_frequency` must have the shape of [1, 1]. So it's just one value, but "wrapped" as a 2D array. In addition, `eeg_frequency` must be the exact sampling frequency used by the recording equipment. Do not round it. In the next step, the rounding will be taken care of for you. If you round it now, it will result in misalignment in the sleep score prediction, especially for longer recordings. `sleep_scores` is the groundtruth. It is assumed to have the same duration as `eeg` and `emg`, and it's of shape [1, *t*]. Make sure that you don't have missing values in beginning or in the middle of your `sleep_scores`. Otherwise, the .mat file that contains such `sleep_scores` will be discarded and won't make it to the dataset for training. However, if the missing values are in the end, they will be trimmed automatically when you proceed to the next step. and all `eeg`, `emg`, and `sleep_scores` will be trimmed (on the end only) to the shorter duration in second, of `eeg` or `sleep_scores`.  Please contact me if your data is in any way different from what's assumed here. We should be able to customize training pipeline to make it work for your data.   

#### Sample data
Below is an example of a preprocessed .mat file.
```python
from scipy.io import loadmat

path = "./user_test_files/"
mat_file = path + "preprocessed_240_BL_v2.mat"
mat = loadmat(mat_file)

mat.keys()
Out[1]: dict_keys(['__header__', '__version__', '__globals__', 'emg', 'eeg', 'eeg_frequency'])

mat["eeg"].shape
Out[2]: (1, 6272524)

mat["eeg"]
Out[3]: 
array([[-6.1035155e-07, -2.2354126e-05, -3.1738280e-05, ...,
        -1.6555787e-05, -1.1062622e-05, -6.1798096e-06]], dtype=float32)
     
mat["emg"].shape
Out[4]: (1, 6272524)

mat["emg"]
Out[5]: 
array([[ 2.1209717e-05, -1.5716552e-05,  1.1749267e-05, ...,
        -4.5013426e-06, -3.0517579e-06, -1.8310546e-06]], dtype=float32)
        
mat["eeg_frequency"].shape
Out[6]: (1, 1)

mat["eeg_frequency"]
Out[7]: array([[511.95964011]])

mat["eeg_frequency"].item()
Out[8]: 511.959640106313

mat['sleep_scores']
Out[9]: array([[ 0,  0,  0, ...,  2]])

mat['sleep_scores'].shape
Out[10]: (1, 12252)

```


### 2. Preparing the dataset
To prepare for the training dataset, run *write_training_data.py*. Change the code inside `if __name__ == "__main__":` as needed. Typically, you only need to change 1) `data_path ` to where you placed your preprocessed mat files, 2) `save_path ` to where you want to save the training and validation dataset, and 3) `on_hold_list` to include any files that you need to exclude (for reasons such as incomplete labels or corrupt data). See the relevant code snippet below. Once you specify these three parameters, *write_training_data.py* will turn the preprocessed data to train and validation set that's ready to be fed to the model. In addition, it will also write a list of tain-validation files, called  *train_val_split.txt*, in `save_path`. 
```python
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
```

### 3. Train the sDREAMER model
Run *run_train_sdreamer.py* after you have prepared the dataset in the previous step. Similar to the previous step, change the first few lines of code inside `if __name__ == "__main__":` as needed. See the relevant code snippet below. You only need to specify three parameters, 1) `data_path`, which should point to the path where you **save** the path in the previous step, 2) `checkpoints`, where you want to save the model weights, ie,. the checkpoints, and 3) `des`, which is a suffix in the saved model checkpoint name that helps you identify the models you've trained. Note that the `data_path` here refers to the path at which you saved the training and validation dataset in the previous step, NOT the preprocessed mat files. `des` is helpful when you may train several sDREAMER models at different time (perhaps after you have acquired more data), then you can use the date as the suffix. When no "des" is given, the model will be automatically assigned a name which includes some important hyperparameters of the model, which looks like *SeqNewMoE2_Seq_ftALL_pl16_ns64_dm128_el2_dff512_eb0_scale0.0_bs64_f1*. But you don't need to worry about theses when you train your first model. Just leave everything else untouched and use the default hyperparameters that are already set for you. When the need arises, you can explore different hyperparamters or config.

```python
if __name__ == "__main__":
    # specify the paths
    data_path = "../sdreamer_data/"
    checkpoints = "../sdreamer_checkpoints/"  # model save directory name
    des_name = "test"  # suffix in the model name
```

## Inference
To use a trained model to run inference on a mat file, run *run_inference.py*. See the relevant code snippet below. You can also import the function `infer()` from this file and create your inference script. 
```python
if __name__ == "__main__":
    from scipy.io import loadmat

    checkpoint_path = 'C:/Users/yzhao/python_projects/sleep_scoring/models/sdreamer/checkpoints/SeqNewMoE2_Seq_ftALL_pl16_ns64_dm128_el2_dff512_eb0_scale0.0_bs64_f1_augment_10.pth.tar'
    mat_file = "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/sal_588.mat"
    data = loadmat(mat_file)
    all_pred, all_prob = infer(data, checkpoint_path)
```

## Citing sDREAMER
Please cite [the paper below](https://www.cs.rochester.edu/u/yyao39/files/sDREAMER.pdf) when you use sDREAMER in your paper.
```
@INPROCEEDINGS{10224751,
  author={Chen, Jingyuan and Yao, Yuan and Anderson, Mie and Hauglund, Natalie and Kjaerby, Celia and Untiet, Verena and Nedergaard, Maiken and Luo, Jiebo},
  booktitle={2023 IEEE International Conference on Digital Health (ICDH)}, 
  title={sDREAMER: Self-distilled Mixture-of-Modality-Experts Transformer for Automatic Sleep Staging}, 
  year={2023},
  volume={},
  number={},
  pages={131-142},
  keywords={Training;Sleep;Brain modeling;Transformers;Electromyography;Electroencephalography;Electronic healthcare;sleep scoring;distillation;transformer;mixture-of-modality experts},
  doi={10.1109/ICDH60066.2023.00028}}
```

## Original README below
### Flow
### File Structure
```bash
├── README.md
├── data
│   ├── dst_data --> processed data wo NE (1st version)
│   ├── dst_data_wNE --> processed data with NE(2nd version)
│   ├── raw_data --> raw data (1st version)
│   └── raw_data_wNE --> raw data with NE (2nd version)
├── ckpt 
│   ├── baseline --> ckpts for baseline model under different settings
│   ├── cm
│   ├── ...
│   └── ...
├── ckpt_ne
│   ├── baseline --> ckpts for baseline model under different settings
│   └── sdreamer 
├── ckpt_seq --> used to store ckpts for seq2seq model(no longer used)
├── data_provider
│   ├── __init__.py
│   ├── data_generator.py --> used to create data and data loader object
│   ├── data_generator_ne.py --> used to create data and data loader object for NE
│   └── data_loader.py --> used to load and preprocess data
├── epoch_pics --> manually created folder to store demo timeseries pics(no longer used)
├── exp
│   ├── exp_main.py --> main file to run experiments for non-MoE models
│   ├── exp_moe_ne.py --> main file to run experiments for MoE models with NE(not used yet)
│   ├── exp_moe.py --> main file to run experiments for MoE models(no longer used)
│   ├── exp_moe2.py --> main file to run experiments for MoE models(currently used)
│   └── exp_ne.py --> main file to run experiments for non-MoE models with NE
├── layers --> all sub-modules used in the model
├── logs --> all training logs are stored here
├── models
│   ├── epoch --> all epoch models (NE models included)
│   └── seq --> all seq models (NE models included)
├── scripts
│   ├── epoch --> used to run all non-NE epoch experiments(subfolders specified by each model)
│   └── seq --> used to run all non-NE seq experiments
├── scripts_ne
│   ├── epoch --> used to run all NE epoch experiments(subfolders specified by each model)
│   └── seq --> used to run all NE seq experiments
├── utils --> some utility functions used in the project
├── visualizations --> visualized results are stored here
├── ml_baseline.ipynb --> baseline ML models
├── moe_Eval.py --> used to evaluate MoE models
├── moe_Launch.py --> used to train MoE models(no longer used)
├── moe_Launch2.py --> used to train MoE models(currently used)
├── moe_LaunchNE.py --> used to train MoE models with NE(not used yet)
├── train_Launch.py --> used to train non-MoE models wo NE(1st version)
└── train_LaunchNE.py --> used to train non-MoE models with NE(2nd version)
```

