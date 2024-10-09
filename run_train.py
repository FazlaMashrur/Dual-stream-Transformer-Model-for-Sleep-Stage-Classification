# -*- coding: utf-8 -*-
"""
Created on Tue May 14 23:28:55 2024

@author: yzhao
"""


import os
import random
import logging
import argparse

import torch
import numpy as np
from pytorch_lightning import seed_everything

from exp.exp_moe2 import Exp_MoE


# hyperparameters
activation = "glu"
norm_type = "layernorm"
patch_len = 16
seed = 42
ca_layers = 1
batch_size = 64
n_sequences = 64
ne_patch_len = 32
e_layers = 2
fold = 1

# SeqNewMoE2_Seq_pl16_el2_cl1_f1_seql3_kl_2.0_t3.5

config = dict(
    seed=42,
    is_training=1,
    model_id="test",
    model="SeqNewMoE2",
    data="Seq",
    isNE=False,
    fold=1,
    root_path="",
    # data_path=data_path,
    features="ALL",
    n_sequences=n_sequences,
    useNorm=True,
    num_workers=10,
    seq_len=512,
    patch_len=patch_len,
    stride=8,
    padding_patch="end",
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    mix_type=0,
    c_out=3,
    d_model=128,
    n_heads=8,
    e_layers=e_layers,
    ca_layers=ca_layers,
    seq_layers=3,
    d_ff=512,
    dropout=0.1,
    path_drop=0.0,
    pos_emb="learned",
    activation=activation,
    norm_type=norm_type,
    output_attentions=False,
    useRaw=False,
    epochs=100,
    batch_size=batch_size,
    patience=30,
    optimizer="adamw",
    lr=0.001,
    weight_decay=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    eps=1e-9,
    scheduler="CosineLR",
    scale=0.0,
    pct_start=0.3,
    step_size=30,
    gamma=0.5,
    weight=[1, 1, 1],
    visualize_mode=[],
    visualizations="",
    # checkpoints=checkpoints,
    reload_best=True,
    reload_ckpt=None,
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    test_flop=False,
    print_freq=50,
    # output_path=output_path,
    # ne_patch_len=ne_patch_len,
    # des=des_name,
    # pad=False,
)

# %%
if __name__ == "__main__":
    # specify the paths
    data_path = "../sdreamer_data/"
    checkpoints = "../sdreamer_checkpoints/"  # model save directory name
    des_name = "test"  # suffix in the model name

    parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")
    args = parser.parse_args()
    parser_dict = vars(args)
    for k, v in config.items():
        parser_dict[k] = v

    args.data_path = data_path
    args.checkpoints = checkpoints
    args.des_name = des_name
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    seed_everything(args.seed)

    print("Args in experiment:")
    print(args)

    Exp = Exp_MoE

    if args.is_training:
        setting = (
            "{}_{}_ft{}_pl{}_ns{}_dm{}_el{}_dff{}_eb{}_scale{}_bs{}_f{}_{}".format(
                args.model,
                args.data,
                args.features,
                args.patch_len,
                args.n_sequences,
                args.d_model,
                args.e_layers,
                args.d_ff,
                args.mix_type,
                args.scale,
                args.batch_size,
                args.fold,
                args.des_name,
            )
        )

        logger = logging.getLogger("logger")
        logging.getLogger().setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(
            filename=os.path.join(checkpoints, f"{setting}.log"), mode="w"
        )
        logger.addHandler(file_handler)

        logging.getLogger("logger").info("Args in experiment:\n")
        logging.getLogger("logger").info(args)

        exp = Exp(args)  # set experiments
        print(">>>>>>>>Start Training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        logging.getLogger("logger").info(
            f">>>>>>>>Start Training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        exp.run_train(setting)

        torch.cuda.empty_cache()

        handlers = logger.handlers
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
