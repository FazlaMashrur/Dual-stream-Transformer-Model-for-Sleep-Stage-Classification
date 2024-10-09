import argparse
import os
from pytorch_lightning import seed_everything
import torch
from exp.exp_moe2 import Exp_MoE
import random
import numpy as np


def argparser():
    parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")

    # random seed
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # basic config
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        default="SeqHMoE",
        help="model name, options: [Vanilla, ViTsh, CM]",
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")

    # data and store
    parser.add_argument("--data", type=str, default="Seq", help="dataset type")
    parser.add_argument("--fold", type=int, default=1, help="fold")
    parser.add_argument(
        "--root_path",
        type=str,
        default="data/raw_data/",
        help="root path of the raw file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/dst_data/epoch/",
        help="path for process data file",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="ALL",
        help="Input Features, options:[EEG, EMG, ALL]",
    )
    parser.add_argument(
        "--n_sequences", type=int, default=16, help="number of input sequences"
    )
    parser.add_argument(
        "--useNorm", action="store_false", help="pre normalize data", default=True
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )

    # PatchTST
    parser.add_argument("--seq_len", type=int, default=512, help="patch length")
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--padding_patch", default="end", help="None: None; end: padding on the end"
    )
    parser.add_argument(
        "--subtract_last",
        type=int,
        default=0,
        help="0: subtract mean; 1: subtract last",
    )
    parser.add_argument(
        "--decomposition", type=int, default=0, help="decomposition; True 1 False 0"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=25, help="decomposition-kernel"
    )
    parser.add_argument(
        "--individual", type=int, default=0, help="individual head; True 1 False 0"
    )

    # Formers
    parser.add_argument(
        "--mix_type",
        type=int,
        default=0,
        help="0: patch emd + pos emd, 1: patch emb, 2:patch emb + pos emb + trace emb",
    )
    parser.add_argument("--c_out", type=int, default=3, help="output size")
    parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument(
        "--ca_layers", type=int, default=1, help="num of cross attention layers"
    )
    parser.add_argument(
        "--seq_layers", type=int, default=3, help="num of encoder layers"
    )
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of fcn")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--path_drop", type=float, default=0.0, help="path drop")
    parser.add_argument(
        "--pos_emb",
        type=str,
        default="learned",
        help="positional encoding, options:[learned, sinusoidal]",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="glu",
        help="activation functions, options:[glu, relu, relu_squared]",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="layernorm",
        help="normalization functions, options:[layernorm, batchnorm]",
    )
    parser.add_argument(
        "--output_attentions",
        action="store_true",
        help="whether to output attention in encoder",
        default=False,
    )
    parser.add_argument(
        "--useRaw", action="store_false", help="use raw cls token", default=False
    )

    # optimization
    parser.add_argument("--epochs", type=int, default=1, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="early stopping patience"
    )
    parser.add_argument("--optimizer", default="adamw", type=str, help="validation set")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument(
        "--weight_decay", default=0.0001, type=float, help="optimizer weight decay"
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.9, help="beta 1 for adam optimizer"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999, help="beta 2 for adam optimizer"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-9, help="eps for adam optimizer"
    )
    parser.add_argument(
        "--scheduler", default="CosineLR", type=str, help="sheduler for training"
    )
    parser.add_argument(
        "--scale", type=float, default=0.0, help="Gamma for LR scheduler"
    )
    # cycle LR
    parser.add_argument(
        "--pct_start", type=float, default=0.3, help="pct_start for cycle sheduler"
    )
    # srtepLR
    parser.add_argument(
        "--step_size", type=float, default=30, help="Step size for LR scheduler"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Gamma for LR scheduler"
    )

    # For weighted loss
    parser.add_argument(
        "--weight",
        type=list,
        default=[1.0, 1.0, 1.0],
        help="Weights for cross entropy loss",
    )

    # Visualizations
    parser.add_argument(
        "--visualize_mode", nargs="*", default=["pred"], help="visualize mode"
    )
    parser.add_argument(
        "--visualizations",
        type=str,
        default="./visualizations/",
        help="location of model checkpoints",
    )

    # model save and load
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--reload_best",
        action="store_true",
        help="reload the best acc model",
        default=True,
    )
    parser.add_argument(
        "--reload_ckpt",
        type=str,
        default="/home/jingyuan_chen/sd/flow/ckpt/hmoe/SeqHMoE_Seq_ftALL_pl16_ns16_dm128_el2_dff512_eb0_bs16_f1_kl_2.0_t3.0/model_best.pth.tar",
        help="full location of model checkpoints",
    )
    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--test_flop",
        action="store_true",
        default=False,
        help="See utils/tools for usage",
    )

    parser.add_argument(
        "--print_freq",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    args = parser.parse_args()
    return args


# random seed
def main():
    args = argparser()

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    seed_everything(args.seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print("Args in experiment:")
    print(args)

    Exp = Exp_MoE

    if args.is_training:
        setting = "{}_{}_ft{}_pl{}_ns{}_dm{}_el{}_dff{}_eb{}_bs{}_f{}_{}".format(
            args.model,
            args.data,
            args.features,
            args.patch_len,
            args.n_sequences,
            args.d_model,
            args.e_layers,
            args.d_ff,
            args.mix_type,
            args.batch_size,
            args.fold,
            args.des,
        )

        exp = Exp(args)  # set experiments
        print(">>>>>>>>Start Training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.run_eval_visualize(setting)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
