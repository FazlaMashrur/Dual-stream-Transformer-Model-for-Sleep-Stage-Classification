import os
import shutil
import logging

import torch


def save_checkpoint(state, is_best, exp_dir, filename="ckpt.pth.tar"):
    ckpt_name = os.path.join(exp_dir, filename)
    torch.save(state, ckpt_name)
    if is_best:
        best_name = os.path.join(exp_dir, "model_best.pth.tar")
        shutil.copyfile(ckpt_name, best_name)
        print("=> saving new best Acc model =========>")
        logging.getLogger("logger").info("=> saving new best Acc model =========>")


def load_checkpoint(
    exp_dir, if_best, device, filename=None, defualt="ckpt.pth.tar", reload_ckpt=None
):
    if filename is None:
        filename = "model_best.pth.tar" if if_best else defualt
    if reload_ckpt is None:
        reload_ckpt = os.path.join(exp_dir, filename)
    if os.path.isfile(reload_ckpt):
        print("=> loading checkpoint '{}'".format(reload_ckpt))
        logging.getLogger("logger").info(f"=> loading checkpoint '{reload_ckpt}'")
        checkpoint = torch.load(reload_ckpt, map_location=device)
        epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        print(
            "=> loaded checkpoint '{}' (epoch {} bestAcc {}))".format(
                reload_ckpt, checkpoint["epoch"], checkpoint["best_acc"]
            )
        )
        logging.getLogger("logger").info(
            f"=> loaded checkpoint '{reload_ckpt}' (epoch {epoch} bestAcc {best_acc}))"
        )
        return checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(reload_ckpt))
        return None


class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False
        self.delta = delta

    def __call__(self, args, epoch, acc, model, optimizer, exp_dir):
        is_best = acc > self.best_acc + self.delta

        if not is_best:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            logging.getLogger("logger").info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = acc
            self.counter = 0
        self.best_acc = max(acc, self.best_acc)
        save_checkpoint(
            {
                "epoch": epoch,
                "model": args.model,
                "state_dict": (
                    model.state_dict()
                    if not isinstance(model, torch.nn.DataParallel)
                    else model.module.state_dict()
                ),
                "best_acc": self.best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            exp_dir,
            filename="ckpt.pth.tar",
        )
