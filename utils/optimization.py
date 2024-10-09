import os
import torch
import numpy as np

# from models import LARS
from collections import deque
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def load_optimizer(args, model):
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )

    elif args.optimizer == "adamw":
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),
        #                          eps=args.eps, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    return optimizer


def load_scheduler(args, optimizer, train_steps=0):
    if args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    elif args.scheduler == "CosineLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )

    elif args.scheduler == "OneCycleLR":
        scheduler = torch.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.epochs,
            max_lr=args.lr,
        )
    else:
        raise NotImplementedError
    return scheduler
