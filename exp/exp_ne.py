from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)
from data_provider.data_generator_ne import data_generator, visualize_data_generator

# from exp.exp_basic import Exp_Basic
from models.epoch import (
    CMTransformer,
    ViTTransformer,
    CMATransformer,
    monoViT,
    CNNViTTransformer,
    MacaronTransformer,
    MixTransformer,
    MacrossTransformer,
    MoETransformer,
    FreqTransformer,
    BaseLineNE,
    CMTransformer,
    TFTransformer,
    TFCMTransformer,
    SMoETransformer,
    SimMoE,
    FreqCM,
    MLP_base,
    SWBaseLine,
    LSTM_base,
)
from models.seq import (
    n2nViTTransformer,
    n2nCMATransformer,
    n2nMacrossTransformer,
    n2nMoETransformer,
    n2nBaseLineNE,
    n2nSeqCM,
    n2nCMTransformer,
    n2nSeqMoE,
    n2nMLP,
    n2nLSTM,
)

# from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import ProgressMeter
from utils.metric_tracker import batch_updater, build_tracker
from utils.optimization import load_optimizer, load_scheduler
from utils.tools import EarlyStopping, load_checkpoint
from utils.visualize import visualize_pred, visualize_tsne, visualize_attn
import matplotlib.patches as mpatches

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


class Exp_Main(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.exp_dir = None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                device = torch.device("cuda")
            else:
                device = torch.device("cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
        return device

    def _build_model(self):
        model_dict = {
            "Vanilla": None,
            # 'ViTsh': ViTTransformer
            "ViTsh": ViTTransformer if self.args.data == "Epoch" else n2nViTTransformer,
            "CMA": CMATransformer if self.args.data == "Epoch" else n2nCMATransformer,
            "CM": CMTransformer if self.args.data == "Epoch" else n2nCMTransformer,
            "MonoViT": monoViT,
            "MIX": MixTransformer,
            "MoE": MoETransformer if self.args.data == "Epoch" else n2nSeqMoE,
            "Macaron": MacaronTransformer if self.args.data == "Epoch" else None,
            "Macross": (
                MacrossTransformer
                if self.args.data == "Epoch"
                else n2nMacrossTransformer
            ),
            "CNN-ViTsh": CNNViTTransformer if self.args.data == "Epoch" else None,
            "Multicross": CMTransformer,
            "Freq": FreqTransformer,
            "TF": TFTransformer,
            "BaseLine": BaseLineNE if self.args.data == "Epoch" else n2nBaseLineNE,
            "TFCM": TFCMTransformer,
            "SMoE": SMoETransformer,
            "Dev": BaseLineNE,
            "SeqCM": n2nSeqCM,
            "SeqMoE": n2nSeqMoE,
            "FreqCM": FreqCM,
            "MLP": MLP_base if self.args.data == "Epoch" else n2nMLP,
            "SWBaseLine": SWBaseLine if self.args.data == "Epoch" else None,
            "LSTM": LSTM_base if self.args.data == "Epoch" else n2nLSTM,
        }
        if self.args.features == "ALL":
            model = model_dict[self.args.model].Model(self.args)
        else:
            model = model_dict[self.args.model].Mono_Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model).cuda()
        elif self.args.use_gpu:
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_generator(self.args, flag)
        return data_set, data_loader

    def _get_visualize_data(self):
        data_set, data_loader = visualize_data_generator(self.args)
        return data_set, data_loader

    def _select_criterion(self):
        weight = torch.tensor(self.args.weight) if len(self.args.weight) < 3 else None
        criterion = nn.CrossEntropyLoss(weight=weight).to(self.device)
        return criterion

    def _reload_model(self):
        ckpt = load_checkpoint(
            self.exp_dir,
            if_best=self.args.reload_best,
            device=self.device,
            reload_ckpt=self.args.reload_ckpt,
        )
        model = self._build_model().to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def eval(self, val_loader, model, criterion, args):
        (
            Time,
            Loss,
            Acc,
            F1,
            Kappa,
            Precision,
            Recall,
            w_f1,
            s_f1,
            r_f1,
            w_prec,
            s_prec,
            r_prec,
            w_rec,
            s_rec,
            r_rec,
        ) = build_tracker(isEval=True)
        progress = ProgressMeter(
            len(val_loader), [Time, Loss, Acc, F1, Precision, Recall], prefix="Test: "
        )
        all_pred = []
        all_gt = []
        self.model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (traces, nes, labels) in enumerate(val_loader):
                traces = traces.to(self.device)
                nes = nes.to(self.device)
                labels = labels.to(self.device)

                out_dict = model(traces, nes, labels)
                out = out_dict["out"]
                label = out_dict["label"]

                loss = criterion(out, label.view(-1))
                pred = np.argmax(out.detach().cpu(), axis=1)
                label = label.detach().cpu()
                all_pred.append(pred)
                all_gt.append(label)
                # metric calculation and update
                Loss.update(loss.data.item())
                Acc.update(accuracy_score(label, pred))
                F1.update(f1_score(label, pred, average="macro"))
                Precision.update(precision_score(label, pred, average="macro"))
                Recall.update(recall_score(label, pred, average="macro"))

                Time.update(time.time() - end)

                if i % args.print_freq == 0:
                    progress.display(i + 1)

        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)
        progress = ProgressMeter(
            len(val_loader),
            [Time, Loss, Acc, F1, Kappa, Precision, Recall],
            prefix="Test: ",
        )
        accuracy = accuracy_score(all_gt, all_pred)
        Acc.reset2update(accuracy)
        F1.reset2update(f1_score(all_gt, all_pred, average="macro"))
        Precision.reset2update(precision_score(all_gt, all_pred, average="macro"))
        Recall.reset2update(recall_score(all_gt, all_pred, average="macro"))
        Kappa.update(cohen_kappa_score(all_gt, all_pred))
        progress.display_summary()
        return accuracy

    def train(
        self, train_loader, model, criterion, optimizer, scheduler, epoch, device, args
    ):
        Time, Loss, Acc, F1, Kappa, Precision, Recall = build_tracker(isEval=False)

        progress = ProgressMeter(
            len(train_loader),
            [Time, Loss, Acc, F1, Precision, Recall],
            prefix="Epoch: [{}]".format(epoch),
        )

        model.train()
        end = time.time()
        all_gt, all_pred = [], []
        for i, (traces, nes, labels) in enumerate(train_loader):
            traces = traces.to(device)
            nes = nes.to(device)
            labels = labels.to(device)

            out_dict = model(traces, nes, labels)
            out = out_dict["out"]
            label = out_dict["label"]

            loss = criterion(out, label.view(-1))
            pred = np.argmax(out.detach().cpu(), axis=1)
            label = label.detach().cpu()
            all_pred.append(pred)
            all_gt.append(label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            Loss.update(loss.data.item())
            Acc.update(accuracy_score(label, pred))
            F1.update(f1_score(label, pred, average="macro"))
            Precision.update(precision_score(label, pred, average="macro"))
            Recall.update(recall_score(label, pred, average="macro"))

            Time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        progress = ProgressMeter(
            len(train_loader),
            [Time, Loss, Acc, F1, Kappa, Precision, Recall],
            prefix="Train:",
        )

        Acc.reset2update(accuracy_score(all_gt, all_pred))
        F1.reset2update(f1_score(all_gt, all_pred, average="macro"))
        Precision.reset2update(precision_score(all_gt, all_pred, average="macro"))
        Recall.reset2update(recall_score(all_gt, all_pred, average="macro"))
        Kappa.update(cohen_kappa_score(all_gt, all_pred))
        progress.display_summary()

    def run_train_visualize(self, setting, visualize_loader):
        visual_model = self._reload_model()
        if "pred" in self.args.visualize_mode:
            visualize_pred(
                setting, visual_model, visualize_loader, self.device, self.args
            )
        if "attn" in self.args.visualize_mode:
            visualize_attn(
                setting, visual_model, visualize_loader, self.device, self.args
            )
        if "emb" in self.args.visualize_mode:
            visualize_tsne(
                setting, visual_model, visualize_loader, self.device, self.args
            )

    def run_train(self, setting):
        if self.exp_dir is None:
            self.exp_dir = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(self.exp_dir):
                os.makedirs(self.exp_dir)

        train_data, train_loader = self._get_data(flag="train")
        val_data, val_loader = self._get_data(flag="val")
        visualize_data, visualize_loader = self._get_visualize_data()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = load_optimizer(self.args, self.model)
        criterion = self._select_criterion()

        scheduler = load_scheduler(
            self.args, optimizer=optimizer, train_steps=train_steps
        )

        for epoch in range(self.args.epochs):
            self.train(
                train_loader,
                self.model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                self.device,
                self.args,
            )
            print("\n")

            acc = self.eval(val_loader, self.model, criterion, self.args)
            print("\n")

            early_stopping(
                args=self.args,
                epoch=epoch,
                acc=acc,
                model=self.model,
                optimizer=optimizer,
                exp_dir=self.exp_dir,
            )

            if early_stopping.early_stop:
                print("Early stopping at epoch {} ...".format(epoch))
                break

        self.run_train_visualize(setting, visualize_loader)

    # def run_infer_visualize(self, setting):
    #     visualize_data, visualize_loader = self._get_visualize_data()
    #     visual_model = self._reload_model()
    #     if 'pred' in self.args.visualize_mode:
    #         visualize_pred(setting, visual_model, visualize_loader, self.device, self.args)
    #     if 'attn' in self.args.visualize_mode:
    #         visualize_attn(setting, visual_model, visualize_loader, self.device, self.args)
    #     if 'emb' in self.args.visualize_mode:
    #         visualize_tsne(setting, visual_model, visualize_loader, self.device, self.args)
