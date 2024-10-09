import os
import time
import cv2
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.patheffects as PathEffects
from einops import rearrange

sns.set(style="white", font="serif", context="paper")


def visualize_pred(setting, model, val_loader, device, args):
    print("Visualizing results...")

    fig_dir = os.path.join(args.visualizations, setting, "figure")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    class_mapping = {-1: "Unknown", 0: "Awake", 1: "SWS", 2: "REM"}

    cmap = sns.xkcd_palette(["bright red", "emerald", "dodger blue"])
    color_mapping = {
        -1: "k",
        0: cmap[0],
        1: cmap[1],
        2: cmap[2],
    }

    patches = [
        mpatches.Patch(
            color=color, alpha=0.25, label="{:s}".format(class_mapping[cls_idx])
        )
        for cls_idx, color in color_mapping.items()
    ]
    model.eval()
    with torch.no_grad():
        for i, (traces_no_norm, traces, labels, stamp) in enumerate(val_loader):
            # TODO: split this data loader!
            traces2plot = traces

            traces = traces.to(device)[0]
            labels = labels.to(device)[0]
            stamp = stamp[0]

            out_dict = model(traces, labels)
            out = out_dict["out"]
            label = out_dict["label"]
            preds = np.argmax(out.detach().cpu(), axis=1)
            label = label.detach().cpu()

            val_idx = stamp[0]
            timestamp = stamp[1]
            x = np.linspace(
                timestamp, timestamp + traces.size(0), traces.size(0) * traces.size(-1)
            )
            EEG = traces2plot[0, :, 0, 0, :].cpu().numpy().reshape(-1)
            EMG = traces2plot[0, :, 1, 0, :].cpu().numpy().reshape(-1)

            EEG_raw = traces_no_norm[0, :, 0, 0, :].cpu().numpy().reshape(-1)
            EMG_raw = traces_no_norm[0, :, 1, 0, :].cpu().numpy().reshape(-1)

            # plt.figure(figsize=(20, 15))
            fig = plt.figure(figsize=(20, 15), dpi=200)

            ax1 = plt.subplot(411)
            ax1.set_title("EEG Signal")
            ax1.plot(x, EEG_raw, label="EEG Label", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax1.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[labels[j].item()],
                    alpha=0.25,
                )
            # ax1.legend(loc='upper right')

            ax2 = plt.subplot(412, sharex=ax1)
            ax2.plot(x, EEG, label="EEG Prediction", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax2.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[preds[j].item()],
                    alpha=0.25,
                )
            # ax2.legend(loc='upper right')

            ax3 = plt.subplot(413, sharex=ax1)
            ax3.set_title("EMG Signal")
            ax3.plot(x, EMG_raw, label="EMG Label", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax3.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[labels[j].item()],
                    alpha=0.25,
                )
            # ax3.legend(loc='upper right')

            ax4 = plt.subplot(414, sharex=ax1)
            ax4.plot(x, EMG, label="EMG Prediction", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax4.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[preds[j].item()],
                    alpha=0.25,
                )
            # ax4.legend(loc='upper right')

            fig.suptitle(
                "idx: {}, time: {}".format(val_idx.item(), timestamp.item()),
                size=20,
                y=0.95,
            )
            fig.legend(
                handles=patches, loc="upper right", labelspacing=0.1, fontsize=12
            )
            fig.savefig(
                "{}/idx: {}, time: {}".format(
                    fig_dir, val_idx.item(), timestamp.item()
                ),
                dpi=400,
            )
            plt.close(fig)


# write a function to visualize embedding using tsne plot
def visualize_tsne(setting, model, val_loader, device, args):
    print("Visualizing tsne...")
    tsne_dir = os.path.join(args.visualizations, setting, "tsne")
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir)
    model.eval()
    emb_list, label_list = [], []
    with torch.no_grad():
        for i, (_, traces, labels, _) in enumerate(val_loader):
            traces = traces.to(device)[0]
            labels = labels.to(device)[0]
            out_dict = model(traces, labels)
            emb = out_dict["emb"]
            label = out_dict["label"]
            emb_list.append(emb.detach().cpu())
            label_list.append(label.detach().cpu().reshape(-1))

        embs = torch.cat(emb_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        labels = labels.numpy()

        class_mapping = {-1: "Unknown", 0: "Awake", 1: "SWS", 2: "REM"}
        cmap = np.array(sns.xkcd_palette(["black", "salmon", "amber", "dodger blue"]))

        patches = [
            mpatches.Patch(color=cmap[cls_idx + 1], label="{:s}".format(cls))
            for cls_idx, cls in class_mapping.items()
        ]
        tsne = TSNE(n_components=2, n_jobs=-1, verbose=1)
        embs_2d = tsne.fit_transform(embs)
        plt.figure(figsize=(18, 16))
        plt.scatter(
            embs_2d[:, 0], embs_2d[:, 1], lw=0, s=40, c=cmap[labels.astype(np.int) + 1]
        )
        # plt.xlim(-25, 25)
        # plt.ylim(-25, 25)
        # plt.axis('off')
        plt.axis("tight")
        plt.title("t-SNE embedding of the EEG-EMG traces", fontsize=24)
        plt.legend(handles=patches, loc="upper right", labelspacing=0.1, fontsize=12)
        plt.savefig("{}/tsne.png".format(tsne_dir), dpi=400)
        plt.close()


# write a function to calculate the cls attention
def get_cls_attn(attn):
    # attn_mat -> (layer, batch, num_heads, tokens, token)
    attn_mat = torch.stack(attn)
    attn_mat = torch.mean(attn_mat, dim=2)
    res_attn = torch.eye(attn_mat.size(-1)).to(attn_mat.device)
    aug_attn = attn_mat + res_attn
    aug_attn = aug_attn / aug_attn.sum(dim=-1).unsqueeze(-1)
    # joint_attn -> (layer, batch, tokens, tokens)
    joint_attn = torch.zeros(aug_attn.size()).to(aug_attn.device)
    joint_attn[0] = aug_attn[0]
    for i in range(1, aug_attn.size(0)):
        joint_attn[i] = torch.matmul(aug_attn[i], joint_attn[i - 1])
    # lastlayer_attn -> (batch, tokens, tokens)
    lastlayer_attn = joint_attn[-1]
    # cls_attn -> (batch, tokens, 1)
    cls_attn = lastlayer_attn[:, -1, :-1].unsqueeze(-1).detach().cpu().numpy()
    cls_attn = cls_attn / cls_attn.max(axis=1, keepdims=True)
    return cls_attn


# write a fucntion to visualize the attention map by recieving the attention map from the model
def visualize_attn(setting, model, val_loader, device, args):
    print("Visualizing attention...")
    attn_dir = os.path.join(args.visualizations, setting, "attention")
    if not os.path.exists(attn_dir):
        os.makedirs(attn_dir)
    model.eval()
    with torch.no_grad():
        for i, (_, traces, labels, stamp) in enumerate(val_loader):
            traces2plot = traces
            EEG = traces2plot[0, :, 0, 0, :].cpu().numpy()
            EMG = traces2plot[0, :, 1, 0, :].cpu().numpy()

            traces = traces.to(device)[0]
            labels = labels.to(device)[0]
            stamp = stamp[0]
            val_idx = stamp[0]
            timestamp = stamp[1]
            out_dict = model(traces, labels)
            sub_dir = os.path.join(
                attn_dir,
                "idx{}".format(val_idx.item()),
                "time{}".format(timestamp.item()),
            )
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            if "eeg_attn" in out_dict.keys() and "emg_attn" in out_dict.keys():
                eeg_attn = out_dict["eeg_attn"]
                emg_attn = out_dict["emg_attn"]
                eeg_cls_att = get_cls_attn(eeg_attn)
                emg_cls_att = get_cls_attn(emg_attn)
                for j in range(eeg_cls_att.shape[0]):
                    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))
                    ax1.set_title("EEG Attention Map Sample {}".format(j))
                    ax2.set_title("EMG Attention Map Sample {}".format(j))
                    rough_eeg_att = eeg_cls_att[j].transpose(1, 0)
                    rough_emg_att = emg_cls_att[j].transpose(1, 0)
                    sns.heatmap(rough_eeg_att, ax=ax1)
                    sns.heatmap(rough_emg_att, ax=ax2)
                    fig.suptitle(
                        "Attention Map of EEG and EMG for idx: {}, time: {}-{}".format(
                            val_idx.item(), timestamp.item(), j
                        ),
                        size=20,
                        y=0.95,
                    )
                    fig.savefig(
                        "{}/sample{}_heatmap_eeg_emg.png".format(sub_dir, j), dpi=400
                    )
                    plt.close(fig)

                    # resize the attention map to the size of the traces
                    fine_eeg_att = cv2.resize(rough_eeg_att, (traces.size(-1), 1))
                    fine_emg_att = cv2.resize(rough_emg_att, (traces.size(-1), 1))
                    fine_eeg_att = (fine_eeg_att - fine_eeg_att.min()) / (
                        fine_eeg_att.max() - fine_eeg_att.min()
                    )
                    fine_emg_att = (fine_emg_att - fine_emg_att.min()) / (
                        fine_emg_att.max() - fine_emg_att.min()
                    )
                    EEG_sig = (EEG[j] - EEG[j].min()) / (EEG[j].max() - EEG[j].min())
                    EMG_sig = (EMG[j] - EMG[j].min()) / (EMG[j].max() - EMG[j].min())
                    # plot finegrained attention map on the traces
                    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))
                    ax1.set_title("EEG Attention Map Sample {}".format(j))
                    ax2.set_title("EMG Attention Map Sample {}".format(j))
                    ax1.plot(EEG_sig, color="grey")
                    ax2.plot(EMG_sig, color="grey")
                    ax1.pcolormesh(fine_eeg_att)
                    ax2.pcolormesh(fine_emg_att)

                    fig.suptitle(
                        "Attention Map of EEG and EMG for idx: {}, time: {}-{}".format(
                            val_idx.item(), timestamp.item(), j
                        ),
                        size=20,
                        y=0.95,
                    )
                    fig.savefig(
                        "{}/sample{}_fine_eeg_emg.png".format(sub_dir, j), dpi=400
                    )
                    plt.close(fig)

            if "cm_eeg_attn" in out_dict.keys() and "cm_emg_attn" in out_dict.keys():
                cm_eeg_attn = out_dict["cm_eeg_attn"]
                cm_emg_attn = out_dict["cm_emg_attn"]
                cm_eeg_cls_att = get_cls_attn(cm_eeg_attn)
                cm_emg_cls_att = get_cls_attn(cm_emg_attn)
                for j in range(cm_eeg_cls_att.shape[0]):
                    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))
                    ax1.set_title("EEG-EMG Cross Attention Map Sample {}".format(j))
                    ax2.set_title("EMG-EEG Cross Attention Map Sample {}".format(j))
                    rough_eeg_att = cm_eeg_cls_att[j].transpose(1, 0)
                    rough_emg_att = cm_emg_cls_att[j].transpose(1, 0)
                    sns.heatmap(rough_eeg_att, ax=ax1)
                    sns.heatmap(rough_emg_att, ax=ax2)
                    fig.suptitle(
                        "Cross Attention Map of EEG and EMG for idx: {}, time: {}-{}".format(
                            val_idx.item(), timestamp.item(), j
                        ),
                        size=20,
                        y=0.95,
                    )
                    fig.savefig(
                        "{}/sample{}_heatmap_eeg_emg_cm.png".format(sub_dir, j), dpi=400
                    )
                    plt.close(fig)

                    # resize the attention map to the size of the traces
                    fine_eeg_att = cv2.resize(rough_eeg_att, (traces.size(-1), 1))
                    fine_emg_att = cv2.resize(rough_emg_att, (traces.size(-1), 1))
                    fine_eeg_att = (fine_eeg_att - fine_eeg_att.min()) / (
                        fine_eeg_att.max() - fine_eeg_att.min()
                    )
                    fine_emg_att = (fine_emg_att - fine_emg_att.min()) / (
                        fine_emg_att.max() - fine_emg_att.min()
                    )
                    EEG_sig = (EEG[j] - EEG[j].min()) / (EEG[j].max() - EEG[j].min())
                    EMG_sig = (EMG[j] - EMG[j].min()) / (EMG[j].max() - EMG[j].min())
                    # plot finegrained attention map on the traces
                    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))
                    ax1.set_title("EEG-EMG Cross Attention Map Sample {}".format(j))
                    ax2.set_title("EMG-EEG Cross Attention Map Sample {}".format(j))
                    ax1.plot(EEG_sig, color="grey")
                    ax2.plot(EMG_sig, color="grey")
                    ax1.pcolormesh(fine_eeg_att)
                    ax2.pcolormesh(fine_emg_att)

                    fig.suptitle(
                        "Cross Attention Map of EEG and EMG for idx: {}, time: {}-{}".format(
                            val_idx.item(), timestamp.item(), j
                        ),
                        size=20,
                        y=0.95,
                    )
                    fig.savefig(
                        "{}/sample{}_fine_eeg_emg_cm.png".format(sub_dir, j), dpi=400
                    )
                    plt.close(fig)


def visualize_pred_seq(setting, model, val_loader, device, args):
    print("Visualizing results...")

    fig_dir = os.path.join(args.visualizations, setting, "plain")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    class_mapping = {-1: "Unknown", 0: "Awake", 1: "SWS", 2: "REM"}

    # cmap = sns.xkcd_palette(["bright red", "emerald", "dodger blue"])
    cmap = np.array(sns.xkcd_palette(["salmon", "amber", "dodger blue"]))
    color_mapping = {
        -1: "k",
        0: cmap[0],
        1: cmap[1],
        2: cmap[2],
    }
    color_mapping2 = {
        0: cmap[0],
        1: cmap[1],
        2: cmap[2],
    }

    patches = [
        mpatches.Patch(
            color=color, alpha=1.0, label="{:s}".format(class_mapping[cls_idx])
        )
        for cls_idx, color in color_mapping2.items()
    ]
    model.eval()
    with torch.no_grad():
        for i, (traces_no_norm, traces, labels, stamp) in enumerate(val_loader):
            # TODO: split this data loader!
            traces = traces[0, :320, :, :, :]
            traces_no_norm = traces_no_norm[0, :320, :, :, :]
            labels = labels[0, :320]
            traces2plot = traces
            traces_in = rearrange(traces, "(b e)... -> b e ...", b=20, e=16)
            labels_in = rearrange(labels, "(b e)... -> b e ...", b=20, e=16)

            traces_in = traces_in.to(device)
            labels_in = labels_in.to(device)
            stamp = stamp[0]

            out_dict = model(traces_in, labels_in)
            out = out_dict["out"]
            label = out_dict["label"]
            preds = np.argmax(out.detach().cpu(), axis=1)
            label = label.detach().cpu()

            val_idx = stamp[0]
            timestamp = stamp[1]
            x = np.linspace(
                timestamp, timestamp + traces.size(0), traces.size(0) * traces.size(-1)
            )
            EEG = traces2plot[:, 0, 0, :].cpu().numpy().reshape(-1)
            EMG = traces2plot[:, 1, 0, :].cpu().numpy().reshape(-1)

            EEG_raw = traces_no_norm[:, 0, 0, :].cpu().numpy().reshape(-1)
            EMG_raw = traces_no_norm[:, 1, 0, :].cpu().numpy().reshape(-1)

            # plt.figure(figsize=(20, 15))
            fig = plt.figure(figsize=(20, 15), dpi=200)
            alpha = 1.0
            ax1 = plt.subplot(411)
            # ax1.set_title('EMG Signal', fontsize=20)
            ax1.plot(x, EEG_raw, label="EMG Label", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax1.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[labels[j].item()],
                    alpha=alpha,
                )
            # ax1.legend(loc='upper right', fontsize=22)

            ax2 = plt.subplot(412, sharex=ax1)
            ax2.plot(x, EEG, label="EMG Prediction", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax2.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[preds[j].item()],
                    alpha=alpha,
                )
            # ax2.legend(loc='upper right', fontsize=22)

            ax3 = plt.subplot(413, sharex=ax1)
            # ax3.set_title('EEG Signal', fontsize=20)
            ax3.plot(x, EMG_raw, label="EEG Label", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax3.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[labels[j].item()],
                    alpha=alpha,
                )
            # ax3.legend(loc='upper right', fontsize=22)

            ax4 = plt.subplot(414, sharex=ax1)
            ax4.plot(x, EMG, label="EEG Prediction", linewidth=0.25, color="k")
            for j in range(traces.size(0)):
                ax4.axvspan(
                    x[j * traces.size(-1)],
                    x[(j + 1) * traces.size(-1) - 1],
                    facecolor=color_mapping[preds[j].item()],
                    alpha=alpha,
                )
            # ax4.legend(loc='upper right',  fontsize=22)

            # fig.suptitle('idx: {}, time: {}'.format(val_idx.item(), timestamp.item()), size=20, y=0.95)
            fig.legend(
                handles=patches, loc="upper right", labelspacing=0.1, fontsize=28
            )
            fig.savefig(
                "{}/idx: {}, time: {}".format(
                    fig_dir, val_idx.item(), timestamp.item()
                ),
                dpi=400,
            )
            plt.close(fig)


# write a function to visualize embedding using tsne plot
def visualize_tsne_seq(setting, model, val_loader, device, args):
    print("Visualizing tsne...")
    tsne_dir = os.path.join(args.visualizations, setting, "tsne")
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir)
    model.eval()
    emb_list, label_list = [], []
    eeg_emb_list, emg_emb_list = [], []
    with torch.no_grad():
        for i, (_, traces, labels, _) in enumerate(val_loader):
            traces = traces[0, :320, :, :, :]
            labels = labels[0, :320]
            traces_in = rearrange(traces, "(b e)... -> b e ...", b=20, e=16)
            labels_in = rearrange(labels, "(b e)... -> b e ...", b=20, e=16)

            traces_in = traces_in.to(device)
            labels_in = labels_in.to(device)

            out_dict = model(traces_in, labels_in)
            emb = out_dict["cls_feats"]
            emg_emb = out_dict["cls_feats_eeg"]
            eeg_emb = out_dict["cls_feats_emg"]

            emb = rearrange(emb, "b e ... -> (b e) ...", b=20, e=16)
            eeg_emb = rearrange(eeg_emb, "b e ... -> (b e) ...", b=20, e=16)
            emg_emb = rearrange(emg_emb, "b e ... -> (b e) ...", b=20, e=16)
            label = out_dict["label"]
            emb_list.append(emb.detach().cpu())
            eeg_emb_list.append(eeg_emb.detach().cpu())
            emg_emb_list.append(emg_emb.detach().cpu())
            label_list.append(label.detach().cpu().reshape(-1))

        embs = torch.cat(emb_list, dim=0)[:]
        eeg_embs = torch.cat(eeg_emb_list, dim=0)[:]
        emg_embs = torch.cat(emg_emb_list, dim=0)[:]
        labels = torch.cat(label_list, dim=0)[:]
        embs, eeg_embs, emg_embs, labels = filter_func(
            [embs, eeg_embs, emg_embs, labels], labels
        )
        labels = labels.numpy()
        class_mapping = {0: "Awake", 1: "SWS", 2: "REM"}
        cmap = np.array(sns.xkcd_palette(["salmon", "amber", "dodger blue"]))

        patches = [
            mpatches.Patch(color=cmap[cls_idx], label="{:s}".format(cls))
            for cls_idx, cls in class_mapping.items()
        ]
        modality = ["EEG-EMG", "EEG", "EMG"]
        for i, embs in enumerate([embs, eeg_embs, emg_embs]):
            tsne = TSNE(n_components=2, n_jobs=-1, verbose=1)
            embs_2d = tsne.fit_transform(embs)
            plt.figure(figsize=(27, 24))
            plt.scatter(
                embs_2d[:, 0],
                embs_2d[:, 1],
                lw=0,
                s=40,
                c=cmap[labels.astype(np.int64)],
            )
            # plt.xlim(-25, 25)
            # plt.ylim(-25, 25)
            plt.axis("off")
            plt.axis("tight")

            # plt.title('t-SNE projection of the learned {} embeddings'.format(modality[i]), fontsize=24)
            # plt.legend(handles=patches, loc='upper right', labelspacing=0.1, fontsize=12)
            plt.savefig("{}/tsne-{}.png".format(tsne_dir, modality[i]), dpi=400)
            plt.close()


def filter_func(data_list, label):
    return list(map(lambda tensor: tensor[torch.where(label[:] >= 0)], data_list))


# def visualize_tsne_seq(setting, model, val_loader, device, args):
#     print("Visualizing tsne...")
#     tsne_dir = os.path.join(args.visualizations, setting, 'tsne')
#     if not os.path.exists(tsne_dir):
#         os.makedirs(tsne_dir)
#     model.eval()
#     emb_list, label_list = [], []
#     eeg_emb_list, emg_emb_list = [], []
#     with torch.no_grad():
#         for i, (_, traces, labels, _) in enumerate(val_loader):
#             traces = traces[0,:320,:,:,:]
#             labels = labels[0,:320]
#             traces_in = rearrange(traces, '(b e)... -> b e ...',b=20, e=16)
#             labels_in = rearrange(labels, '(b e)... -> b e ...',b=20, e=16)


#             traces_in = traces_in.to(device)
#             labels_in = labels_in.to(device)

#             out_dict = model(traces_in, labels_in)
#             emb = out_dict['cls_feats']
#             emg_emb = out_dict['cls_feats_eeg']
#             eeg_emb = out_dict['cls_feats_emg']

#             emb = rearrange(emb, 'b e ... -> (b e) ...',b=20, e=16)
#             eeg_emb = rearrange(eeg_emb, 'b e ... -> (b e) ...',b=20, e=16)
#             emg_emb = rearrange(emg_emb, 'b e ... -> (b e) ...',b=20, e=16)
#             label = out_dict['label']
#             emb_list.append(emb.detach().cpu())
#             eeg_emb_list.append(eeg_emb.detach().cpu())
#             emg_emb_list.append(emg_emb.detach().cpu())
#             label_list.append(label.detach().cpu().reshape(-1))

#         embs = torch.cat(emb_list, dim=0)[:]
#         eeg_embs = torch.cat(eeg_emb_list, dim=0)[:]
#         emg_embs = torch.cat(emg_emb_list, dim=0)[:]
#         labels = torch.cat(label_list, dim=0)[:]
#         embs, eeg_embs, emg_embs, labels = filter_func([embs, eeg_embs, emg_embs, labels], labels)
#         labels = labels.numpy()

#         class_mapping = {
#             0 : 'Awake',
#             1 : 'SWS',
#             2 : 'REM'
#         }
#         cmap = np.array(sns.xkcd_palette(["salmon", "amber", "dodger blue"]))

#         patches = [
#             mpatches.Patch(color=cmap[cls_idx], label="{:s}".format(cls))
#             for cls_idx, cls in class_mapping.items()
#         ]
#         modality = ['EEG-EMG', 'EEG', 'EMG']
#         fig = plt.figure(figsize=(60, 20))

#         for i, embs in enumerate([embs, eeg_embs, emg_embs]):

#             tsne = TSNE(n_components=2, n_jobs=-1, verbose=1)
#             embs_2d = tsne.fit_transform(embs)
#             ax1 = plt.subplot(1,3,i+1)
#             ax1.scatter(embs_2d[:,0], embs_2d[:,1], lw=0, s=40, c=cmap[labels.astype(np.int64)])
#             # ax1.axis('tight')
#             ax1.set_title('{} embeddings'.format(modality[i]), fontsize=16)
#         fig.suptitle('t-SNE projection of the learned embeddings', fontsize=24)
#         fig.legend(handles=patches, loc='upper right', labelspacing=0.1, fontsize=20)
#         fig.savefig('{}/tsne.png'.format(tsne_dir), dpi=400)
#         plt.close(fig)
