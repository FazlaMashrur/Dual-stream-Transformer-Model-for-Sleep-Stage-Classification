# import os
# import shutil
import logging

import torch
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

# from torch.autograd import Variable
from sklearn.metrics import cohen_kappa_score


def accuracy(label, pred):
    label = np.concatenate(label)
    pred = np.concatenate(pred)


def kappa(output, label):
    preds = torch.argmax(output, 1)
    return cohen_kappa_score(label, preds)


def plot_confusion_matrix(
    cm, target_names, title="Confusion matrix", cmap=None, normalize=True
):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")
    if normalize:
        cm2 = cm.astype("float") / cm.sum(axis=0)  # [:, np.newaxis]
    # print(cm2)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(
        np.transpose(cm2 * 100), interpolation="nearest", cmap=cmap, vmin=-5, vmax=80
    )
    plt.title(title, fontsize=20)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=15)
        plt.yticks(tick_marks, target_names, fontsize=15)

    thresh = 500  # cm.max() / 4 if normalize else cm.max() / 4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                i,
                j,
                "{:,}\n{:0.2f}%".format(int(cm[i, j]), cm2[i, j] * 100),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )
        else:
            plt.text(
                i,
                j,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=18)
    plt.xlabel(
        "Predictions", fontsize=18
    )  # \naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    fig.savefig("confusion_matrix.png")


def confusion_matrix(output, label, n_classes, batch_size, print_conf_mat=False):
    preds = torch.argmax(output, 1)

    conf_matrix = torch.zeros(n_classes, n_classes)
    avg_sensitivity = 0
    avg_specificity = 0
    avg_F1_score = 0
    avg_precision = 0
    sens_list = []
    spec_list = []
    F1_list = []
    precision_list = []

    for p, t in zip(preds, label):
        if torch.is_tensor(p):
            p = p.item()
            t = int(t.item())
        conf_matrix[p, t] += 1
    if print_conf_mat == True:
        # print(conf_matrix)

        plot_confusion_matrix(
            cm=conf_matrix.cpu().numpy(),
            normalize=True,
            target_names=["Wake", "SWS", "REM"],
            title="Confusion Matrix (3-Class)",
        )

        plt.show()

    TP = conf_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        if (TP[c] + FN) != 0:
            sensitivity = TP[c] / (TP[c] + FN)
        else:
            sensitivity = 0

        if (TN + FP) != 0:
            specificity = TN / (TN + FP)
        else:
            specificity = 0

        if ((2 * TP[c]) + (FN + FP)) != 0:
            F1_score = (2 * TP[c]) / ((2 * TP[c]) + (FN + FP))
        else:
            F1_score = 0

        if (TP[c] + FP) != 0:
            precision = TP[c] / (TP[c] + FP)
        else:
            precision = 0

        sens_list.append(float(sensitivity))
        spec_list.append(float(specificity))
        F1_list.append(float(F1_score))
        precision_list.append(float(precision))

        avg_sensitivity += float(sensitivity)
        avg_specificity += float(specificity)
        avg_F1_score += float(F1_score)
        avg_precision += float(precision)
    return (
        sens_list,
        spec_list,
        F1_list,
        precision_list,
        avg_sensitivity / n_classes,
        avg_specificity / n_classes,
        avg_F1_score / n_classes,
        avg_precision / n_classes,
    )


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset2update(self, val, n=1):
        self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.5f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.5f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.5f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(" ".join(entries))
        logging.getLogger("logger").info(" ".join(entries))

    def display_summary(self):
        entries = [self.prefix + " *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))
        logging.getLogger("logger").info(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"