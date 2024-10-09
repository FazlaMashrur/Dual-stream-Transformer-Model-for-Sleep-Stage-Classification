from utils.metrics import AverageMeter


def batch_updater(metric_list, meter_list, batch_size):
    for metric, meter in zip(metric_list, meter_list):
        meter.update(metric, batch_size)


def batch_tracker(metric_list, format_list):
    return list(
        map(lambda metric, fmt: AverageMeter(metric, fmt), metric_list, format_list)
    )


def build_tracker(isEval=False):
    metrics = ["Time", "Loss", "Acc", "F1", "Kappa", "Precision", "Recall"]
    formats = [":6.3f", ":.4e", ":6.3f", ":6.3f", ":6.3f", ":6.3f", ":6.3f"]
    time, loss, acc, f1, kappa, prec, recall = batch_tracker(metrics, formats)

    if not isEval:
        return time, loss, acc, f1, kappa, prec, recall

    classMetrics = [
        "W-F1",
        "S-F1",
        "R-F1",
        "W-Prec",
        "S-Prec",
        "R-Prec",
        "W-Recall",
        "S-Recall",
        "R-Recall",
    ]
    formats = [
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
    ]
    w_f1, s_f1, r_f1, w_prec, s_prec, r_prec, w_rec, s_rec, r_rec = batch_tracker(
        classMetrics, formats
    )
    return (
        time,
        loss,
        acc,
        f1,
        kappa,
        prec,
        recall,
        w_f1,
        s_f1,
        r_f1,
        w_prec,
        s_prec,
        r_prec,
        w_rec,
        s_rec,
        r_rec,
    )


def build_tracker_mome(isEval=False):
    metrics = [
        "Time",
        "Loss",
        "Acc",
        "F1",
        "Acc_eeg",
        "F1_eeg",
        "Acc_emg",
        "F1_emg",
        "Kappa",
        "Precision",
        "Recall",
    ]
    formats = [
        ":6.3f",
        ":.4e",
        ":6.3f",
        ":6.3",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
    ]
    (
        time,
        loss,
        acc,
        f1,
        acc_eeg,
        f1_eeg,
        acc_emg,
        f1_emg,
        kappa,
        prec,
        recall,
    ) = batch_tracker(metrics, formats)
    if not isEval:
        return (
            time,
            loss,
            acc,
            f1,
            acc_eeg,
            f1_eeg,
            acc_emg,
            f1_emg,
            kappa,
            prec,
            recall,
        )

    classMetrics = [
        "W-F1",
        "S-F1",
        "R-F1",
        "W-Prec",
        "S-Prec",
        "R-Prec",
        "W-Recall",
        "S-Recall",
        "R-Recall",
    ]
    formats = [
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
    ]
    w_f1, s_f1, r_f1, w_prec, s_prec, r_prec, w_rec, s_rec, r_rec = batch_tracker(
        classMetrics, formats
    )
    return (
        time,
        loss,
        acc,
        f1,
        acc_eeg,
        f1_eeg,
        acc_emg,
        f1_emg,
        kappa,
        prec,
        recall,
        w_f1,
        s_f1,
        r_f1,
        w_prec,
        s_prec,
        r_prec,
        w_rec,
        s_rec,
        r_rec,
    )


def build_tracker_mome_wNE(isEval=False):
    metrics = [
        "Time",
        "Loss",
        "Acc",
        "F1",
        "Acc_eeg",
        "F1_eeg",
        "Acc_emg",
        "F1_emg",
        "Acc_ne",
        "F1_ne",
        "Kappa",
        "Precision",
        "Recall",
    ]

    formats = [
        ":6.3f",
        ":.4e",
        ":6.3f",
        ":6.3",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
    ]
    (
        time,
        loss,
        acc,
        f1,
        acc_eeg,
        f1_eeg,
        acc_emg,
        f1_emg,
        acc_ne,
        f1_ne,
        kappa,
        prec,
        recall,
    ) = batch_tracker(metrics, formats)
    if not isEval:
        return (
            time,
            loss,
            acc,
            f1,
            acc_eeg,
            f1_eeg,
            acc_emg,
            f1_emg,
            acc_ne,
            f1_ne,
            kappa,
            prec,
            recall,
        )

    classMetrics = [
        "W-F1",
        "S-F1",
        "R-F1",
        "W-Prec",
        "S-Prec",
        "R-Prec",
        "W-Recall",
        "S-Recall",
        "R-Recall",
    ]
    formats = [
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
        ":6.3f",
    ]
    w_f1, s_f1, r_f1, w_prec, s_prec, r_prec, w_rec, s_rec, r_rec = batch_tracker(
        classMetrics, formats
    )
    return (
        time,
        loss,
        acc,
        f1,
        acc_eeg,
        f1_eeg,
        acc_emg,
        f1_emg,
        acc_ne,
        f1_ne,
        kappa,
        prec,
        recall,
        w_f1,
        s_f1,
        r_f1,
        w_prec,
        s_prec,
        r_prec,
        w_rec,
        s_rec,
        r_rec,
    )
