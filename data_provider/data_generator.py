from torch.utils.data import DataLoader
from data_provider.data_loader import Epoch_Loader, Seq_Loader, Item_Loader

data_dict = {
    "Epoch": Epoch_Loader,
    "Seq": Seq_Loader,
}


def data_summarize(data_loader):
    trace, label = next(iter(data_loader))
    print(f"\t Traces batch shape: {trace.shape}")
    print(f"\t Labels batch shape: {label.shape}")


def data_generator(args, flag):
    Data = data_dict[args.data]
    batch_size = args.batch_size

    if flag == "val":
        shuffle_flag = False
        drop_last = False
        isEval = True
    else:
        shuffle_flag = True
        drop_last = True
        isEval = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        isEval=isEval,
        fold=args.fold,
        n_sequences=args.n_sequences,
        useNorm=args.useNorm,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )

    return data_set, data_loader


def visualize_data_generator(args):
    data_set = Item_Loader(
        root_path=args.root_path,
        data_path=args.data_path,
        isEval=True,
        fold=args.fold,
        n_sequences=args.n_sequences,
        useNorm=args.useNorm,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return data_set, data_loader
