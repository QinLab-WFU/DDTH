import argparse
from os import path as osp


def get_config():
    parser = argparse.ArgumentParser(description=osp.basename(osp.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50", help="see _network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--multi-thread", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=16, help="length of hashing binary")

    # special settings
    parser.add_argument("--method", type=str, default="sct-margin", help="sct-nca/sct-margin")
    parser.add_argument("--lam", type=float, default=1, help="hyper-parameter lambda: 1/0.1")
    parser.add_argument("--temperature", type=float, default=0.1, help="hyper-parameter for nca loss")
    parser.add_argument("--margin", type=float, default=0.25, help="hyper-parameter for triplet loss")
    parser.add_argument("--xbm-size", type=int, default=0, help="hyper-parameter XBM.SIZE")

    args = parser.parse_args()

    # mods
    # args.xbm_size = 2048
    # args.method = "sct-nca"
    # args.n_epochs = 200

    return args
