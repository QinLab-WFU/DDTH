import json
import os
import time

import torch
from loguru import logger


from XBM_MOD.xbm import XBM
from _data import build_loaders, get_topk, get_class_num
from _network import build_model
from _utils import (
    AverageMeter,
    build_optimizer,
    calc_learnable_params,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
    calc_map_eval,
)
from config import get_config
from loss import SCTLoss


def train_epoch(args, dataloader, net, criterion, optimizer, xbm, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["n_triplets", "hn_ratio", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, _ in dataloader:
        images, labels = images.to(args.device), labels.to(args.device)
        embeddings = net(images)

        # xbm uses current embeddings
        # xbm.set(embeddings.detach(), labels)

        loss, hn_ratio, n_triplets = criterion(embeddings, labels, xbm)
        stat_meters["hn_ratio"].update(hn_ratio)
        stat_meters["n_triplets"].update(n_triplets)

        # xbm doesn't use current embeddings
        if xbm:
            xbm.set(embeddings.detach(), labels)

        if n_triplets == 0:
            continue

        stat_meters["loss"].update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        map_v = calc_map_eval(embeddings.detach().sign(), labels)
        stat_meters["mAP"].update(map_v)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    # setup net
    net = build_model(args, True)

    # setup criterion
    criterion = SCTLoss(args.method, args.lam, args.temperature, args.margin)

    logger.info(f"Number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, optimizer = train_init(args)

    xbm = XBM(args) if args.xbm_size > 0 else None

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, xbm, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                multi_thread=args.multi_thread,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"Without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"Reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()

    if "rename" in args and args.rename:
        rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        # for dataset in ["flickr"]:
        print(f"Processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        )
        args.n_samples = train_loader.dataset.__len__()

        for hash_bit in [16, 32, 64, 128]:
            # for hash_bit in [32]:
            print(f"Processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                # raise Exception(f"*.pkl exists in {args.save_dir}")
                print(f"*.pth exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    # for x in rst:
    #     print(
    #         f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
    #     )
    print_in_md(rst)


if __name__ == "__main__":
    main()
