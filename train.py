import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
import os
import neptune
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import logging
from dataset import SegIndexDataset

from model import load_model
from dataset_config import (
    sample_dict,
    expressions,
    ds_replace_indices_1,
    ds_replace_indices_2,
)
from utils import set_seed

logging.disable(logging.CRITICAL)


def train_model(
    arch,
    dataset_name,
    exps,
    lr,
    seed,
    model_name,
    batch_size=8,
    replace_indices=None,
    encoder="resnet50",
    model_dir="",
    patience=1000,
    run=None,
    min_epochs=20,
    max_epochs=10000,
    sample_sizes=tuple(),
    deterministic=True,
):
    set_seed(
        seed,
        (arch not in ("deeplabv3", "manet", "pan", "msnet", "cainet"))
        and deterministic,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "dataset/"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_samples = sample_dict["train"][dataset_name]
    val_samples = sample_dict["val"][dataset_name]
    if sample_sizes:
        train_size, val_size = sample_sizes
        rng = random.Random(seed)
        train_samples = rng.sample(train_samples, train_size)
        val_samples = rng.sample(val_samples, val_size)

    train_set = SegIndexDataset(
        train_dir,
        train_samples,
        exps,
        dataset_name,
        True,
        replace_indices,
    )
    val_set = SegIndexDataset(
        val_dir,
        val_samples,
        exps,
        dataset_name,
        False,
        replace_indices,
    )

    train_loader = DataLoader(train_set, batch_size, True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size, False, num_workers=0)

    model = load_model(
        model_path="",
        arch=arch,
        replace_indices=replace_indices,
        dataset_name=dataset_name,
        lr=lr,
        run=run,
        num_channels=train_set[0]["image"].shape[0],
    )

    model.to(device)
    model = model.float()

    callbacks = [EarlyStopping("valid_loss", patience=patience)]
    enable_checkpointing = bool(model_name)
    if enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=model_dir,
                filename=model_name,
                save_top_k=1,
                monitor="valid_loss",
            )
        )

    trainer = pl.Trainer(
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        enable_checkpointing=enable_checkpointing,
        logger=False,
        callbacks=callbacks,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    out = trainer.validate(model, dataloaders=val_loader)
    print(out)

    return out


def iterate_matrix(archs, datasets, replace_idxs, lrs):
    for arch in archs:
        for dataset_name in datasets:
            for replace_indices in replace_idxs[dataset_name]:
                for lr in lrs:
                    yield arch, dataset_name, replace_indices, lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="baseline",
        help="Choose mode to update the dataset (baseline, concat, concat_multi, replace, replace_multi)",
    )
    parser.add_argument(
        "-md",
        "--model_dir",
        default="models-train/",
        help="Directory to output the models. Subdirectories will be created for each mode.",
    )
    parser.add_argument(
        "-np",
        "--neptune_project",
        default="",
        help="Name of your neptune project",
    )
    parser.add_argument(
        "-nt",
        "--neptune_token",
        default="",
        help="Neptune API token",
    )
    args = parser.parse_args()

    mode = args.mode
    model_dir = os.path.join(args.model_dir, mode)
    os.makedirs(model_dir, exist_ok=True)

    optim = "adamw"
    encoder = "resnet50"
    patience = 1000
    seed = 0
    batch_size = 8
    archs = ("unet", "deeplabv3", "unetplusplus", "pan", "manet", "msnet", "cainet")
    datasets = (
        "car",
        "person",
        "bike",
        "cloud",
        "landslide",
        "grass",
        "sand",
        "irseg",
        "rit18",
    )
    lrs = (1e-3,)
    if mode == "replace":
        ds_replace_indices = ds_replace_indices_1
    elif mode == "replace_multi":
        ds_replace_indices = ds_replace_indices_2
    else:
        ds_replace_indices = {dataset_name: [tuple()] for dataset_name in datasets}

    for arch, dataset_name, replace_indices, lr in iterate_matrix(
        archs, datasets, ds_replace_indices, lrs
    ):
        if arch == "cainet" and (
            "concat" in mode or dataset_name in ("landslide", "cloud")
        ):
            print(f"Settings ({arch=}, {dataset_name=}, {mode=}) not supported")
            continue
        if mode == "baseline":
            exps = []
        elif mode in ("concat", "replace"):
            exps = [expressions[dataset_name][0]]
        elif mode in ("concat_multi", "replace_multi"):
            exps = expressions[dataset_name]
        print(arch, dataset_name, exps, replace_indices, lr)
        model_name = f"{arch}_{dataset_name}_{mode}_{lr}"
        model_path = os.path.join(model_dir, f"{model_name}.ckpt")
        if os.path.exists(model_path):
            os.remove(model_path)
        use_neptune = args.neptune_project and args.neptune_token
        if use_neptune:
            run = neptune.init_run(
                project=args.neptune_project,
                api_token=args.neptune_token,
            )
            params = {
                "patience": patience,
                "batch_size": batch_size,
                "encoder": encoder,
                "optimizer": optim,
                "lr": lr,
                "seed": seed,
                "arch": arch,
                "num_workers": 0,
                "dataset": dataset_name,
                "expression": str(exps),
                "mode": mode,
            }
            run["parameters"] = params
        else:
            run = None
        train_model(
            arch,
            dataset_name,
            exps,
            lr,
            seed,
            model_name,
            batch_size,
            replace_indices,
            encoder,
            model_dir,
            patience,
            run,
        )
        if use_neptune:
            run["model/best_model"].upload(model_path)
            run.stop()
