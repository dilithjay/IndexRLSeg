import os
import argparse
import pandas as pd
from glob import glob
from dataset_config import sample_dict, n_dataset_classes
from model import load_model, get_model_config
from dataset import SegIndexDataset
from utils import set_seed

import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--mode",
    default="baseline",
    help="Choose mode to update the dataset (baseline, concat, concat_multi, replace, replace_multi, best, nonminified, train_size)",
)
args = parser.parse_args()
mode = args.mode
is_multiclass = mode == "multiclass"
model_dir = f"models/{mode}"

test_samples = sample_dict["test"]
encoder = "resnet50"
optim = "adamw"
seed = 0
batch_size = 8
metrics = ["test_dataset_iou", "test_per_image_iou"]
results = {metric: [] for metric in metrics}
paths = sorted(glob(os.path.join(model_dir, "*.ckpt")))
names = []


for path in paths:
    name = os.path.basename(path).split(".")[0]
    names.append(name)

    arch, dataset_name, exps, replace_indices, channels = get_model_config(mode, name)
    set_seed(seed, arch not in ("deeplabv3", "pan", "manet", "msnet", "cainet"))

    data_dir = "dataset/test"
    samples = test_samples[dataset_name]
    eval_set = SegIndexDataset(
        data_dir,
        samples,
        exps,
        dataset_name,
        False,
        replace_indices,
        is_multiclass=is_multiclass,
    )
    eval_loader = DataLoader(eval_set, batch_size, False, num_workers=0)
    n_classes = (
        n_dataset_classes[dataset_name] if dataset_name in n_dataset_classes else 1
    )

    results_seed = {metric: [] for metric in metrics}
    for model_path in [path]:
        model = load_model(
            model_path,
            arch,
            replace_indices,
            dataset_name,
            num_channels=len(channels),
        )
        model.freeze()

        trainer = pl.Trainer()

        out = trainer.test(model, eval_loader, None, True)
        print(out)
        for metric in results:
            results_seed[metric].append(out[0][metric])

    for metric in results:
        results[metric].append(np.mean(results_seed[metric]))

df = pd.DataFrame()
df["name"] = names
for metric in results:
    df[metric.split("_", 2)[-1]] = results[metric]
df.to_csv(os.path.join(model_dir, f"results_{mode}.csv"), index=False)
