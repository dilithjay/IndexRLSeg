import argparse
import os
import pickle
import torch
from glob import glob
from pathlib import Path

from indexrl.training import (
    DynamicBuffer,
    create_model,
    save_model,
    explore,
    train_iter,
)
from indexrl.environment import IndexRLEnv
from indexrl.utils import get_n_channels, set_seed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-dd",
    "--data_dir",
    default="dataset/train/",
    help="Directory with the entire training set",
)
parser.add_argument(
    "-o",
    "--indexrl_out_dir",
    default="indexrl_out/",
    help="Directory to save all outputs of the training (models, logs, and cache)",
)
parser.add_argument(
    "-a",
    "--arch",
    choices=("gpt", "lstm"),
    default="gpt",
    help="Agent model architecture",
)
parser.add_argument(
    "-dn",
    "--dataset_name",
    help="Name of the dataset to find indices for",
)
args = parser.parse_args()

set_seed(0)

data_dir = args.data_dir
indexrl_out_dir = args.indexrl_out_dir
os.environ["DATASET_NAME"] = dataset_name = postfix = args.dataset_name

image_dir = os.path.join(data_dir, f"images-{postfix}")
mask_dir = os.path.join(data_dir, f"masks-{postfix}")

img_path = glob(os.path.join(image_dir, "*.npy"))[0]
n_channels = get_n_channels(img_path)

cache_dir = os.path.join(indexrl_out_dir, f"cache-{postfix}")
logs_dir = os.path.join(indexrl_out_dir, f"logs-{postfix}")
models_dir = os.path.join(indexrl_out_dir, f"models-{postfix}")
for dir_name in (cache_dir, logs_dir, models_dir):
    Path(dir_name).mkdir(parents=True, exist_ok=True)

action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

env = IndexRLEnv(action_list, 12)
agent, optimizer = create_model(
    args.arch,
    len(action_list),
)
seen_path = os.path.join(cache_dir, "seen.pkl") if cache_dir else ""
env.save_seen(seen_path)
data_buffer = DynamicBuffer()

train_start_count = 1000

i = 0
while True:
    i += 1
    i_str = str(i).rjust(3, "0")

    data = explore(
        env.copy(),
        agent,
        image_dir,
        mask_dir,
        10,
        logs_dir,
        seen_path,
        n_iters=1000,
    )

    data_buffer.add_data(data)
    if cache_dir and i % 10 == 0:
        with open(f"{cache_dir}/data_buffer_{i_str}.pkl", "wb") as fp:
            pickle.dump(data_buffer, fp)

    print("Buffer size:", len(data_buffer))

    if len(data_buffer) < train_start_count:
        continue

    agent, optimizer, loss = train_iter(agent, optimizer, data_buffer)
    assert not torch.isnan(torch.tensor(loss))

    if models_dir:
        save_model(agent, f"{models_dir}/model_{i_str}_loss-{loss}.pt")
