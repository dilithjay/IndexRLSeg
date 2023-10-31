import neptune
from model import SegModel
import pandas as pd
from glob import glob
import os

encoder = "resnet50"
optim = "adamw"
lr_scheduler = "ReduceLROnPlateau"
# df_res = pd.read_csv("results.csv")
# df_res["exp"] = df_res["exp"].fillna("")
patience = 50
num_workers = 4
seed = 0
expressions = {
    "car": [
        [],
        ["c1", "sqrt", "="],
    ],
    "person": [
        [],
        ["(", "c3", "/", "c2", ")", "*", "c0", "="],
    ],
    "bike": [
        [],
        ["(", "c3", "+", "c1", ")", "sq", "-", "c0", "sqrt", "="],
    ],
}
for arch in ("unetplusplus", "unet"):
    for dataset in expressions:
        for expression in expressions[dataset]:
            lr = 1e-3
            import numpy as np
            import torch
            from torch.utils.data import Dataset, DataLoader
            from glob import glob
            import os
            import random
            import torchvision.transforms.functional as TF
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = arch != "deeplabv3"
            torch.use_deterministic_algorithms(arch != "deeplabv3")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            def eval_expression(exp: list, image: np.ndarray = None):
                expression = ""

                for token in exp:
                    if token[0] == "c":
                        channel = eval(token[1:])
                        expression += f"(image[{channel}] + 0.0001)"  # To prevent divide by zero
                    elif token == "sq":
                        expression += "**2"
                    elif token == "sqrt":
                        expression += "**0.5"
                    elif token == "=":
                        break
                    else:
                        expression += token

                return eval(expression)

            without_index = len(expression) == 0

            class CustomDataset(Dataset):
                def __init__(self, data_dir: str, exp: list, is_train: bool):
                    super().__init__()
                    self.img_list = sorted(glob(os.path.join(data_dir, f"images-{dataset}", "*.npy")))
                    self.mask_list = sorted(glob(os.path.join(data_dir, f"masks-{dataset}", "*.npy")))
                    self.is_train = is_train
                    self.exp = exp

                def __len__(self):
                    return len(self.img_list)

                def __getitem__(self, index):
                    img = np.load(self.img_list[index]).astype(float)

                    if without_index:
                        img = img[:4, :, :]
                    else:
                        idx = eval_expression(self.exp, img)
                        max_z = 3
                        idx = (idx - idx.mean()) / idx.std()
                        idx = (np.clip(idx, -max_z, max_z) + max_z) / (2 * max_z)
                        img = np.concatenate([img[:4, :, :], idx[None, :, :]], axis=0)
                    img = img.transpose(1, 2, 0)
                    img = TF.to_tensor(img)

                    mask = np.load(self.mask_list[index]).astype(float)
                    mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))
                    mask = mask.transpose(1, 2, 0)
                    mask = TF.to_tensor(mask)

                    # Transforms
                    if self.is_train and random.random() > 0.5:
                        img = TF.hflip(img)
                        mask = TF.hflip(mask)
                    if self.is_train and random.random() > 0.5:
                        img = TF.vflip(img)
                        mask = TF.vflip(mask)

                    return {"image": img.float(), "mask": mask.float()}

            data_dir = "dataset/"

            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(0)

            train_set = CustomDataset(data_dir + "train/", expression, True)
            val_set = CustomDataset(data_dir + "val/", expression, False)

            batch_size = 8
            train_loader = DataLoader(
                train_set,
                batch_size,
                True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_set,
                batch_size,
                False,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
            )

            run = neptune.init_run(
                project="dilithjay/IndexRL",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MjVlMDAxOS0wYjQyLTRhMDEtOWNlOS1hZWEwMGE5MzQ4NTUifQ==",
            )
            params = {
                "patience": patience,
                "batch_size": batch_size,
                "encoder": encoder,
                "optimizer": optim,
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "without_index": without_index,
                "seed": seed,
                "arch": arch,
                "num_workers": num_workers,
                "dataset": f"irseg-{dataset}",
                "expression": "" if without_index else str(expression),
            }
            run["parameters"] = params

            model = SegModel(
                arch,
                encoder,
                in_channels=train_set[0]["image"].shape[0],
                out_classes=1,
                lr=lr,
                run=run,
            )

            model.to(device)
            model = model.float()

            trainer = pl.Trainer(
                min_epochs=20,
                max_epochs=2000,
                enable_checkpointing=True,
                logger=False,
                # precision="16-mixed",
                callbacks=[
                    EarlyStopping("valid_loss", patience=patience),
                    ModelCheckpoint(
                        dirpath=f"models/{dataset}_{arch}_{'idxrl' if expression else 'regular'}_{lr=}_{seed=}/",
                        save_top_k=1,
                        monitor="valid_loss",
                    ),
                ],
            )

            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            run.stop()
