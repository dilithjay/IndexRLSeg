from model import SegModel
import pandas as pd


encoder = "resnet50"
train_sample_dict = {
    "car": [
        1054,
        147,
        1198,
        1397,
        183,
        1449,
        1136,
        1199,
        474,
        828,
        159,
        775,
        1123,
        617,
        1156,
        1142,
        267,
        612,
        654,
        166,
    ],
    "person": [737, 92, 522, 1226, 728, 975, 958, 977, 922, 1005, 512, 899, 416, 1160, 894, 834, 458, 722, 1153, 190],
    "bike": [1061, 1135, 1083, 906, 179, 948, 689, 1151, 1188, 1136, 158, 149, 363, 31, 1232, 254, 214, 372, 1112, 158],
}
val_sample_dict = {
    "car": [1281, 1087, 915, 555, 311, 766, 1351, 1377, 118, 1322, 566, 713, 1518, 731, 355, 763, 571, 907, 185, 81],
    "person": [947, 1265, 606, 734, 612, 446, 1059, 918, 281, 863, 302, 613, 895, 601, 184, 478, 1093, 557, 273, 446],
    "bike": [240, 301, 751, 92, 570, 381, 1058, 24, 1089, 124, 554, 887, 357, 92, 591, 552, 814, 956, 684, 997],
}

expressions = {
    "car": [
        ["(", "c2", "+", "(", "c2", "sq", "*", "c3", "sq", "*", "c2", ")", ")", "sqrt", "="],
        ["(", "c2", "-", "(", "c3", "sqrt", "/", "c1", ")", "sqrt", ")", "="],
    ],
    "person": [
        ["(", "(", "c2", "+", "c2", ")", "sqrt", ")", "-", "c0", "sq", "="],
        ["(", "c2", "+", "(", "c3", "*", "c2", "+", "c1", ")", "sqrt", "-", "c1", ")", "="],
    ],
    "bike": [
        ["(", "c3", "/", "c0", "sqrt", "*", "c0", "*", "c1", "sq", ")", "sq", "="],
        ["(", "c0", "-", "(", "c1", "/", "c0", "*", "c3", ")", "+", "c2", ")", "sq", "="],
    ],
}

optim = "adamw"
df_res = pd.read_csv("results.csv")
df_res["exp"] = df_res["exp"].fillna("")

for seed in range(5):
    for dataset in expressions:
        for arch in ("unet", "unetplusplus", "deeplabv3"):
            exps = expressions[dataset]
            samples = train_sample_dict[dataset]

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

            class CustomDataset(Dataset):
                def __init__(self, data_dir: str, samples: list, exps: list, is_train: bool):
                    super().__init__()
                    self.img_list = sorted(glob(os.path.join(data_dir, f"images-{dataset}", "*.npy")))
                    self.mask_list = sorted(glob(os.path.join(data_dir, f"masks-{dataset}", "*.npy")))
                    self.is_train = is_train
                    self.exps = exps
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    index = self.samples[idx]
                    img = np.load(self.img_list[index]).astype(float)

                    for exp in self.exps:
                        idx = eval_expression(exp, img)
                        max_z = 3
                        idx = (idx - idx.mean()) / idx.std()
                        idx = (np.clip(idx, -max_z, max_z) + max_z) / (2 * max_z)
                        img = np.concatenate([img, idx[None, :, :]], axis=0)

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

            train_set = CustomDataset(data_dir, train_sample_dict[dataset], exps, True)
            val_set = CustomDataset(data_dir, val_sample_dict[dataset], exps, False)

            batch_size = 8
            train_loader = DataLoader(train_set, batch_size, True, num_workers=0)
            val_loader = DataLoader(val_set, batch_size, False, num_workers=0)

            model = SegModel(
                arch,
                encoder,
                in_channels=train_set[0]["image"].shape[0],
                out_classes=1,
                lr=0.0001,
            )

            model.to(device)
            model = model.float()
            trainer = pl.Trainer(
                min_epochs=20,
                max_epochs=2000,
                enable_checkpointing=True,
                logger=True,
                callbacks=[
                    EarlyStopping("valid_loss", patience=40),
                    ModelCheckpoint(
                        dirpath=f"models/{dataset}_{arch}_multi_idxrl_{seed}/",
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

            torch.save(model, f"models/{dataset}_{arch}_multi_idxrl_{seed}.pkl")
