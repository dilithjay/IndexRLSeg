import os
import pandas as pd
from glob import glob

test_samples = {
    "car": [55, 111, 61, 208, 194, 116, 191, 74, 197, 90, 184, 107, 171, 77, 195, 81, 15, 14, 93, 19],
    "person": [187, 243, 155, 165, 91, 21, 269, 161, 268, 244, 230, 40, 261, 46, 233, 104, 165, 239, 270, 57],
    "bike": [6, 54, 66, 68, 24, 104, 24, 104, 11, 136, 67, 40, 129, 139, 137, 54, 153, 49, 7, 119],
}

expressions = {
    "car": [["c1", "sqrt", "="], ["(", "c2", "sq", "-", "c3", "sqrt", ")", "="]],
    "person": [["(", "c3", "/", "c2", ")", "*", "c0", "="], ["c3", "/", "c1", "sq", "*", "c0", "sq", "="]],
    "bike": [
        ["(", "c3", "+", "c1", ")", "sq", "-", "c0", "sqrt", "="],
        ["c3", "*", "c2", "sqrt", "/", "c3", "sq", "="],
    ],
}

encoder = "resnet50"

optim = "adamw"
seed = 0
metrics = [
    "test_dataset_iou",
    "test_per_image_iou",
]
results = {metric: [] for metric in metrics}
model_dir = "models/4-concat_multi"
paths = glob(os.path.join(model_dir, "*.ckpt"))
names = []
for path in paths:
    name = os.path.basename(path).split(".")[0]
    names.append(name)
    arch, dataset = name.split("_")
    samples = test_samples[dataset]
    expression = expressions[dataset]

    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from model import SegModel
    import os
    import random
    import torchvision.transforms.functional as TF
    import pytorch_lightning as pl

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
        def __init__(self, data_dir: str, samples: list, exp: list):
            super().__init__()
            self.img_list = sorted(glob(os.path.join(data_dir, f"images-{dataset}", "*.npy")))
            self.mask_list = sorted(glob(os.path.join(data_dir, f"masks-{dataset}", "*.npy")))
            self.samples = samples
            self.exp = exp

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            index = self.samples[idx]
            img = np.load(self.img_list[index]).astype(float)

            max_z = 3
            idx = eval_expression(self.exp[0], img)
            idx = (idx - idx.mean()) / idx.std()
            idx = (np.clip(idx, -max_z, max_z) + max_z) / (2 * max_z)
            img = np.concatenate([img, idx[None, :, :]], axis=0)

            idx = eval_expression(self.exp[1], img)
            idx = (idx - idx.mean()) / idx.std()
            idx = (np.clip(idx, -max_z, max_z) + max_z) / (2 * max_z)
            img = np.concatenate([img, idx[None, :, :]], axis=0)

            img = img.transpose(1, 2, 0)
            img = TF.to_tensor(img)

            mask = np.load(self.mask_list[index]).astype(float)
            mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))
            mask = mask.transpose(1, 2, 0)
            mask = TF.to_tensor(mask)

            return {"image": img.float(), "mask": mask.float()}

    data_dir = "dataset/test"

    eval_set = CustomDataset(data_dir, samples, expression)

    batch_size = 8
    eval_loader = DataLoader(eval_set, batch_size, False, num_workers=0)

    results_seed = {metric: [] for metric in metrics}
    for model_path in [path]:
        kwargs = dict(
            arch=arch,
            encoder_name=encoder,
            in_channels=eval_set[0]["image"].shape[0],
            out_classes=1,
            lr=0.001,
        )
        model = SegModel.load_from_checkpoint(model_path, **kwargs)
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
path = os.path.join(model_dir, "results.csv")
df.to_csv(path, index=False)
print("Done", path)
