from glob import glob
import os
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


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


def get_normalized_index(img, exp):
    max_z = 3
    idx = eval_expression(exp, img)
    idx = (idx - idx.mean()) / (idx.std() if idx.std() != 0 else 1e-5)
    return (np.clip(idx, -max_z, max_z) + max_z) / (2 * max_z)


def update_image(img, exps, replace_indices=None):
    for i, exp in enumerate(exps):
        if len(exp) == 0:
            continue
        idx = get_normalized_index(img, exp)
        if len(replace_indices) == 0:
            img = np.concatenate([img, idx[None, :, :]], axis=0)
        else:
            img[replace_indices[i]] = idx
    img = img.transpose(1, 2, 0)
    img = TF.to_tensor(img)
    return img


class SegIndexDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        samples: list,
        exps: list,
        dataset_name: str,
        is_train: bool = False,
        replace_indices=None,
        is_multiclass: bool = False,
    ):
        super().__init__()
        self.img_list = sorted(
            glob(os.path.join(data_dir, f"images-{dataset_name}", "*.npy"))
        )
        self.mask_list = sorted(
            glob(os.path.join(data_dir, f"masks-{dataset_name}", "*.npy"))
        )
        self.samples = samples
        self.exps = exps
        self.replace_indices = replace_indices
        self.is_train = is_train
        self.is_multiclass = is_multiclass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        index = self.samples[idx]
        img = np.load(self.img_list[index]).astype(float)

        img = update_image(img, self.exps, self.replace_indices)

        mask = np.load(self.mask_list[index]).astype(float)
        if self.is_multiclass and len(mask.shape) > 2:
            mask = np.argmax(mask, axis=2)
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

        return {"image": img.float(), "mask": mask.long()}

    def get_n_classes(self):
        shape = self[0]["mask"].shape
        if len(shape) == 3:
            return shape[-1]
        return 1
