# Reference: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb

import os
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from dataset_config import (
    cloud_params,
    landslide_params,
    rit_params,
    class_rgb_channels,
    n_dataset_classes,
    n_channels,
    expressions,
)
from msnet import MSNet

if os.path.exists("CAINet/"):
    import sys

    sys.path.append("CAINet/")
    from CAINet.toolbox.models.cainet import mobilenetGloRe3_CRRM_dule_arm_bou_att

device = "cuda" if torch.cuda.is_available() else "cpu"


class SegModelReplace(pl.LightningModule):
    def __init__(
        self,
        dataset_name,
        in_channels=None,
        lr=0.001,
        encoder_name="resnet50",
        arch=None,
        replace_indices=None,
        run=None,
    ):
        super().__init__()

        if not in_channels:
            in_channels = n_channels[dataset_name]

        if arch not in ("cainet", "msnet"):
            self.model = smp.create_model(
                arch,
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=(
                    n_dataset_classes[dataset_name]
                    if dataset_name in n_dataset_classes
                    else 1
                ),
            )
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = True

        if arch != "cainet":
            img_params = load_params(dataset_name)
            for idx in replace_indices:
                img_params["mean"][idx] = 0
                img_params["std"][idx] = 1
            std_ = img_params["std"] + [1] * (in_channels - len(img_params["std"]))
            mean_ = img_params["mean"] + [0] * (in_channels - len(img_params["std"]))

            self.register_buffer("std", torch.tensor(std_).view(1, in_channels, 1, 1))
            self.register_buffer("mean", torch.tensor(mean_).view(1, in_channels, 1, 1))

        self.is_multiclass = (dataset_name in n_dataset_classes) and (
            n_dataset_classes[dataset_name] > 1
        )
        # for image segmentation dice loss could be the best first choice
        loss_mode = (
            smp.losses.MULTICLASS_MODE if self.is_multiclass else smp.losses.BINARY_MODE
        )
        self.loss_fn = smp.losses.DiceLoss(loss_mode, from_logits=True)

        self.lr = lr
        self.validation_step_outputs = []
        self.run = run

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f"{h=} or {w=} is not divisible by 32"

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4
        #         assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        if self.is_multiclass:
            pred_mask = torch.argmax(logits_mask, dim=1, keepdim=True)
        else:
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

        is_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode="multiclass" if self.is_multiclass else "binary",
            num_classes=logits_mask.shape[1],
        )
        torch.use_deterministic_algorithms(is_deterministic)

        out_dict = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "bsz": image.shape[0],
        }
        self.outputs.append(out_dict)
        return out_dict

    def shared_epoch_start(self):
        self.outputs = []

    def shared_epoch_end(self, stage):
        outputs = self.outputs
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        dataset_prec = torch.tensor(0) if torch.isnan(dataset_prec) else dataset_prec
        dataset_rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        dataset_rec = torch.tensor(0) if torch.isnan(dataset_rec) else dataset_rec
        dataset_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        total_bsz = sum([x["bsz"] for x in outputs])
        loss = torch.tensor([x["loss"] * x["bsz"] for x in outputs]).mean() / total_bsz

        # log result to neptune
        if self.run:
            self.run[f"{stage}/per_image_iou"].append(per_image_iou)
            self.run[f"{stage}/dataset_iou"].append(dataset_iou)
            self.run[f"{stage}/dataset_prec"].append(dataset_prec)
            self.run[f"{stage}/dataset_rec"].append(dataset_rec)
            self.run[f"{stage}/dataset_acc"].append(dataset_acc)
            self.run[f"{stage}/dataset_f1"].append(dataset_f1)
            self.run[f"{stage}/loss"].append(loss)

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_dataset_prec": dataset_prec,
            f"{stage}_dataset_rec": dataset_rec,
            f"{stage}_dataset_acc": dataset_acc,
            f"{stage}_dataset_f1": dataset_f1,
            f"{stage}_loss": loss,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        return self.shared_epoch_start()

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        return self.shared_epoch_start()

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        return self.shared_epoch_start()

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class SegModelMSNet(SegModelReplace):
    def __init__(self, dataset_name, lr, in_channels, replace_indices=None, run=None):
        rgb_channels = class_rgb_channels[dataset_name]
        nnn_channels = sorted(set(range(n_channels[dataset_name])) - set(rgb_channels))

        super().__init__(
            arch="msnet",
            in_channels=in_channels,
            dataset_name=dataset_name,
            lr=lr,
            replace_indices=replace_indices,
            run=run,
        )
        self.model = MSNet(
            rgb_channels,
            nnn_channels,
            n_dataset_classes[dataset_name] if dataset_name in n_dataset_classes else 1,
        )


class SegModelCAINet(SegModelReplace):
    def __init__(
        self,
        lr,
        dataset_name,
        in_channels,
        replace_indices=None,
        run=None,
    ):
        super().__init__(
            arch="cainet",
            in_channels=in_channels,
            dataset_name=dataset_name,
            lr=lr,
            replace_indices=replace_indices,
            run=run,
        )
        self.model = mobilenetGloRe3_CRRM_dule_arm_bou_att(
            n_classes=(
                n_dataset_classes[dataset_name]
                if dataset_name in n_dataset_classes
                else 1
            )
        )

        img_params = load_params(dataset_name)
        for idx in replace_indices:
            img_params["mean"][idx] = 0
            img_params["std"][idx] = 1
        n = len(img_params["std"])
        self.rgb_std = torch.tensor(img_params["std"][:3]).view(1, 3, 1, 1).to(device)
        self.rgb_mean = torch.tensor(img_params["mean"][:3]).view(1, 3, 1, 1).to(device)
        self.depth_std = (
            torch.tensor(img_params["std"][3:]).view(1, n - 3, 1, 1).to(device)
        )
        self.depth_mean = (
            torch.tensor(img_params["mean"][3:]).view(1, n - 3, 1, 1).to(device)
        )

        # Update the loss function to CrossEntropyLoss to match training code
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image, depth):
        # Normalize image here
        image = (image - self.rgb_mean) / self.rgb_std
        # print(depth.shape, self.depth_mean.shape)
        depth = (depth - self.depth_mean) / self.depth_std
        if depth.shape[0] == 1:
            depth = depth.squeeze().unsqueeze(1)  # Ensure depth has the correct shape
            depth = torch.concat([depth, depth, depth], axis=1)
        mask = self.model(image, depth)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"][:, :3].to(device)  # RGB image
        if batch["image"].shape[1] == 4:
            depth = batch["image"][:, 3].to(device)  # Depth channel
        else:
            depth = batch["image"][:, 3:].to(device)  # Depth channel
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f"{h=} or {w=} is not divisible by 32"

        mask = batch["mask"]
        assert mask.ndim == 4
        #         assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image, depth)
        loss = self.loss_fn(logits_mask, mask.squeeze())
        pred_mask = torch.argmax(logits_mask, dim=1, keepdim=True)

        is_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode="multiclass",
            num_classes=logits_mask.shape[1],
        )
        torch.use_deterministic_algorithms(is_deterministic)

        out_dict = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "bsz": image.shape[0],
        }
        self.outputs.append(out_dict)
        return out_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda ep: (1 - ep / self.trainer.max_epochs) ** 0.9
        )
        return [optimizer], [scheduler]


def load_params(dataset_name):
    if dataset_name == "cloud":
        return cloud_params
    elif dataset_name == "landslide":
        return landslide_params
    elif dataset_name in ("grass", "sand", "rit18"):
        return rit_params
    else:
        img_params = smp.encoders.get_preprocessing_params("resnet50")
        img_params["std"] += [0.0765]
        img_params["mean"] += [0.3445]
        return img_params


def load_model(
    model_path,
    arch,
    replace_indices,
    dataset_name,
    lr=0.001,
    num_channels=None,
    run=None,
):
    in_channels = num_channels if num_channels else n_channels[dataset_name]
    if arch == "msnet":
        kwargs = dict(
            in_channels=in_channels,
            lr=lr,
            dataset_name=dataset_name,
            replace_indices=replace_indices,
            run=run,
        )
        if model_path:
            model = SegModelMSNet.load_from_checkpoint(model_path, **kwargs)
        else:
            model = SegModelMSNet(**kwargs)
    elif arch == "cainet":
        kwargs = dict(
            in_channels=in_channels,
            lr=lr,
            dataset_name=dataset_name,
            replace_indices=replace_indices,
            run=run,
        )
        if model_path:
            model = SegModelCAINet.load_from_checkpoint(model_path, **kwargs)
        else:
            model = SegModelCAINet(**kwargs)
    else:
        kwargs = dict(
            arch=arch,
            encoder_name="resnet50",
            in_channels=in_channels,
            lr=lr,
            replace_indices=replace_indices,
            dataset_name=dataset_name,
            run=run,
        )
        if model_path:
            model = SegModelReplace.load_from_checkpoint(model_path, **kwargs)
        else:
            model = SegModelReplace(**kwargs)

    return model


def get_model_config(mode, name):
    if (
        mode == "best"
        or mode == "multiclass"
        or mode == "nonminified"
        or mode == "train_size"
        or mode == "ndvi"
    ):
        if mode == "train_size":
            # Remove train size from name
            splits = name.split("_")
            name = "_".join(splits[:2] + splits[3:])
        n_splits = name.count("_")
        if "baseline" in name:
            mode = "baseline"
        elif n_splits == 2:
            mode = "concat"
        elif "replace" in name:
            if n_splits == 3:
                mode = "replace"
            else:
                mode = "replace_multi"
        else:
            mode = "concat_multi"
        name = name.replace(f"_{mode}", "")

    if mode == "baseline" or "concat" in mode:
        splits = name.split("_")
        arch = splits[0]
        dataset_name = splits[1]
        exps = (
            []
            if mode == "baseline"
            else (
                [expressions[dataset_name][0]]
                if mode == "concat"
                else (
                    ["(", "c2", "-", "c7", ")", "/", "(", "c2", "+", "c7", ")", "="]
                    if splits[-1] == "ndwi"
                    else (
                        ["(", "c7", "-", "c3", ")", "/", "(", "c7", "+", "c3", ")", "="]
                        if splits[-1] == "ndvi"
                        else expressions[dataset_name]
                    )
                )
            )
        )
        replace_indices = []
        channels = ["a"] * n_channels[dataset_name] + ["1"] * len(exps)
    elif mode == "replace":
        splits = name.split("_")
        arch, dataset_name, replace_idx = splits[:3]
        exps = [expressions[dataset_name][0]]
        replace_indices = [int(replace_idx)]
        if len(splits) == 4:
            if splits[3] == "ndvi":
                exps = [
                    ["(", "c7", "-", "c3", ")", "/", "(", "c7", "+", "c3", ")", "="]
                ]
            elif splits[3] == "ndwi":
                exps = [
                    ["(", "c2", "-", "c7", ")", "/", "(", "c2", "+", "c7", ")", "="]
                ]
        channels = ["a"] * n_channels[dataset_name]
        channels[replace_indices[0]] = "1"
    else:
        arch, dataset_name, channels = name.split("_")
        exps = expressions[dataset_name]
        replace_indices = [channels.index("1"), channels.index("2")]
    return arch, dataset_name, exps, replace_indices, channels
