# Reference: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts


class SegModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, run=None, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # preprocessing parameteres for image
        std_ = [0.0765] + [1] * (in_channels - 4)
        mean_ = [0.3445] + [0] * (in_channels - 4)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"] + std_).view(1, in_channels, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"] + mean_).view(1, in_channels, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # for neptune logging
        self.run = run

        self.lr = lr
        self.validation_step_outputs = []
        self.best_val_iou = 0

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

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

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
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        if stage == "valid":
            self.best_val_iou = max(self.best_val_iou, dataset_iou)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "valid_loss",
            },
        }


class SegModelMini(pl.LightningModule):
    def __init__(self, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model("unet", in_channels=in_channels, classes=out_classes, **kwargs)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # preprocessing parameteres for image
        std_ = [0.0765] + [1] * (in_channels - 4)
        mean_ = [0.3445] + [0] * (in_channels - 4)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"] + std_).view(1, in_channels, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"] + mean_).view(1, in_channels, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.validation_step_outputs = []
        self.best_val_iou = 0

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

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

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
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        if stage == "valid":
            self.best_val_iou = max(self.best_val_iou, dataset_iou)
        dataset_prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        dataset_prec = torch.tensor(0) if torch.isnan(dataset_prec) else dataset_prec
        dataset_rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        dataset_rec = torch.tensor(0) if torch.isnan(dataset_rec) else dataset_rec

        total_bsz = sum([x["bsz"] for x in outputs])
        loss = torch.tensor([x["loss"] * x["bsz"] for x in outputs]).mean() / total_bsz

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
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
        return torch.optim.AdamW(self.parameters(), lr=0.001)


class SegModelReplace(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, channels=list("rgbt"), run=None, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        std_, mean_ = [], []
        for channel in channels:
            if channel == "r":
                std_.append(params["std"][0])
                mean_.append(params["mean"][0])
            elif channel == "g":
                std_.append(params["std"][1])
                mean_.append(params["mean"][1])
            elif channel == "b":
                std_.append(params["std"][2])
                mean_.append(params["mean"][2])
            elif channel == "t":
                std_.append(0.0765)
                mean_.append(0.3445)
            else:
                std_.append(1)
                mean_.append(0)
        self.register_buffer("std", torch.tensor(std_).view(1, in_channels, 1, 1))
        self.register_buffer("mean", torch.tensor(mean_).view(1, in_channels, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

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

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

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
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
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
