n_folds = 5
# car_samples = [1054, 147, 1198, 1397, 183, 1449, 1136, 1199, 474, 828, 159, 775, 1123, 617, 1156, 1142, 267, 612, 654, 166]
# person_samples = [737, 92, 522, 1226, 728, 975, 958, 977, 922, 1005, 512, 899, 416, 1160, 894, 834, 458, 722, 1153, 190]
samples = [1061, 1135, 1083, 906, 179, 948, 689, 1151, 1188, 1136, 158, 149, 363, 31, 1232, 254, 214, 372, 1112, 158]

expressions = [
    [],
    ["(", "c1", "+", "(", "c1", "sq", "+", "c3", "sq", ")", "sq", "/", "c3", ")", "="],
    ["(", "c2", "sq", "-", "c3", "sqrt", "+", "c1", "sq", "/", "c3", "+", "c3", ")", "="],
]
# bike_expressions = [
#     [],
#     ["(", "c3", "/", "c0", "sqrt", "*", "c0", "*", "c1", "sq", ")", "sq", "="],
#     ["(", "c0", "-", "(", "c1", "/", "c0", "*", "c3", ")", "+", "c2", ")", "sq", "="],
# ]
# car_expressions = [
#     [],
#     ["(", "c2", "+", "(", "c2", "sq", "*", "c3", "sq", "*", "c2", ")", ")", "sqrt", "="],
#     ["(", "c2", "-", "(", "c3", "sqrt", "/", "c1", ")", "sqrt", ")", "="],
# ]
# person_expressions = [
#     [],
#     ["(", "(", "c2", "+", "c2", ")", "sqrt", ")", "-", "c0", "sq", "="],
#     ["(", "c2", "+", "(", "c3", "*", "c2", "+", "c1", ")", "sqrt", "-", "c1", ")", "="],
# ]
convergence_eps = 40
for optim in ("adamw",):
    for seed in range(10):
        for expression in expressions:
            for cur_fold in range(5):
                for learning_rate in (1e-3, 3e-4, 1e-4):
                    import segmentation_models_pytorch as smp
                    import numpy as np
                    import torch
                    import torch.nn as nn
                    from torch.utils.data import Dataset, DataLoader
                    from glob import glob
                    import os
                    import random
                    import torchvision.transforms.functional as TF
                    from tqdm import tqdm

                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    torch.use_deterministic_algorithms(True)

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
                        def __init__(self, data_dir: str, indices: list, exp: list, is_train: bool):
                            super().__init__()
                            self.img_list = sorted(glob(os.path.join(data_dir, "images", "*.npy")))
                            self.mask_list = sorted(glob(os.path.join(data_dir, "masks", "*.npy")))
                            self.indices = indices
                            self.is_train = is_train
                            self.exp = exp

                        def __len__(self):
                            return len(self.indices)

                        def __getitem__(self, idx):
                            index = samples[self.indices[idx]]
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
                            # img = cv2.resize(img, (512, 512))
                            # img = np.pad(img, ((1, 1), (1, 1), (0, 0)), 'edge')
                            img = TF.to_tensor(img)

                            mask = np.load(self.mask_list[index]).astype(float)
                            # mask = cv2.resize(mask, (512, 512))
                            # mask = np.pad(mask, ((1, 1), (1, 1), (0, 0)), 'edge')
                            mask = TF.to_tensor(mask).squeeze()

                            # Transforms
                            if self.is_train and random.random() > 0.5:
                                img = TF.hflip(img)
                                mask = TF.hflip(mask)
                            if self.is_train and random.random() > 0.5:
                                img = TF.vflip(img)
                                mask = TF.vflip(mask)
                            # if self.is_train and random.random() > 0.5:
                            #     angle = random.randint(0, 45)
                            #     img = TF.rotate(img, angle)
                            #     mask = TF.rotate(mask, angle)
                            return img.float(), mask.float()

                    def jaccard_index(pred, target, threshold=0.5):
                        pred_thresh = pred > threshold
                        return (
                            torch.logical_and(pred_thresh, target).sum() / torch.logical_or(pred_thresh, target).sum()
                        )

                    data_dir = "dataset/"
                    size = len(samples)
                    fold_size = size // n_folds

                    indices = list(range(size))

                    l_idx = fold_size * cur_fold
                    r_idx = fold_size * (cur_fold + 1)

                    train_indices = indices[:l_idx] + (indices[r_idx:] if r_idx < size else [])
                    val_indices = indices[l_idx:r_idx]

                    train_set = CustomDataset(data_dir, train_indices, expression, True)
                    val_set = CustomDataset(data_dir, val_indices, expression, False)

                    batch_size = 8
                    train_loader = DataLoader(train_set, batch_size, True, num_workers=0)
                    val_loader = DataLoader(val_set, batch_size, False, num_workers=0)

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_name = "unet++"

                    if model_name == "unet":
                        model = smp.Unet(
                            encoder_name="resnet50", in_channels=val_set[0][0].shape[0], classes=1, activation=None
                        )
                    elif model_name == "deeplabv3":
                        model = smp.DeepLabV3(
                            encoder_name="resnet50", in_channels=val_set[0][0].shape[0], classes=1, activation=None
                        )
                    elif model_name == "unet++":
                        model = smp.UnetPlusPlus(
                            encoder_name="resnet50", in_channels=val_set[0][0].shape[0], classes=1, activation=None
                        )
                    elif model_name == "deeplabv3+":
                        model = smp.DeepLabV3Plus(
                            encoder_name="resnet50", in_channels=val_set[0][0].shape[0], classes=1, activation=None
                        )

                    model.to(device)
                    model = model.float()

                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                    criterion = nn.BCEWithLogitsLoss()

                    import neptune

                    run = neptune.init_run(
                        project="dilithjay/FYP",
                        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MjVlMDAxOS0wYjQyLTRhMDEtOWNlOS1hZWEwMGE5MzQ4NTUifQ==",
                    )
                    params = {
                        "converge_eps": convergence_eps,
                        "batch_size": batch_size,
                        "optimizer": optim,
                        "lr": learning_rate,
                        "without_index": without_index,
                        "seed": seed,
                        "n_folds": n_folds,
                        "fold": cur_fold,
                        "arch": model_name,
                        "dataset": "ir-sample-bike",
                        "expression": "" if without_index else str(expression),
                    }
                    run["parameters"] = params

                    best_val_loss = 1
                    best_val_iou = 0
                    best_ep = 0
                    best_params = {}
                    epoch = 0
                    while True:
                        print("-----------\nEpoch:", epoch)
                        train_loss = val_loss = val_iou = 0
                        for img, mask in tqdm(train_loader, "Training:"):
                            img = img.to(device).squeeze()
                            mask = mask.to(device).squeeze()

                            logits = model(img).squeeze()
                            # print(logits.shape, logits.max(), logits.min(), mask.shape, mask.max(), mask.min())
                            loss = criterion(logits, mask)
                            # print(loss.item())
                            train_loss += loss.item()

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        with torch.no_grad():
                            for img, mask in tqdm(val_loader, "Validation:"):
                                img = img.to(device).squeeze()
                                mask = mask.to(device).squeeze()
                                logits = model(img).squeeze()
                                loss = criterion(logits, mask)
                                # print(loss.item())
                                val_loss += loss.item()
                                val_iou += jaccard_index(logits, mask).item()

                        ep_train_loss = train_loss / len(train_loader)
                        ep_val_loss = val_loss / len(val_loader)
                        ep_val_iou = val_iou / len(val_loader)

                        run["train/loss"].append(ep_train_loss)
                        run["val/loss"].append(ep_val_loss)
                        run["val/iou"].append(ep_val_iou)

                        print(f"Train loss: {ep_train_loss}, Val loss: {ep_val_loss}, Val IoU: {ep_val_iou}\n")

                        if epoch >= 20:
                            if ep_val_loss < best_val_loss:
                                best_val_loss = ep_val_loss
                                best_ep = epoch
                                best_params = model.state_dict()
                                # torch.save(model, 'models/cloud_model_2.pt')
                            elif ep_val_iou > best_val_iou:
                                best_val_iou = ep_val_iou
                                best_ep = epoch
                                best_params = model.state_dict()
                                run["val/best_iou"] = best_val_iou
                            elif epoch - best_ep >= convergence_eps:
                                break

                        epoch += 1

                    run.stop()
