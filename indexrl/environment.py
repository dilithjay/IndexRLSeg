from glob import glob
import os
import random
import gym
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.metrics import roc_curve, auc

from indexrl.expression_handler import check_unitless_validity, eval_expression
from indexrl.utils import standardize
from train import train_model


class IndexRLEnv(gym.Env):
    def __init__(
        self,
        discrete_actions: list,
        max_exp_len: int = 100,
        ohe: bool = False,
        masked_actions: list = None,
        unitless: bool = False,
    ):
        super(IndexRLEnv, self).__init__()
        self.actions = discrete_actions
        self.image = None
        self.mask = None
        self.cur_exp = []
        self.parentheses_level = 0
        self.max_exp_len = max_exp_len
        self.ohe = ohe
        self.masked_actions = masked_actions
        self.unitless = unitless

        self.best_reward = 0
        self.best_exp = []
        self.seen = set()

    def load_seen(self, seen_path):
        if os.path.exists(seen_path):
            with open(seen_path, "rb") as fp:
                self.seen = pickle.load(fp)

    def save_seen(self, seen_path):
        print("Seen length:", len(self.seen))
        with open(seen_path, "wb") as fp:
            pickle.dump(self.seen, fp)

    def get_cur_state(self):
        if self.ohe:
            cur_exp_indices = [self.actions.index(act) for act in self.cur_exp] + [
                0
            ] * (self.max_exp_len - len(self.cur_exp))
            enc_state = np.zeros((self.max_exp_len, len(self.actions)))
            enc_state[np.arange(self.max_exp_len), cur_exp_indices] = 1
            return enc_state.flatten()
        else:
            cur_exp_indices = [self.actions.index(act) for act in self.cur_exp]
            return np.array(cur_exp_indices)

    def step(self, action_idx: int) -> tuple:
        """Take a step in the environment with the specified action.

        Args:
            action_idx (int): discrete action index

        Returns:
            np.ndarray: current state
            float:      reward
            bool:       done
        """
        done = self.take_action(action_idx)

        if len(self.cur_exp) >= self.max_exp_len:
            done = True

        if done:
            exp_s = str(self.cur_exp)
            if exp_s in self.seen:
                return self.get_cur_state(), -1, True
            self.seen.add(exp_s)

        reward = self.get_reward(done)

        return self.get_cur_state(), reward, done

    def reset(self, image: np.ndarray = None, mask: np.ndarray = None) -> np.ndarray:
        if image is not None and mask is not None:
            self.image = image
            self.mask = mask
        self.cur_exp = []
        self.parentheses_level = 0

        return self.get_cur_state()

    def render(self):
        print(self.cur_exp)

    def get_reward(self, done: bool) -> float:
        if not done:
            return 0

        if len(self.cur_exp) < 3:
            return -1

        unitless = check_unitless_validity(self.cur_exp)
        result = eval_expression(self.cur_exp, self.image.squeeze())
        if result is False:
            return -1

        if len(self.mask) > 2:
            reward = get_correlation(result, self.mask > 0)
            # reward = (np.abs(result - self.mask) < 0.1).sum() / len(self.mask)
            self.best_reward = max(self.best_reward, reward)
            self.best_exp = self.cur_exp
            return reward
        else:  # Pretraining stage
            # Motivate longer expressions
            if len(self.cur_exp) < self.max_exp_len // 3 or "(" not in self.cur_exp:
                return -1
            return (
                0.5
                + 0.05 * len(self.cur_exp)
                + 0.01 * self.cur_exp.count(")")
                + 1 * unitless
            )

    def take_action(self, action_idx: int) -> bool:
        action = self.actions[action_idx]
        if action == "(":
            self.parentheses_level += 1
        elif action == ")":
            self.parentheses_level -= 1
        self.cur_exp.append(action)

        return action == "="

    def get_valid_actions(self):
        if len(self.cur_exp) == self.max_exp_len - 1:
            return {self.actions.index("=")}

        # Include all the channels and opening brackets in action set 1
        acts_1 = []
        for i, act in enumerate(self.actions):
            if act[0] == "c" or act in ("("):
                acts_1.append(i)

        # Remove actions specified as masked actions from acts_1
        if self.masked_actions:
            for act in self.masked_actions:
                idx = self.actions.index(act)
                if idx in acts_1:
                    acts_1.remove(idx)

        # Include the inverse of the action set 1 in action set 2
        acts_2 = list(set(range(len(self.actions))) - set(acts_1))

        # Allow action set 1 when just starting the episode
        if not self.cur_exp:
            return acts_1

        last_act = self.cur_exp[-1]

        # Disallow any other actions if episode ended with "="
        if last_act == "=":
            return []

        # If last action was one of the following, allow selecting channels or an open parenthesis
        if last_act in list("(+-*/"):
            return acts_1

        # Disallow consecutive squares, square roots, or combinations of them.
        if last_act == "sq" or last_act == "sqrt":
            acts_2.remove(self.actions.index("sq"))
            acts_2.remove(self.actions.index("sqrt"))

        # Remove actions specified as masked actions from acts_2
        if self.masked_actions:
            for act in self.masked_actions:
                idx = self.actions.index(act)
                if idx in acts_2:
                    acts_2.remove(idx)

        # Disallow closing the brackets after just one character
        if len(self.cur_exp) > 1 and self.cur_exp[-2] == "(":
            acts_2.remove(self.actions.index(")"))

        # Disallow closing paranthesis if there are no open paranthesis
        if self.parentheses_level <= 0:
            acts_2.remove(self.actions.index(")"))
        else:
            acts_2.remove(self.actions.index("="))

        return acts_2

    def get_invalid_actions(self):
        return list(set(range(len(self.actions))) - set(self.get_valid_actions()))

    def copy(self):
        return deepcopy(self)

    def state_to_expression(self, state):
        return list(map(lambda x: self.actions[x], state))


def get_final_reward(
    exp: list, image_dir: str, mask_dir: str, rew_type="-", k=None, indices=None
) -> float:
    image_paths = sorted(glob(os.path.join(image_dir, "*.npy")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))
    assert len(image_paths) == len(mask_paths)

    if indices == None:
        if k is None:
            indices = range(len(image_paths))
        else:
            random.seed(0)
            indices = random.choices(range(len(image_paths)), k=k)

    unitless = check_unitless_validity(exp)
    if unitless is False:
        return -1

    if rew_type == "train":
        return get_training_reward(exp, image_paths, mask_paths, indices)

    tot_reward = 0
    for i in indices:
        image_path = image_paths[i]
        mask_path = mask_paths[i]
        image = np.load(image_path)
        mask = np.load(mask_path)

        result = eval_expression(exp, image.squeeze())
        if (
            result is False
            or np.abs(result).sum() < 1
            or (result.max() - result.min()) < 0.01
        ):
            tot_reward -= 1
        elif len(mask) > 2:
            norm_result = standardize(result, 3, (0, 1))
            if rew_type == "corr":
                tot_reward += abs(get_correlation(norm_result, mask > 0))
            elif rew_type == "f1":
                tot_reward += get_f1_score(norm_result, mask > 0)
            elif rew_type == "auc":
                tot_reward += get_auc_score(norm_result, mask > 0)
            elif rew_type == "sim":
                tot_reward += get_similarity(norm_result, mask > 0)
            elif rew_type == "iou":
                tot_reward += get_jaccard(norm_result, mask > 0)
            elif rew_type == "usl":
                tot_reward += get_unsupervised_reward(image, norm_result)
            else:  # AUC F1
                tot_reward += min(
                    get_auc_f1(standardize(result), mask > 0),
                    get_auc_f1(1 - standardize(result), mask > 0),
                )
        else:  # Pretraining stage
            tot_reward += 0.5 + 0.02 * len(exp) + 0.2 * exp.count(")") + 1 * unitless

    return tot_reward / len(indices)


def get_precision_recall(result: np.ndarray, mask: np.ndarray, threshold: float = 0.5):
    pred_mask = result > threshold
    tp = np.logical_and(pred_mask, mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), mask).sum()

    return tp / (tp + fp + 0.0001), tp / (tp + fn + 0.0001)


def get_auc_f1(result: np.ndarray, mask: np.ndarray):
    tot_score = 0
    for thresh in np.arange(-1, 1, 0.1):
        # pr, rec = get_precision_recall(result, mask, thresh)
        tot_score += get_f1_score(result > thresh, mask)

    return tot_score / 20


def get_auc_score(result: np.ndarray, mask: np.ndarray):
    fpr, tpr, _ = roc_curve(mask.ravel(), result.ravel())
    return auc(fpr, tpr)


def get_f1_score(result: np.ndarray, mask: np.ndarray):
    pr, rec = get_precision_recall(result.ravel(), mask.ravel())
    return 2 * pr * rec / (pr + rec + 0.0001)


def get_correlation(result: np.ndarray, mask: np.ndarray):
    corr = np.abs(pd.Series(result.flatten()).corr(pd.Series(mask.flatten())))
    return corr


def get_similarity(result: np.ndarray, mask: np.ndarray):
    return np.dot(result.flatten(), mask.flatten()) / (
        np.linalg.norm(result) * np.linalg.norm(mask)
    )


def get_jaccard(pred, target, threshold=0.5):
    pred_thresh = pred > threshold
    return (
        np.logical_and(pred_thresh, target).sum()
        / np.logical_or(pred_thresh, target).sum()
    )


def get_unsupervised_reward(image: np.ndarray, index: np.ndarray):
    corr = 0
    for channel in image:
        corr += 1 - np.abs(np.corrcoef(channel, index)[0, 1])
    return corr / len(image)


def get_training_reward(exp: list, image_paths: str, mask_paths: str, indices=None):
    out = train_model(
        "unet",
        os.environ["DATASET_NAME"],
        [exp],
        1e-3,
        0,
        None,
        min_epochs=3,
        max_epochs=10,
        deterministic=False,
    )
    return out[-1]["valid_dataset_iou"]
