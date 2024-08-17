from collections.abc import Iterator
import os
import pickle
import random
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from indexrl.mcts import MCTS
from indexrl.utils import device, log_all_seen_exps
from indexrl.environment import get_final_reward
from indexrl.gpt import GPT, GPTConfig
from indexrl.lstm import LSTMModel
from indexrl.configs.config_transfomer import (
    n_layer,
    n_head,
    n_embd,
    block_size,
    bias,
    dropout,
    weight_decay,
    learning_rate,
    beta1,
    beta2,
)


def explore(
    env,
    agent,
    image_dir: str,
    mask_dir: str,
    img_count: int = 5,
    logs_dir: str = None,
    seen_path: str = None,
    n_iters=None,
):
    data = []
    root_vals = []

    image_paths = glob(os.path.join(image_dir, "*.npy"))
    mask_paths = glob(os.path.join(mask_dir, "*.npy"))

    for i in range(img_count):
        idx = random.randrange(0, len(image_paths) - 1)
        print("Split:", i, ", Image:", idx)
        image_path = image_paths[idx]
        mask_path = mask_paths[idx]

        done = False
        count = 0
        root_vals_split = []

        image_env = env.copy()
        img_split = np.load(image_path)
        mask_split = np.load(mask_path)
        mcts = MCTS(image_env.copy(), agent, img_split, mask_split, True)
        state = image_env.reset(img_split, mask_split)

        if seen_path:
            image_env.load_seen(seen_path)

        while not done:
            count += 1

            probs = (
                mcts.run(max(n_iters // 2, n_iters - 50 * count))
                if n_iters
                else mcts.run()
            )
            action = random.choices(range(len(probs)), weights=probs, k=1)[0]
            state, reward, done = image_env.step(action)

            root_vals_split.append(round(mcts.root_node.value / mcts.root_node.n, 4))
            tree_str = mcts.root_node.display_tree(stdout=False)
            if logs_dir:
                with open(
                    os.path.join(logs_dir, f"tree_{count}.txt"), "w", encoding="utf-8"
                ) as fp:
                    fp.write(f"Expression: {mcts.env.cur_exp}\n{tree_str}")
            mcts = MCTS(image_env.copy(), agent, img_split, mask_split)

        if seen_path:
            image_env.save_seen(seen_path)

        if reward < 0:
            print(f"Expression: {mcts.env.cur_exp}, Reward < 0. Skipping.")
            continue
        final_reward = get_final_reward(
            image_env.cur_exp, image_dir, mask_dir, rew_type="train"
        )
        if final_reward > 0.01:
            data.append((state, final_reward))

        if logs_dir:
            with open(f"{logs_dir}/rewards.txt", "a") as fp:
                fp.write(f"{i} {final_reward} {image_env.cur_exp}\n")

        root_vals.append(root_vals_split)
        log_all_seen_exps(image_env.cur_exp, "train", final_reward)
        print(image_env.cur_exp, final_reward)

    if root_vals:
        with open(f"{logs_dir}/root_vals.txt", "a") as fp:
            fp.write(f"{np.concatenate(root_vals).mean()}\t{root_vals}\n")

    return data


def create_model(
    model_type: str,
    vocab_size: int = None,
    model_path: str = "",
    model_args: dict = None,
):
    if model_type == "gpt":
        if model_path:
            model = torch.load(model_path)
        else:
            if not model_args:
                model_args = dict(
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    block_size=block_size,
                    bias=bias,
                    vocab_size=vocab_size,
                    dropout=dropout,
                )
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            model.to(device)
        optimizer = model.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2), device
        )
    else:
        model = LSTMModel(vocab_size, 4)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    return model, optimizer


def save_model(model, model_path):
    torch.save(model, model_path)


class DynamicBuffer(list):
    def __init__(
        self,
        keep_fraction: float = 0.97,
        max_buffer_size: int = 8000,
        min_buffer_size: int = 20,
        max_reached: bool = False,
        cached_buffer_path: str = "",
    ):
        self.keep_fraction = keep_fraction
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_reached = max_reached
        self.buffer = []
        if cached_buffer_path:
            with open(cached_buffer_path, "rb") as fp:
                self.buffer = pickle.load(fp)

    def add_data(self, new_data):
        current_capacity = len(self.buffer)
        if not self.max_reached:
            if len(new_data) + current_capacity < self.max_buffer_size:
                self.buffer = new_data + self.buffer
            else:
                self.max_reached = True

        if self.max_reached:
            self.buffer = sorted(
                new_data + self.buffer, key=lambda x: x[1], reverse=True
            )[: max(int(current_capacity * self.keep_fraction), self.min_buffer_size)]

    def get_top_n(self, n: int):
        return sorted(self.buffer, key=lambda x: x[1], reverse=True)[:n]

    def __setitem__(self, index, item):
        self.buffer.__setitem__(index, item)

    def __len__(self) -> int:
        return len(self.buffer)

    def __str__(self) -> str:
        return str(self.buffer)

    def __iter__(self) -> Iterator:
        return self.buffer.__iter__()

    def insert(self, index, item):
        self.buffer.insert(index, item)

    def append(self, item):
        self.buffer.append(item)

    def extend(self, other):
        if isinstance(other, type(self)):
            self.buffer.extend(other)
        else:
            self.buffer.extend(item for item in other)


def train_iter(
    agent,
    optimizer,
    data_buffer: DynamicBuffer,
    n_epochs: int = 100,
):
    states = {}
    actions = {}
    rewards = {}
    reward_min, reward_max = 1, 0
    for state, reward in data_buffer:
        states[len(state) - 1] = states.get(len(state) - 1, []) + [state[:-1]]
        actions[len(state) - 1] = actions.get(len(state) - 1, []) + [state[1:]]
        rewards[len(state) - 1] = rewards.get(len(state) - 1, []) + [reward]
        reward_min = min(reward_min, reward)
        reward_max = max(reward_max, reward)

    buffer = []
    for key in states:
        state = torch.tensor(np.array(states[key]), dtype=torch.long)
        acts = torch.tensor(np.array(actions[key]), dtype=torch.long)
        rews = torch.tensor(np.array(rewards[key]))
        buffer.append((state, acts, rews))
    random.shuffle(buffer)

    if type(agent) == GPT:
        loss = train_gpt_iter(
            agent, optimizer, n_epochs, reward_min, reward_max, buffer
        )
    else:
        loss = train_lstm_iter(agent, optimizer, n_epochs, buffer)

    return agent, optimizer, loss


def train_gpt_iter(model, optimizer, n_epochs, reward_min, reward_max, buffer):
    losses = []
    for _ in tqdm(range(n_epochs), "Training..."):
        for state, acts, rews in buffer:
            state = state.to(device)
            acts = acts.to(device)
            _, loss = model(state, acts)
            if reward_max != reward_min:
                rew_scaled = (rews - reward_min) / (reward_max - reward_min)
                loss *= rew_scaled.mean()
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss = sum(losses) / len(losses)
    return loss


def train_lstm_iter(model, optimizer, n_epochs, buffer):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for _ in tqdm(range(n_epochs), "Training..."):
        state_h, state_c = model.init_state()

        for states, actions, rewards in buffer:
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(states, (state_h, state_c))
            if actions.shape[1] < 4:
                actions = torch.nn.functional.pad(
                    actions, (4 - actions.shape[1], 0), "constant", 0
                )
            elif actions.shape[1] > 4:
                actions = actions[:, -4:]
            loss = criterion(y_pred.transpose(1, 2), actions)
            # loss *= rewards
            losses.append(loss.item())

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

    loss = sum(losses) / len(losses)
    return loss


def train_agent(
    main_env,
    agent,
    optimizer,
    image_dir: str,
    mask_dir: str,
    logs_dir: str = None,
    models_dir: str = None,
    cache_dir: str = None,
    max_buffer_size: int = 500,
    min_buffer_size: int = 20,
    keep_fraction: float = 0.97,
    n_epochs: int = 100,
):
    seen_path = os.path.join(cache_dir, "seen.pkl") if cache_dir else ""
    data_buffer = DynamicBuffer(keep_fraction, max_buffer_size, min_buffer_size)
    i = 0
    while True:
        i += 1
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        data = explore(
            main_env.copy(),
            agent,
            image_dir,
            mask_dir,
            logs_dir,
            seen_path,
            n_iters=1000,
        )
        print(
            f"Data collection done. Collected {len(data)} examples. Buffer size = {len(data_buffer)}."
        )

        data_buffer.add_data(data)
        print(f"Buffer size new = {len(data_buffer)}.")

        agent, optimizer, loss = train_iter(agent, optimizer, data_buffer, n_epochs)

        i_str = str(i).rjust(3, "0")
        if models_dir:
            save_model(agent, f"{models_dir}/model_{i_str}_loss-{loss}.pt")
        if cache_dir:
            with open(f"{cache_dir}/data_buffer_{i_str}.pkl", "wb") as fp:
                pickle.dump(data_buffer, fp)
