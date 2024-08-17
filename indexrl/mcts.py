import math
import numpy as np
import torch

from tqdm import tqdm
from treelib import Tree

from indexrl.utils import device, log_all_seen_exps


class Node:
    def __init__(self, state=None, name="", index=0):
        self.value = 0
        self.n = 0
        self.state = state
        self.name = name
        self.index = index
        self.children = []

    def get_max_ucb1_child(self):
        if not self.children:
            return None

        max_ucb1 = float("-inf")
        max_nodes = []

        for child in self.children:
            ucb1 = self.get_child_ucb1(child)

            if ucb1 == max_ucb1:
                max_nodes.append(child)
            elif ucb1 > max_ucb1:
                max_ucb1 = ucb1
                max_nodes = [child]

        return np.random.choice(max_nodes)

    def get_child_ucb1(self, child):
        if child.n == 0:
            return float("inf")
        return child.value / child.n + 2 * math.sqrt(math.log(self.n, math.e) / child.n)

    def display_tree(self, stdout=True):
        tree = Tree()

        tree.create_node(f"Root => value: {self.value}, n: {self.n}", "root")
        stack = [(self, "root")]
        count = 0

        while stack:
            cur_node, node_name = stack.pop()
            for child in cur_node.children:
                tree.create_node(
                    f"{child.name} => value: {child.value}, n: {child.n}",
                    f"node_{count}",
                    parent=node_name,
                )
                stack.append((child, f"node_{count}"))
                count += 1

        return tree.show(stdout=stdout)


class MCTS:
    def __init__(self, env, agent, image, mask, reset=False):
        self.env = env
        self.agent = agent

        if reset:
            start_state = self.env.reset(image, mask)
        else:
            start_state = self.env.get_cur_state()
        self.start_env = self.env.copy()
        self.root_node = Node(start_state)

        for act in self.env.get_valid_actions():
            env_copy = self.env.copy()
            new_state, _, _ = env_copy.step(act)
            new_node = Node(new_state, self.env.actions[act], act)
            self.root_node.children.append(new_node)

    def run(self, n_iter=200):
        for _ in tqdm(range(n_iter)):
            value, node_path = self.traverse()
            self.backpropagate(node_path, value)
            self.env = self.start_env.copy()

        # self.root_node.display_tree()
        vals = [float("-inf")] * len(self.env.actions)
        for child in self.root_node.children:
            vals[child.index] = (child.value / child.n) if child.n else 0
        exp_vals = np.exp((np.array(vals) + 1) * 3)  # Scale to amplify difference
        return exp_vals / sum(exp_vals)

    def traverse(self):
        cur_node = self.root_node
        node_path = [cur_node]
        while cur_node.children:
            cur_node = cur_node.get_max_ucb1_child()
            self.env.step(cur_node.index)
            node_path.append(cur_node)

        if cur_node.n:
            for act in self.env.get_valid_actions():
                env_copy = self.env.copy()
                new_state, _, _ = env_copy.step(act)
                new_node = Node(new_state, self.env.actions[act], act)
                cur_node.children.append(new_node)
            while cur_node.children:
                cur_node = cur_node.get_max_ucb1_child()
                node_path.append(cur_node)
        # print("Node path:", [node.name for node in node_path])

        return self.rollout(cur_node), node_path

    def backpropagate(self, node_path: list, last_value: float):
        for node in node_path[::-1]:
            node.value += last_value
            node.n += 1

    def rollout(self, state_node: Node) -> float:
        if state_node.name == "=":
            return self.env.get_reward(done=True)

        tot_reward = 0
        cur_state = state_node.state

        step = 0
        while True:
            # print(step, tot_reward, self.env.cur_exp)
            step += 1
            state = torch.tensor(np.expand_dims(cur_state, 0)).int().to(device)
            with torch.no_grad():
                probs = self.agent.generate_single(state).squeeze()

            invalid_acts = self.env.get_invalid_actions()
            probs[invalid_acts] = float("-inf")
            probs = torch.softmax(probs, 0).cpu().numpy().astype("float64")
            probs /= probs.sum()
            action_idx = np.random.choice(len(probs), 1, p=probs)[0]

            cur_state, reward, done = self.env.step(action_idx)
            tot_reward += reward

            if done:
                log_all_seen_exps(cur_state, "corr", tot_reward, self.env.actions)
                break

        return tot_reward
