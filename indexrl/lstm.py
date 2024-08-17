# Reference: https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

import numpy as np
import torch
from torch import nn
from indexrl.utils import device


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, sequence_length):
        super(LSTMModel, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, vocab_size)

    def forward(self, x, prev_state):
        if x.shape[1] < self.sequence_length:
            x = torch.nn.functional.pad(
                x, (self.sequence_length - x.shape[1], 0), "constant", 0
            )
        elif x.shape[1] > self.sequence_length:
            x = x[:, -self.sequence_length :]
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self):
        return (
            torch.zeros(
                self.num_layers, self.sequence_length, self.lstm_size, device=device
            ),
            torch.zeros(
                self.num_layers, self.sequence_length, self.lstm_size, device=device
            ),
        )

    @torch.no_grad()
    def generate_single(self, state):
        h, c = self.init_state()
        y_pred, _ = self(state, (h, c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits.cpu(), dim=0)
        # word_index = np.random.choice(len(last_word_logits), p=p)
        return p
