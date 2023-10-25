import math
import torch
from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F
from models.encoders.S4DWrapper import S4DWrapper
from models.encoders.OrderedMemory import OrderedMemory
from models.encoders.RecurrentGRCX import RecurrentGRCX

class HGRC(nn.Module):
    def __init__(self, config):
        super(HGRC, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.model_chunk_size = config["model_chunk_size"]
        self.small_d = 64
        self.chunk_size = 30

        self.RNN = S4DWrapper(config)
        self.initial_transform = nn.Linear(self.hidden_dim, self.hidden_dim)
        if config and "rvnn_norm" in config:
            self.norm = config["rvnn_norm"]
        else:
            self.norm = "layer"

        if self.norm == "batch":
            self.NT = nn.BatchNorm1d(self.hidden_dim)
        elif self.norm == "skip":
            pass
        else:
            self.NT = nn.LayerNorm(self.hidden_dim)

        self.GRC = RecurrentGRCX(config)

    def normalize(self, state):
        if self.norm == "batch":
            return self.NT(state.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif self.norm == "skip":
            return state
        else:
            return self.NT(state)


    def forward(self, input, input_mask):

        sequence = self.RNN(input, input_mask)["sequence"]
        osequence = sequence.clone()
        oinput_mask = input_mask.clone()

        sequence = self.normalize(self.initial_transform(sequence))
        N, S, D = sequence.size()
        if not self.config["chunk_mode_inference"] and not self.training:
            self.chunk_size = S
        else:
            self.chunk_size = self.model_chunk_size

        while S > 1:
            N, S, D = sequence.size()
            if S >= (self.chunk_size + self.chunk_size // 2):
                if S % self.chunk_size != 0:
                    e = ((S // self.chunk_size) * self.chunk_size) + self.chunk_size - S
                    S = S + e
                    pad = T.zeros(N, e, D).float().to(sequence.device)
                    input_mask = T.cat([input_mask, T.zeros(N, e).float().to(sequence.device)], dim=-1)
                    sequence = T.cat([sequence, pad], dim=-2)
                    assert sequence.size() == (N, S, D)
                    assert input_mask.size() == (N, S)
                S1 = S // self.chunk_size
                chunk_size = self.chunk_size
            else:
                S1 = 1
                chunk_size = S
            sequence = sequence.view(N, S1, chunk_size, D)
            sequence = sequence.view(N * S1, chunk_size, D)

            input_mask = input_mask.view(N, S1, chunk_size)
            input_mask = input_mask.view(N * S1, chunk_size)

            N0 = N
            N, S, D = sequence.size()
            assert N == N0 * S1

            sequence = self.GRC(sequence, input_mask)["global_state"]
            assert sequence.size() == (N, D)
            sequence = sequence.view(N0, S1, D)
            input_mask = input_mask.view(N0, S1, chunk_size)[:, :, 0]
            S = S1
            N = N0

        assert sequence.size() == (N, 1, D)
        global_state = sequence.squeeze(1)

        return {"sequence": osequence,
                "global_state": global_state,
                "input_mask": oinput_mask,
                "aux_loss": None}
