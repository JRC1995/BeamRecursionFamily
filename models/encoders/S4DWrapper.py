import math
import torch
from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F
from models.modules.S4 import S4Block
from models.utils import reverse


class S4DWrapper(nn.Module):
    def __init__(self, config):
        super(S4DWrapper, self).__init__()
        self.config = config
        self.hidden_dim = config["hidden_size"]
        self.norm = config["norm"]
        self.dropout = config["s4_dropout"]
        self.prenorm = config["prenorm"]

        self.S4DBlock = S4Block(d_model=2 * self.hidden_dim,
                                bidirectional=False,
                                mode="diag",
                                init="diag-inv",
                                disc="bilinear",
                                final_act=None,
                                transposed=False,
                                freeze_B=False,
                                dropout=self.dropout,
                                tie_dropout=True,
                                lr=min(1e-3, config["lr"]))

        self.GLU = nn.Sequential(nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
                                 nn.GLU())
        if self.norm == "batch":
            self.NT = nn.BatchNorm1d(self.hidden_dim)
        elif self.norm == "skip":
            pass
        else:
            self.NT = nn.LayerNorm(self.hidden_dim)

    def normalize(self, state):
        if self.norm == "batch":
            return self.NT(state.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif self.norm == "skip":
            return state
        else:
            return self.NT(state)

    def forward(self, input, input_mask):
        N, S, D = input.size()
        input_mask = input_mask.view(N, S, 1)

        state = input
        res = input.clone()

        if self.prenorm:
            state = self.normalize(state)

        lengths = T.sum(input_mask, dim=1).view(N)

        fstate = state.clone()
        bstate = state.clone()

        count_zeros = T.sum(1 - input_mask, dim=1).view(N).long()
        count_zeros_end = count_zeros + S

        reverse_bstate = reverse(bstate, count_zeros, count_zeros_end)
        state = T.cat([fstate, reverse_bstate], dim=-1)

        state, _ = self.S4DBlock(x=state, lengths=lengths)
        assert state.size() == (N, S, 2 * D)

        fstate = state[..., 0:D]
        reverse_bstate = state[..., D:]
        bstate = reverse(reverse_bstate, count_zeros, count_zeros_end)
        state = T.cat([fstate, bstate], dim=-1)

        assert state.size() == (N, S, 2 * D)

        state = self.GLU(state)
        assert state.size() == (N, S, D)

        state = state + res
        if not self.prenorm:
            state = self.normalize(state)
        assert state.size() == (N, S, D)

        global_state = None
        aux_loss = None

        return {"sequence": state,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": aux_loss}