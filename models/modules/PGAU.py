
import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.utils import RoFormerSinusoidalPositionalEmbedding, laplace_act


class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(T.ones(3, dim))
        self.beta = nn.Parameter(T.zeros(3, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        N, S, D = x.size()
        x = x.unsqueeze(-2).repeat(1, 1, 3, 1)
        gamma = self.gamma.view(1, 1, 3, D)
        beta = self.beta.view(1, 1, 3, D)

        out = x * gamma + beta
        Q = out[..., 0, :]
        Qp = out[..., 1, :]
        K = out[..., 2, :]
        assert Q.size() == (N, S, D)
        assert Qp.size() == (N, S, D)
        assert K.size() == (N, S, D)

        return Q, Qp, K


class PGAU(nn.Module):
    def __init__(self, config):
        super(PGAU, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.hidden_dim = self.hidden_size

        self.config = config
        self.d = min(128, self.hidden_dim)
        self.dropout = 0.1  # config["dropout"]
        self.scaling = math.sqrt(2 * self.d)

        self.rel_k_heights = 10
        self.embed_heights = nn.Parameter(T.randn(self.rel_k_heights, self.d))

        self.initial_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                               nn.LayerNorm(self.hidden_dim))

        self.U_transform = nn.Sequential(nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                                         nn.SiLU())

        self.V_transform = nn.Sequential(nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                                         nn.SiLU())

        self.Z_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.d),
                                         nn.SiLU())

        self.offsetscale = OffsetScale(dim=self.d)

        self.out_transform = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                           nn.Dropout(p=self.dropout))

        self.gate_transform = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                            nn.Sigmoid())

    """
    Forward Function
    """

    def masked_softmax(self, logits, mask, dim):
        if mask is None:
            return F.softmax(logits, dim=dim)

        logits = logits.masked_fill(~mask, -9999)
        logits = F.softmax(logits, dim=dim)
        return logits

    def create_pos(self, positions, q_pos, rel_k):
        N, S = positions.size()
        N2 = q_pos.size(0)
        S0 = (S - 1) // 2
        S1 = S - S0
        rel_positions = positions.unsqueeze(1) - positions.unsqueeze(2)
        assert rel_positions.size() == (N, S, S)
        rel_positions = rel_positions[:, :S0, S0:]
        rel_positions = T.clip(rel_positions, min=1, max=rel_k) -1
        pos_range = rel_k
        rel_positions = F.one_hot(rel_positions, num_classes=pos_range)
        assert rel_positions.size() == (N, S0, S1, pos_range)
        assert q_pos.size() == (N2, S0, pos_range)
        rel_energy = T.sum(rel_positions * q_pos.unsqueeze(-2), dim=-1)
        assert rel_energy.size() == (N2, S0, S1)
        return rel_energy

    # %%
    def forward(self, sequence,
                attention_mask,
                heights):
        N, S, D = sequence.size()
        assert attention_mask.size() == (N, S, S)
        S0 = (S - 1) // 2
        S1 = S - S0
        attention_mask = attention_mask[:, :S0, S0:]
        assert attention_mask.size() == (N, S0, S1)

        res = sequence.clone()
        sequence = self.initial_transform(sequence)

        U = self.U_transform(sequence[:, 0:S0, :])
        assert U.size() == (N, S0, 2 * D)
        Z = self.Z_transform(sequence)
        V = self.V_transform(sequence[:, S0:, :])

        assert Z.size() == (N, S, self.d)

        Q, Qp, K = self.offsetscale(Z)
        K = K[:, S0:, :]
        Q = Q[:, :S0, :]
        Qp = Qp[:, :S0, :]

        energy = T.matmul(Q, K.permute(0, 2, 1).contiguous())
        assert energy.size() == (N, S0, S1)

        h_range = self.rel_k_heights
        Q_h_embed = T.sum(Qp.unsqueeze(-2) * self.embed_heights.view(1, 1, h_range, self.d), dim=-1)
        assert Q_h_embed.size() == (N, S0, h_range)

        rel_energy = self.create_pos(positions=heights, q_pos=Q_h_embed, rel_k=self.rel_k_heights)

        attn = self.masked_softmax(logits=(energy + rel_energy) / self.scaling,
                                   mask=attention_mask.bool(),
                                   dim=-1)
        assert attn.size() == (N, S0, S1)

        V_ = T.matmul(attn, V)
        assert V_.size() == (N, S0, 2 * D)
        out = self.out_transform(U * V_)
        assert out.size() == (N, S0, D)

        gates = self.gate_transform(T.cat([out, res[:, :S0, :]], dim=-1))
        assert gates.size() == (N, S0, D)
        out = gates * out + (1 - gates) * res[:, :S0, :]
        assert out.size() == (N, S0, D)

        out = T.cat([out, res[:, S0:, :]], dim=1)
        assert out.size() == (N, S, D)

        return {"attended_values": out}