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


class GAU(nn.Module):
    def __init__(self, config):
        super(GAU, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.hidden_dim = self.hidden_size

        self.config = config
        self.d = min(128, self.hidden_dim)
        self.dropout = 0.1  # config["dropout"]
        self.scaling = math.sqrt(2 * self.d)

        self.rel_k = 5
        self.embed_positions = nn.Parameter(T.randn(2 * self.rel_k + 1, self.d))

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

        self.pos_cache = None

    """
    Forward Function
    """

    def masked_softmax(self, logits, mask, dim):
        if mask is None:
            return F.softmax(logits, dim=dim)

        logits = logits.masked_fill(~mask, float("-inf"))
        logits = F.softmax(logits, dim=dim)
        return logits

    def create_pos(self, positions, q_pos, rel_k):
        N, S = positions.size()
        N2 = q_pos.size(0)
        rel_positions = positions.unsqueeze(1) - positions.unsqueeze(2)
        assert rel_positions.size() == (N, S, S)
        rel_positions = T.clip(rel_positions, min=-rel_k, max=rel_k) + rel_k
        pos_range = 2 * rel_k + 1
        rel_positions = F.one_hot(rel_positions, num_classes=pos_range)
        assert rel_positions.size() == (N, S, S, pos_range)
        assert q_pos.size() == (N2, S, pos_range)
        rel_energy = T.sum(rel_positions * q_pos.unsqueeze(-2), dim=-1)
        assert rel_energy.size() == (N2, S, S)
        return rel_energy

    # %%
    def forward(self, sequence,
                attention_mask,
                positions):
        N, S, D = sequence.size()
        assert attention_mask.size() == (N, S, S)
        assert positions.size() == (N, S)

        res = sequence.clone()
        sequence = self.initial_transform(sequence)

        U = self.U_transform(sequence)
        assert U.size() == (N, S, 2 * D)
        Z = self.Z_transform(sequence)
        V = self.V_transform(sequence)

        assert Z.size() == (N, S, self.d)

        Q, Qp, K = self.offsetscale(Z)
        energy = T.matmul(Q, K.permute(0, 2, 1).contiguous())
        assert energy.size() == (N, S, S)

        pos_range = 2 * self.rel_k + 1
        Q_pos_embed = T.sum(Qp.unsqueeze(-2) * self.embed_positions.view(1, 1, pos_range, self.d), dim=-1)
        assert Q_pos_embed.size() == (N, S, pos_range)
        rel_energy = self.create_pos(positions=positions, q_pos=Q_pos_embed, rel_k=self.rel_k)

        attn = self.masked_softmax(logits=(energy + rel_energy) / self.scaling,
                                   mask=attention_mask.bool(),
                                   dim=-1)
        assert attn.size() == (N, S, S)

        V_ = T.matmul(attn, V)
        assert V_.size() == (N, S, 2 * D)

        out = self.out_transform(U * V_)
        assert out.size() == (N, S, D)

        gates = self.gate_transform(T.cat([out, res], dim=-1))
        assert gates.size() == (N, S, D)

        out = gates * out + (1 - gates) * res

        assert out.size() == (N, S, D)

        return {"attended_values": out}
