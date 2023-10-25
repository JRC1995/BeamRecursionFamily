import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from typing import Optional, Tuple, Union
import math


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, positions) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        return super().forward(positions)


def laplace_act(x):
    mu = math.sqrt(0.5)
    std = math.sqrt((4 * math.pi) ** -1)
    return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5


def sum_normalize(logits, dim=-1):
    eps = 1e-8
    return logits / T.sum(logits + eps, keepdim=True, dim=dim)


def masked_softmax(logits, mask=None, dim=-1):
    eps = 1e-20
    probs = F.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)
    return probs


def stochastic_topk(logits, select_k=1, mask=None, training=True):
    N, S = logits.size()

    soft_scores = masked_softmax(logits=logits, mask=mask)
    if training:
        eps = 1e-20
        u = logits.data.new(*logits.size()).uniform_()
        gumbel_noise = -T.log(-T.log(u + eps) + eps)
        perturbed_soft_scores = masked_softmax(logits=logits + gumbel_noise, mask=mask)
    else:
        perturbed_soft_scores = soft_scores
    topk_idx = T.topk(perturbed_soft_scores, dim=-1, k=select_k, sorted=True)[1]
    assert topk_idx.size() == (N, select_k)
    select_mask = F.one_hot(topk_idx, num_classes=S).float()
    assert select_mask.size() == (N, select_k, S)

    return select_mask, soft_scores


def reverse(state, count_zeros, count_zeros_end):
    with T.no_grad():
        N, S, D = state.size()
        reverse_state = T.flip(state, dims=[1])
        reverse_state = T.cat([reverse_state,
                               T.zeros(N, S, D).float().to(state.device)], dim=1)
        new_batch_stack = []
        for i in range(N):
            start_id = count_zeros[i]
            end_id = count_zeros_end[i]
            new_batch_stack.append(reverse_state[i, start_id:end_id, :])
        reverse_state = T.stack(new_batch_stack, dim=0)
        return reverse_state


def st_gumbel_softmax(logits, temperature=1.0, mask=None, training=False):
    eps = 1e-20
    N, S = logits.size()

    soft_y = masked_softmax(logits=logits / temperature, mask=mask)
    if training:
        u = logits.data.new(*logits.size()).uniform_()
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    else:
        gumbel_noise = 0
    y = logits + gumbel_noise

    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(dim=-1)[1]
    y_hard = F.one_hot(y_argmax, num_classes=y.size(-1)).float()

    assert y.size() == (N, S)

    y = (y_hard - y).detach() + y
    return y, soft_y
