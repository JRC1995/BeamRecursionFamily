import math
import torch
from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F
from models.modules.GRC import GRC
from models.utils.utils import stochastic_topk


class EBT_GRC_noslice(nn.Module):
    def __init__(self, config):
        super(EBT_GRC_noslice, self).__init__()
        self.config = config
        self.word_dim = config["input_size"]
        self.hidden_dim = config["hidden_size"]
        self.beam_size = config["beam_size"]

        self.small_d = self.hidden_dim

        self.initial_transform = nn.Sequential(nn.Linear(self.word_dim, self.hidden_dim),
                                               nn.LayerNorm(self.hidden_dim))

        self.treecell_layer = GRC(hidden_size=self.hidden_dim,
                                  cell_hidden_size=4 * self.hidden_dim,
                                  dropout=config["dropout"])

        self.decision_module = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.small_d),
                                             nn.GELU(),
                                             nn.Linear(self.small_d, 1))

        #self.decision_module = nn.Linear(2 * self.hidden_dim, 1)

    @staticmethod
    def update_state(old_content_state, new_content_state, done_mask):
        N = old_content_state.size(0)
        done_mask = done_mask.view(N, 1, 1, 1)
        content_state = done_mask * new_content_state + (1 - done_mask) * old_content_state[..., :-1, :]
        return content_state

    def select_composition(self, old_content_state, mask, accu_scores, beam_mask):

        N, B, S, D = old_content_state.size()
        S = S - 1
        assert accu_scores.size() == (N, B)
        assert mask.size() == (N, S)
        assert beam_mask.size() == (N, B)

        l = old_content_state[:, :, :-1, 0:self.small_d]
        r = old_content_state[:, :, 1:, 0:self.small_d]
        assert l.size() == (N, B, S, self.small_d)
        assert r.size() == (N, B, S, self.small_d)
        cat_state = T.cat([l, r], dim=-1)

        comp_weights = self.decision_module(cat_state).squeeze(-1)

        topk = min(S, self.beam_size)  # beam_size
        select_mask, soft_scores = stochastic_topk(logits=comp_weights.view(N * B, S),
                                                   mask=mask.view(N, 1, S).repeat(1, B, 1).view(N * B, S),
                                                   select_k=topk,
                                                   training=self.training)

        soft_scores = soft_scores.view(N, B, 1, S)
        assert select_mask.size() == (N * B, topk, S)
        select_mask = select_mask.view(N, B, topk, S)

        new_scores = T.log(T.sum(select_mask * soft_scores, dim=-1) + 1e-20)
        assert new_scores.size() == (N, B, topk)

        done_mask = 1 - mask[:, 0].view(N, 1, 1).repeat(1, B, 1)
        if topk == 1:
            done_topk = T.ones(N, B, topk).float().to(mask.device)
        else:
            done_topk = T.cat([T.ones(N, B, 1).float().to(mask.device),
                               T.zeros(N, B, topk - 1).float().to(mask.device)], dim=-1)
        assert done_topk.size() == (N, B, topk)

        not_done_topk = T.ones(N, B, topk).float().to(mask.device)
        new_beam_mask = done_mask * done_topk + (1 - done_mask) * not_done_topk
        beam_mask = beam_mask.unsqueeze(-1) * new_beam_mask

        assert beam_mask.size() == (N, B, topk)
        beam_mask = beam_mask.view(N, B * topk)

        accu_scores = accu_scores.view(N, B, 1) + new_scores
        accu_scores = accu_scores.view(N, B * topk)
        # accu_scores = T.clip(accu_scores, min=-999999)

        select_mask = select_mask.view(N, B * topk, S)

        old_content_state = old_content_state.unsqueeze(2).repeat(1, 1, topk, 1, 1)
        assert old_content_state.size() == (N, B, topk, S + 1, D)
        old_content_state = old_content_state.view(N, B * topk, S + 1, D)

        if (B * topk) > self.beam_size:
            B2 = self.beam_size
            assert accu_scores.size() == beam_mask.size()
            beam_select_mask, _ = stochastic_topk(logits=accu_scores,
                                                  mask=beam_mask,
                                                  select_k=B2,
                                                  training=self.training)
            assert beam_select_mask.size() == (N, B2, B * topk)

            old_content_state = T.matmul(beam_select_mask, old_content_state.view(N, B * topk, -1))
            old_content_state = old_content_state.view(N, B2, S + 1, D)

            select_mask = T.matmul(beam_select_mask, select_mask)
            assert select_mask.size() == (N, B2, S)

            accu_scores = T.matmul(beam_select_mask, accu_scores.unsqueeze(-1)).squeeze(-1)
            assert accu_scores.size() == (N, B2)

            beam_mask = T.matmul(beam_select_mask, beam_mask.unsqueeze(-1)).squeeze(-1)
            assert beam_mask.size() == (N, B2)
        else:
            B2 = B * topk

        assert old_content_state.size() == (N, B2, S + 1, D)
        assert select_mask.size() == (N, B2, S)
        l = old_content_state[:, :, :-1, :]
        r = old_content_state[:, :, 1:, :]
        l = T.matmul(select_mask.unsqueeze(-2), l)
        r = T.matmul(select_mask.unsqueeze(-2), r)
        assert l.size() == (N, B2, 1, D)
        assert r.size() == (N, B2, 1, D)
        l = l.view(N * B2, D)
        r = r.view(N * B2, D)
        new_content_state = self.treecell_layer(left=l, right=r)
        assert new_content_state.size() == (N * B2, D)
        new_content_state = new_content_state.view(N, B2, 1, D)

        select_mask_expand = select_mask.unsqueeze(-1)
        select_mask_cumsum = select_mask.cumsum(-1)

        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(-1)

        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(-1)

        olc, orc = old_content_state[..., :-1, :], old_content_state[..., 1:, :]

        assert select_mask_expand.size() == (N, B2, S, 1)
        assert left_mask_expand.size() == (N, B2, S, 1)
        assert right_mask_expand.size() == (N, B2, S, 1)
        assert new_content_state.size() == (N, B2, 1, D)
        assert olc.size() == (N, B2, S, D)
        assert orc.size() == (N, B2, S, D)

        new_content_state = (select_mask_expand * new_content_state
                             + left_mask_expand * olc
                             + right_mask_expand * orc)

        return old_content_state, new_content_state, accu_scores, beam_mask

    def forward(self, input, input_mask):

        max_depth = input.size(1)
        length_mask = input_mask
        content_state = self.initial_transform(input)

        N, S, D = content_state.size()
        B = 1
        content_state = content_state.unsqueeze(1)
        assert content_state.size() == (N, B, S, D)

        accu_scores = T.zeros(N, B).float().to(content_state.device)
        beam_mask = T.ones(N, B).float().to(content_state.device)

        for i in range(max_depth - 1):
            S = content_state.size(-2)
            B = content_state.size(1)

            if i < max_depth - 2:
                old_content_state, new_content_state, \
                accu_scores, beam_mask = self.select_composition(old_content_state=content_state,
                                                                 mask=length_mask[:, i + 1:],
                                                                 accu_scores=accu_scores,
                                                                 beam_mask=beam_mask)
            else:
                old_content_state = content_state.clone()
                l = content_state[:, :, :-1, :]
                r = content_state[:, :, 1:, :]
                assert l.size() == (N, B, 1, D)
                assert r.size() == (N, B, 1, D)
                l = l.view(N * B, D)
                r = r.view(N * B, D)
                new_content_state = self.treecell_layer(left=l, right=r)
                new_content_state = new_content_state.view(N, B, 1, D)

            done_mask = length_mask[:, i + 1]
            content_state = self.update_state(old_content_state=old_content_state,
                                              new_content_state=new_content_state,
                                              done_mask=done_mask)
        h = content_state
        sequence = input
        input_mask = input_mask.unsqueeze(-1)
        aux_loss = None

        N, B, S, D = h.size()
        assert S == 1
        h = h.squeeze(-2)
        assert h.size() == (N, B, D)
        assert accu_scores.size() == (N, B)
        assert beam_mask.size() == (N, B)

        scores = beam_mask * accu_scores + (1 - beam_mask) * -999999
        normed_scores = F.softmax(scores, dim=-1)

        global_state = T.sum(normed_scores.unsqueeze(-1) * h, dim=1)
        assert global_state.size() == (N, self.config["hidden_size"])

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": aux_loss}
