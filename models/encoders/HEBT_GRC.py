import math
import torch
from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F
from models.modules.GRC import GRC
from models.utils.utils import stochastic_topk, masked_softmax
from models.encoders.S4DWrapper import S4DWrapper
from torch.distributions.categorical import Categorical


class HEBT_GRC(nn.Module):
    def __init__(self, config):
        super(HEBT_GRC, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.beam_size = config["beam_size"]
        self.small_d = 64
        self.rba_temp = config["rba_temp"]
        self.model_chunk_size = config["model_chunk_size"]

        if self.config["pre_SSM"]:
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

        self.treecell_layer = GRC(hidden_size=self.hidden_dim,
                                  cell_hidden_size=4 * self.hidden_dim,
                                  dropout=config["dropout"])

        self.decision_module = nn.Sequential(nn.Linear(2 * self.small_d, self.small_d),
                                             nn.GELU(),
                                             nn.Linear(self.small_d, 1))

    def normalize(self, state):
        if self.norm == "batch":
            return self.NT(state.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif self.norm == "skip":
            return state
        else:
            return self.NT(state)

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

    def beam_search(self, content_state, length_mask, accu_scores, beam_mask):

        N, B, S, D = content_state.size()

        for i in range(S - 1):
            B = content_state.size(1)

            if i < S - 2:
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
        assert content_state.size() == (N, B, 1, D)
        content_state = content_state.view(N, B, D)

        return {"content_state": content_state, "beam_mask": beam_mask, "accu_scores": accu_scores}

    def RBA(self, group, k, uniform_randomize=False):
        sequence = group["sequence"]
        beam_mask = group["beam_mask"]
        accu_scores = group["accu_scores"]
        N, B, D = sequence.size()

        with T.no_grad():
            if uniform_randomize:
                accu_scores_ = T.ones(N, k, B).float().to(accu_scores.device)
            else:
                accu_scores_ = accu_scores.unsqueeze(1).repeat(1, k, 1)
            dist = Categorical(probs=masked_softmax(accu_scores_ / self.rba_temp,
                                                    mask=beam_mask.unsqueeze(1), dim=-1))
            max_scores_idx = dist.sample()
            assert max_scores_idx.size() == (N, k)
            select_mask = F.one_hot(max_scores_idx, num_classes=B).float()

        assert select_mask.size() == (N, k, B)
        sequence = T.matmul(select_mask, sequence)
        accu_scores = T.matmul(select_mask, accu_scores.unsqueeze(-1)).squeeze(-1)
        beam_mask = T.matmul(select_mask, beam_mask.unsqueeze(-1)).squeeze(-1)
        assert sequence.size() == (N, k, D)
        assert accu_scores.size() == (N, k)
        assert beam_mask.size() == (N, k)

        return {"sequence": sequence,
                "beam_mask": beam_mask,
                "accu_scores": accu_scores}

    def permute_beams(self, group):
        sequence = group["sequence"]
        beam_mask = group["beam_mask"]
        accu_scores = group["accu_scores"]
        N, B, D = sequence.size()

        uniform_scores = T.ones(N, B).float().to(accu_scores.device)

        permute_mask, _ = stochastic_topk(logits=uniform_scores,
                                          mask=beam_mask,
                                          select_k=B,
                                          training=True)
        assert permute_mask.size() == (N, B, B)

        sequence = T.matmul(permute_mask, sequence)
        accu_scores = T.matmul(permute_mask, accu_scores.unsqueeze(-1)).squeeze(-1)
        beam_mask = T.matmul(permute_mask, beam_mask.unsqueeze(-1)).squeeze(-1)
        assert sequence.size() == (N, B, D)
        assert accu_scores.size() == (N, B)
        assert beam_mask.size() == (N, B)

        return sequence, accu_scores, beam_mask

    def forward(self, input, input_mask):

        S = input.size(1)
        sequence = input

        if self.config["pre_SSM"]:
            sequence = self.RNN(sequence, input_mask)["sequence"]
        sequence = self.normalize(self.initial_transform(sequence))

        osequence = sequence.clone()
        oinput_mask = input_mask.clone()

        N, S, D = sequence.size()
        if not self.config["chunk_mode_inference"] and not self.training:
            self.chunk_size = S
        else:
            self.chunk_size = self.model_chunk_size

        B = 1
        sequence = sequence.unsqueeze(1)
        accu_scores = T.zeros(N, B, S).float().to(sequence.device)
        beam_mask = T.ones(N, B, S).float().to(sequence.device)
        N0 = N
        while S > 1:
            N, B, S, D = sequence.size()
            assert accu_scores.size() == (N, B, S)
            assert beam_mask.size() == (N, B, S)
            if S >= (self.chunk_size + self.chunk_size // 2):
                if S % self.chunk_size != 0:
                    e = ((S // self.chunk_size) * self.chunk_size) + self.chunk_size - S
                    S = S + e
                    pad = T.zeros(N, B, e, D).float().to(sequence.device)
                    input_mask = T.cat([input_mask, T.zeros(N, e).float().to(sequence.device)], dim=-1)
                    sequence = T.cat([sequence, pad], dim=-2)
                    assert sequence.size() == (N, B, S, D)
                    assert input_mask.size() == (N, S)
                    pad2 = pad[..., 0]
                    accu_scores = T.cat([accu_scores, pad2], dim=-1)
                    beam_mask = T.cat([beam_mask, pad2], dim=-1)
                    assert accu_scores.size() == (N, B, S)
                    assert beam_mask.size() == (N, B, S)
                S1 = S // self.chunk_size
                chunk_size = self.chunk_size
            else:
                S1 = 1
                chunk_size = S
            sequence = sequence.view(N, B, S1, chunk_size, D).permute(0, 2, 1, 3, 4).contiguous()
            assert sequence.size() == (N, S1, B, chunk_size, D)
            sequence = sequence.view(N * S1, B, chunk_size, D)

            input_mask = input_mask.view(N, S1, chunk_size)
            input_mask = input_mask.view(N * S1, chunk_size)

            accu_scores = accu_scores.view(N, B, S1, chunk_size).permute(0, 2, 1, 3).contiguous()
            assert accu_scores.size() == (N, S1, B, chunk_size)
            accu_scores = T.sum(accu_scores, dim=-1)
            accu_scores = accu_scores.view(N * S1, B)

            beam_mask = beam_mask.view(N, B, S1, chunk_size).permute(0, 2, 1, 3).contiguous()
            assert beam_mask.size() == (N, S1, B, chunk_size)
            beam_mask = beam_mask[..., 0]
            beam_mask = beam_mask.view(N * S1, B)

            N0 = N
            N, B, S, D = sequence.size()
            assert N == N0 * S1

            content_dict = self.beam_search(content_state=sequence,
                                            length_mask=input_mask,
                                            beam_mask=beam_mask, accu_scores=accu_scores)

            sequence = content_dict["content_state"]
            beam_mask = content_dict["beam_mask"]
            accu_scores = content_dict["accu_scores"]

            B = sequence.size(1)

            assert sequence.size() == (N, B, D)
            assert accu_scores.size() == (N, B)
            assert beam_mask.size() == (N, B)

            if S1 > 1 and self.config["RBA"] and B >= 2:
                if self.config["RBA_random"]:
                    zero_group = {"sequence": sequence[:, 0, :].unsqueeze(1),
                                  "accu_scores": accu_scores[:, 0].unsqueeze(1),
                                  "beam_mask": beam_mask[:, 0].unsqueeze(1)}
                    B1 = B - 1
                    group = {"sequence": sequence,
                             "accu_scores": accu_scores,
                             "beam_mask": beam_mask}

                    first_group = self.RBA(group, k=B1, uniform_randomize=True)
                    sequence = T.cat([zero_group["sequence"],
                                      first_group["sequence"]], dim=1)
                    accu_scores = T.cat([zero_group["accu_scores"],
                                         first_group["accu_scores"]], dim=1)
                    beam_mask = T.cat([zero_group["beam_mask"],
                                       first_group["beam_mask"]], dim=1)
                elif self.config["RBA_advanced"]:
                    zero_group = {"sequence": sequence[:, 0, :].unsqueeze(1),
                                  "accu_scores": accu_scores[:, 0].unsqueeze(1),
                                  "beam_mask": beam_mask[:, 0].unsqueeze(1)}

                    B1 = (B - 1) // 2
                    B2 = B - 1 - B1

                    first_group = {"sequence": sequence[:, 0::2, :],
                                   "accu_scores": accu_scores[:, 0::2],
                                   "beam_mask": beam_mask[:, 0::2]}
                    second_group = {"sequence": sequence[:, 1::2, :],
                                    "accu_scores": accu_scores[:, 1::2],
                                    "beam_mask": beam_mask[:, 1::2]}

                    first_group = self.RBA(first_group, k=B1, uniform_randomize=False)
                    second_group = self.RBA(second_group, k=B2, uniform_randomize=False)

                    sequence = T.cat([first_group["sequence"],
                                      second_group["sequence"]], dim=1)
                    accu_scores = T.cat([first_group["accu_scores"],
                                         second_group["accu_scores"]], dim=1)
                    beam_mask = T.cat([first_group["beam_mask"],
                                       second_group["beam_mask"]], dim=1)

                    group = {"sequence": sequence,
                             "accu_scores": accu_scores,
                             "beam_mask": beam_mask}
                    sequence, accu_scores, beam_mask = self.permute_beams(group)

                    sequence = T.cat([zero_group["sequence"],
                                      sequence], dim=1)
                    accu_scores = T.cat([zero_group["accu_scores"],
                                         accu_scores], dim=1)
                    beam_mask = T.cat([zero_group["beam_mask"],
                                       beam_mask], dim=1)
                else:
                    zero_group = {"sequence": sequence[:, 0, :].unsqueeze(1),
                                  "accu_scores": accu_scores[:, 0].unsqueeze(1),
                                  "beam_mask": beam_mask[:, 0].unsqueeze(1)}
                    B1 = B - 1
                    group = {"sequence": sequence,
                             "accu_scores": accu_scores,
                             "beam_mask": beam_mask}

                    first_group = self.RBA(group, k=B1, uniform_randomize=False)
                    sequence = T.cat([zero_group["sequence"],
                                      first_group["sequence"]], dim=1)
                    accu_scores = T.cat([zero_group["accu_scores"],
                                         first_group["accu_scores"]], dim=1)
                    beam_mask = T.cat([zero_group["beam_mask"],
                                       first_group["beam_mask"]], dim=1)

                assert sequence.size() == (N, B, D)
                assert accu_scores.size() == (N, B)
                assert beam_mask.size() == (N, B)

            sequence = sequence.view(N0, S1, B, D).permute(0, 2, 1, 3).contiguous()
            beam_mask = beam_mask.view(N0, S1, B).permute(0, 2, 1).contiguous()
            accu_scores = accu_scores.view(N0, S1, B).permute(0, 2, 1).contiguous()
            input_mask = input_mask.view(N0, S1, chunk_size)[:, :, 0]

            assert sequence.size() == (N0, B, S1, D)
            assert input_mask.size() == (N0, S1)
            assert accu_scores.size() == (N0, B, S1)
            assert beam_mask.size() == (N0, B, S1)
            S = S1

        N = N0
        assert sequence.size() == (N, B, 1, D)
        assert accu_scores.size() == (N, B, 1)
        assert beam_mask.size() == (N, B, 1)
        sequence = sequence.squeeze(-2)
        accu_scores = accu_scores.squeeze(-1)
        beam_mask = beam_mask.squeeze(-1)

        h = sequence
        input_mask = oinput_mask.unsqueeze(-1)
        sequence = osequence
        aux_loss = None

        N, B, D = h.size()
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
