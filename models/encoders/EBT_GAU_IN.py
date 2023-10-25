from torch import nn
import torch as T
import torch.nn.functional as F
from models.modules import GRC, GAU, PGAU
from models.utils import stochastic_topk


class EBT_GAU_IN(nn.Module):
    def __init__(self, config):
        super(EBT_GAU_IN, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.form_hidden_dim = 64
        self.beam_size = config["beam_size"]

        self.init_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                            nn.LayerNorm(self.hidden_dim))

        self.treecell_layer = GRC(hidden_size=self.hidden_dim,
                                  cell_hidden_size=4 * self.hidden_dim,
                                  dropout=config["dropout"])

        self.decision_module = nn.Sequential(nn.Linear(2 * self.form_hidden_dim, self.form_hidden_dim),
                                             nn.GELU(),
                                             nn.Linear(self.form_hidden_dim, 1))

        self.cls_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                           nn.GELU(),
                                           nn.Linear(self.hidden_dim, self.hidden_dim))

        self.layers1 = 2
        self.layers2 = 3
        self.PGAU = PGAU(config)
        self.GAU = GAU(config)

        self.SEP = nn.Parameter(T.randn(self.hidden_dim))
        self.seg1 = nn.Parameter(T.zeros(self.hidden_dim))
        self.seg2 = nn.Parameter(T.zeros(self.hidden_dim))

        self.energy_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                              nn.GELU(),
                                              nn.Linear(self.hidden_dim, 1))

    @staticmethod
    def update_state(old_content_state, new_content_state, done_mask):
        N = old_content_state.size(0)
        done_mask = done_mask.view(N, 1, 1, 1)
        content_state = done_mask * new_content_state + (1 - done_mask) * old_content_state[..., :-1, :]
        return content_state

    def select_composition(self, content_state,
                           new_sequence,
                           graph_state, graph_structure,
                           mask, accu_scores, beam_mask):

        S = mask.size(-1)

        N, B, _, _ = content_state.size()
        D = content_state.size(-1)
        fD = self.form_hidden_dim
        assert accu_scores.size() == (N, B)
        assert mask.size() == (N, S)
        assert beam_mask.size() == (N, B)

        l = content_state[:, :, :-1, 0:self.form_hidden_dim]
        r = content_state[:, :, 1:, 0:self.form_hidden_dim]
        assert l.size() == (N, B, S, fD)
        assert r.size() == (N, B, S, fD)

        comp_weights = self.decision_module(T.cat([l, r], dim=-1)).squeeze(-1)
        topk = min(S, self.beam_size)  # beam_size
        select_mask, soft_scores = stochastic_topk(logits=comp_weights.view(N * B, S),
                                                   mask=mask.view(N, 1, S).repeat(1, B, 1).view(N * B, S),
                                                   select_k=topk,
                                                   training=self.training)

        soft_scores = soft_scores.view(N, B, 1, S)
        assert select_mask.size() == (N * B, topk, S)
        select_mask = select_mask.view(N, B, topk, S)

        new_scores = mask[:, 0].view(N, 1, 1) * T.log(T.sum(select_mask * soft_scores, dim=-1) + 1e-20)
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

        content_state = content_state.unsqueeze(2).repeat(1, 1, topk, 1, 1)
        assert content_state.size() == (N, B, topk, S + 1, D)
        content_state = content_state.view(N, B * topk, S + 1, D)

        graph_state = graph_state.unsqueeze(2).repeat(1, 1, topk, 1, 1)
        X = graph_state.size(-1)
        assert graph_state.size() == (N, B, topk, S + 1, X)
        graph_state = graph_state.view(N, B * topk, S + 1, X)

        graph_structure = graph_structure.unsqueeze(2).repeat(1, 1, topk, 1, 1)
        X2 = graph_structure.size(-2)
        assert graph_structure.size() == (N, B, topk, X2, X)
        graph_structure = graph_structure.view(N, B * topk, X2, X)

        new_sequence = new_sequence.unsqueeze(2).repeat(1, 1, topk, 1, 1)
        assert new_sequence.size() == (N, B, topk, X2, D)
        new_sequence = new_sequence.view(N, B * topk, X2, D)

        if (B * topk) > self.beam_size:
            B2 = self.beam_size
            assert accu_scores.size() == beam_mask.size()
            beam_select_mask, _ = stochastic_topk(logits=accu_scores,
                                                  mask=beam_mask,
                                                  select_k=B2,
                                                  training=self.training)
            assert beam_select_mask.size() == (N, B2, B * topk)

            content_state = T.matmul(beam_select_mask, content_state.view(N, B * topk, -1))
            content_state = content_state.view(N, B2, S + 1, D)

            graph_state = T.matmul(beam_select_mask, graph_state.view(N, B * topk, -1))
            graph_state = graph_state.view(N, B2, S + 1, X)

            graph_structure = T.matmul(beam_select_mask, graph_structure.view(N, B * topk, -1))
            graph_structure = graph_structure.view(N, B2, X2, X)

            new_sequence = T.matmul(beam_select_mask, new_sequence.view(N, B * topk, -1))
            new_sequence = new_sequence.view(N, B2, X2, D)

            select_mask = T.matmul(beam_select_mask, select_mask)
            assert select_mask.size() == (N, B2, S)

            accu_scores = T.matmul(beam_select_mask, accu_scores.unsqueeze(-1)).squeeze(-1)
            assert accu_scores.size() == (N, B2)

            beam_mask = T.matmul(beam_select_mask, beam_mask.unsqueeze(-1)).squeeze(-1)
            assert beam_mask.size() == (N, B2)
        else:
            B2 = B * topk

        assert content_state.size() == (N, B2, S + 1, D)
        assert select_mask.size() == (N, B2, S)

        l = content_state[:, :, :-1, :]
        r = content_state[:, :, 1:, :]
        l = T.matmul(select_mask.unsqueeze(-2), l)
        r = T.matmul(select_mask.unsqueeze(-2), r)
        assert l.size() == (N, B2, 1, D)
        assert r.size() == (N, B2, 1, D)
        new_content_state = self.treecell_layer(l.view(N * B2, D), r.view(N * B2, D)).view(N, B2, 1, D)

        new_sequence = T.cat([new_sequence, new_content_state], dim=-2)
        assert new_sequence.size() == (N, B2, X2 + 1, D)

        l = graph_state[:, :, :-1, 0:X - 1]
        r = graph_state[:, :, 1:, 0:X - 1]
        new_graph_state_p1 = l + r
        l = graph_state[:, :, :-1, -1]
        r = graph_state[:, :, 1:, -1]
        new_graph_state_p2 = T.max(l, r) + 1
        assert new_graph_state_p2.size() == (N, B2, S)
        new_graph_state_p2 = new_graph_state_p2.unsqueeze(-1)
        new_graph_state = T.cat([new_graph_state_p1, new_graph_state_p2], dim=-1)

        new_graph_state = T.matmul(select_mask.unsqueeze(-2), new_graph_state)
        assert new_graph_state.size() == (N, B2, 1, X)

        graph_structure = T.cat([graph_structure, new_graph_state], dim=-2)
        assert graph_structure.size() == (N, B2, X2 + 1, X)

        select_mask_expand = select_mask.unsqueeze(-1)
        select_mask_cumsum = select_mask.cumsum(-1)

        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(-1)

        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(-1)

        olc, orc = content_state[..., :-1, :], content_state[..., 1:, :]
        olg, org = graph_state[..., :-1, :], graph_state[..., 1:, :]

        assert select_mask_expand.size() == (N, B2, S, 1)
        assert left_mask_expand.size() == (N, B2, S, 1)
        assert right_mask_expand.size() == (N, B2, S, 1)
        assert new_content_state.size() == (N, B2, 1, D)
        assert olc.size() == (N, B2, S, D)
        assert orc.size() == (N, B2, S, D)

        new_content_state = (select_mask_expand * new_content_state
                             + left_mask_expand * olc
                             + right_mask_expand * orc)

        graph_state = (select_mask_expand * new_graph_state
                       + left_mask_expand * olg
                       + right_mask_expand * org)

        return {"new_content_state": new_content_state, "graph_state": graph_state,
                "content_state": content_state,
                "graph_structure": graph_structure, "new_sequence": new_sequence,
                "accu_scores": accu_scores, "beam_mask": beam_mask}

    def masked_softmax(self, logits, mask, dim):
        if mask is None:
            return F.softmax(logits, dim=dim)

        logits = logits.masked_fill(~mask, float("-inf"))
        logits = F.softmax(logits, dim=dim)
        return logits

    def forward(self, input, input_mask):

        max_depth = input.size(1)
        length_mask = input_mask

        content_state = self.init_transform(input)

        N, S, D = content_state.size()
        assert input_mask.size() == (N, S)
        B = 1
        content_state = content_state.unsqueeze(1)
        assert content_state.size() == (N, B, S, D)

        accu_scores = T.zeros(N, B).float().to(content_state.device)
        beam_mask = T.ones(N, B).float().to(content_state.device)

        graph_idx = T.arange(0, 2 * S - 1).long().to(content_state.device).view(1, 1, 2 * S - 1).repeat(N, B, 1)
        graph = F.one_hot(graph_idx, num_classes=2 * S - 1).float()
        assert graph.size() == (N, B, 2 * S - 1, 2 * S - 1)
        heights = T.zeros(N, B, 2 * S - 1, 1).float().to(content_state.device)
        graph = T.cat([graph, heights], dim=-1)
        assert graph.size() == (N, B, 2 * S - 1, 2 * S)

        graph_state = graph[:, :, 0:S, :]
        graph_structure = graph[:, :, 0:S, :]
        new_sequence = content_state.clone()

        original_new_sequence = new_sequence.clone()
        original_graph_structure = graph_structure.clone()
        new_sequence = new_sequence[:, :, 0, :].unsqueeze(-2)
        graph_structure = graph_structure[:, :, 0, :].unsqueeze(-2)

        graph_mask = T.cat([input_mask, input_mask[:, 1:]], dim=-1)
        assert graph_mask.size() == (N, 2 * S - 1)

        for i in range(max_depth - 1):
            B = content_state.size(1)

            if i < max_depth - 2:
                out_dict = self.select_composition(content_state=content_state,
                                                   graph_state=graph_state,
                                                   graph_structure=graph_structure,
                                                   new_sequence=new_sequence,
                                                   mask=length_mask[:, i + 1:],
                                                   accu_scores=accu_scores,
                                                   beam_mask=beam_mask)
                new_content_state = out_dict["new_content_state"]
                graph_state = out_dict["graph_state"]
                graph_structure = out_dict["graph_structure"]
                new_sequence = out_dict["new_sequence"]
                accu_scores = out_dict["accu_scores"]
                beam_mask = out_dict["beam_mask"]
                content_state = out_dict["content_state"]
            else:
                l = content_state[:, :, :-1, :]
                r = content_state[:, :, 1:, :]
                assert l.size() == (N, B, 1, D)
                assert r.size() == (N, B, 1, D)
                new_content_state = self.treecell_layer(l.view(N * B, D), r.view(N * B, D)).view(N, B, 1, D)

                new_sequence = T.cat([original_new_sequence.repeat(1, B, 1, 1),
                                      new_sequence[:, :, 1:, :], new_content_state], dim=-2)
                assert new_sequence.size() == (N, B, 2 * S - 1, D)

                l = graph_state[:, :, :-1, 0:- 1]
                r = graph_state[:, :, 1:, 0:- 1]
                new_graph_state_p1 = l + r
                l = graph_state[:, :, :-1, -1]
                r = graph_state[:, :, 1:, -1]
                new_graph_state_p2 = T.max(l, r) + 1
                assert new_graph_state_p2.size() == (N, B, 1)
                new_graph_state_p2 = new_graph_state_p2.unsqueeze(-1)
                new_graph_state = T.cat([new_graph_state_p1, new_graph_state_p2], dim=-1)
                assert new_graph_state.size() == (N, B, 1, 2 * S)

                graph_structure = T.cat([original_graph_structure.repeat(1, B, 1, 1),
                                         graph_structure[..., 1:, :],
                                         new_graph_state], dim=-2)
                assert graph_structure.size() == (N, B, 2 * S - 1, 2 * S)

            done_mask = length_mask[:, i + 1]
            content_state = self.update_state(old_content_state=content_state,
                                              new_content_state=new_content_state,
                                              done_mask=done_mask)

        graph_mask_beam = graph_mask.unsqueeze(1).repeat(1, B, 1)
        assert graph_mask_beam.size() == (N, B, 2 * S - 1)
        attention_mask = graph_structure[..., 0:- 1] * graph_mask_beam.unsqueeze(-2) * graph_mask_beam.unsqueeze(-1)
        assert attention_mask.size() == (N, B, 2 * S - 1, 2 * S - 1)

        heights = graph_structure[..., -1].long()
        assert heights.size() == (N, B, 2 * S - 1)

        sequence = new_sequence.view(N * B, 2 * S - 1, D)
        heights = heights.view(N * B, 2 * S - 1)
        child_mask = attention_mask.view(N * B, 2 * S - 1, 2 * S - 1)
        parent_mask = child_mask.permute(0, 2, 1).contiguous()

        diag = T.eye(2 * S - 1).float().to(attention_mask.device).view(1, 2 * S - 1, 2 * S - 1).repeat(N * B, 1, 1)
        parent_mask = (1 - diag) * parent_mask + diag

        for l in range(self.layers1):
            sequence = self.PGAU(sequence=sequence,
                                 attention_mask=parent_mask,
                                 heights=heights)["attended_values"]

        assert accu_scores.size() == (N, B)
        assert beam_mask.size() == (N, B)
        normed_scores = self.masked_softmax(accu_scores, dim=-1, mask=beam_mask.bool())
        assert normed_scores.size() == (N, B)

        assert sequence.size() == (N * B, 2 * S - 1, D)
        sequence = sequence.view(N, B, 2 * S - 1, D)
        sequence = T.sum(normed_scores.view(N, B, 1, 1) * sequence, dim=1)
        assert sequence.size() == (N, 2 * S - 1, D)

        assert content_state.size() == (N, B, 1, D)
        CLS = T.sum(normed_scores.unsqueeze(-1) * content_state.squeeze(-2), dim=1)
        assert CLS.size() == (N, D)
        CLS = self.cls_transform(CLS)

        sequence = sequence[:, 0:S, :]
        sequence = T.cat([CLS.unsqueeze(1), sequence], dim=1)
        S = S + 1
        assert sequence.size() == (N, S, D)
        sequence = sequence.view(N, S, D)

        input_mask = T.cat([T.ones(N, 1).float().to(input_mask.device),
                            input_mask], dim=-1)
        assert input_mask.size() == (N, S)

        lengths = T.sum(input_mask, dim=-1).view(N, 1)
        N = N // 2
        OS = S
        S = 2 * S + 1
        SEP = self.SEP.view(1, 1, D).repeat(N, 1, 1)
        seg1 = self.seg1.view(1, 1, D).repeat(N, 1, 1)
        seg2 = self.seg2.view(1, 1, D).repeat(N, 1, 1)
        sequence = T.cat([sequence[0:N] + seg1, SEP, sequence[N:] + seg2], dim=1)
        assert sequence.size() == (N, S, D)
        input_mask = T.cat([input_mask[0:N], T.ones(N, 1).float().to(input_mask.device), input_mask[N:]], dim=1)
        assert input_mask.size() == (N, S)

        attention_mask = input_mask.unsqueeze(1).repeat(1, S, 1)

        positions1 = T.arange(0, OS).long().to(attention_mask.device).view(1, OS).repeat(N, 1)
        positions2 = lengths[0:N].long() + T.arange(0, OS + 1).long().to(attention_mask.device).view(1, OS + 1)
        assert positions1.size() == (N, OS)
        assert positions2.size() == (N, OS + 1)
        positions = T.cat([positions1, positions2], dim=1)
        assert positions.size() == (N, S)

        for l in range(self.layers2):
            sequence = self.GAU(sequence=sequence,
                                attention_mask=attention_mask,
                                positions=positions)["attended_values"]

        assert sequence.size() == (N, S, D)

        e = self.energy_transform(sequence)
        assert e.size() == (N, S, 1)
        a = self.masked_softmax(e, mask=input_mask.unsqueeze(-1).bool(), dim=1)
        assert a.size() == (N, S, 1)

        global_state = T.sum(a * sequence, dim=1)
        assert global_state.size() == (N, D)

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": None}
