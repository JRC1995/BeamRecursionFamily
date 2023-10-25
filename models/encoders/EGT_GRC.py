from torch import nn
import torch as T
import torch.nn.functional as F
from models.modules import GRC
from models.utils import st_gumbel_softmax


class EGT_GRC(nn.Module):
    def __init__(self, config):
        super(EGT_GRC, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.form_hidden_dim = 64

        self.init_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                            nn.LayerNorm(self.hidden_dim))

        self.treecell_layer = GRC(hidden_size=self.hidden_dim,
                                  cell_hidden_size=4 * self.hidden_dim,
                                  dropout=config["dropout"])

        self.decision_module = nn.Sequential(nn.Linear(2 * self.form_hidden_dim, self.form_hidden_dim),
                                             nn.GELU(),
                                             nn.Linear(self.form_hidden_dim, 1))

    @staticmethod
    def update_state(old_content_state, new_content_state, done_mask):
        N = old_content_state.size(0)
        done_mask = done_mask.view(N, 1, 1)
        content_state = done_mask * new_content_state + (1 - done_mask) * old_content_state[..., :-1, :]
        return content_state

    def select_composition(self, content_state,
                           mask):

        S = mask.size(-1)

        N, _, _ = content_state.size()
        D = content_state.size(-1)
        fD = self.form_hidden_dim
        assert mask.size() == (N, S)

        l = content_state[:, :-1, 0:self.form_hidden_dim]
        r = content_state[:, 1:, 0:self.form_hidden_dim]
        assert l.size() == (N, S, fD)
        assert r.size() == (N, S, fD)

        comp_weights = self.decision_module(T.cat([l, r], dim=-1)).squeeze(-1)
        assert comp_weights.size() == (N, S)
        select_mask, _ = st_gumbel_softmax(logits=comp_weights,
                                           mask=mask,
                                           training=self.training)

        assert content_state.size() == (N, S + 1, D)
        assert select_mask.size() == (N, S)

        l = content_state[:, :-1, :]
        r = content_state[:, 1:, :]
        l = T.matmul(select_mask.unsqueeze(-2), l)
        r = T.matmul(select_mask.unsqueeze(-2), r)
        assert l.size() == (N, 1, D)
        assert r.size() == (N, 1, D)
        new_content_state = self.treecell_layer(l.view(N, D), r.view(N, D)).view(N, 1, D)

        select_mask_expand = select_mask.unsqueeze(-1)
        select_mask_cumsum = select_mask.cumsum(-1)

        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(-1)

        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(-1)

        olc, orc = content_state[..., :-1, :], content_state[..., 1:, :]

        assert select_mask_expand.size() == (N, S, 1)
        assert left_mask_expand.size() == (N, S, 1)
        assert right_mask_expand.size() == (N, S, 1)
        assert new_content_state.size() == (N, 1, D)
        assert olc.size() == (N, S, D)
        assert orc.size() == (N, S, D)

        new_content_state = (select_mask_expand * new_content_state
                             + left_mask_expand * olc
                             + right_mask_expand * orc)

        return {"new_content_state": new_content_state,
                "content_state": content_state}

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

        for i in range(max_depth - 1):
            B = content_state.size(1)

            if i < max_depth - 2:
                out_dict = self.select_composition(content_state=content_state,
                                                   mask=length_mask[:, i + 1:])
                new_content_state = out_dict["new_content_state"]
                content_state = out_dict["content_state"]
            else:
                l = content_state[:, :-1, :]
                r = content_state[:, 1:, :]
                assert l.size() == (N, 1, D)
                assert r.size() == (N, 1, D)
                new_content_state = self.treecell_layer(l.view(N, D), r.view(N, D)).view(N, 1, D)

            done_mask = length_mask[:, i + 1]
            content_state = self.update_state(old_content_state=content_state,
                                              new_content_state=new_content_state,
                                              done_mask=done_mask)

        assert content_state.size() == (N, 1, D)
        CLS = content_state.squeeze(1)
        assert CLS.size() == (N, D)

        return {"sequence": input,
                "global_state": CLS,
                "input_mask": input_mask,
                "aux_loss": None}
