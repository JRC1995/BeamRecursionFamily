from torch import nn
import torch as T
from models.modules.GRC import GRC


class RecurrentGRC(nn.Module):

    def __init__(self, config):
        super(RecurrentGRC, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]

        self.fcell = GRC(hidden_size=self.hidden_dim // 2,
                         cell_hidden_size=2 * self.hidden_dim,
                         dropout=config["dropout"])
        self.bcell = GRC(hidden_size=self.hidden_dim // 2,
                         cell_hidden_size=2 * self.hidden_dim,
                         dropout=config["dropout"])

        self.init_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                            nn.LayerNorm(self.hidden_dim))
        self.out_transform = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, input, input_mask):

        state = input
        N, S, D = state.size()
        input_mask = input_mask.view(N, S, 1)

        res = state.clone()
        state = self.init_transform(state)
        fstate = state[..., 0:D // 2]
        bstate = state[..., D // 2:]

        D = D // 2
        h = T.zeros(N, D).float().to(state.device)
        f_hs = []
        for i in range(S):
            m = input_mask[:, i, :]
            inp = fstate[:, i, :]
            assert h.size() == (N, D)
            assert inp.size() == (N, D)
            assert m.size() == (N, 1)
            h_ = self.fcell(left=h, right=inp)
            h = m * h_ + (1 - m) * h
            f_hs.append(h)

        fsequence = T.stack(f_hs, dim=1)
        assert fsequence.size() == (N, S, D)

        b_hs = []
        h = T.zeros(N, D).float().to(state.device)
        for i in range(S):
            m = input_mask[:, S - 1 - i, :]
            inp = bstate[:, S - 1 - i, :]
            assert h.size() == (N, D)
            assert inp.size() == (N, D)
            assert m.size() == (N, 1)
            h_ = self.bcell(left=h, right=inp)
            h = m * h_ + (1 - m) * h
            b_hs.append(h)

        b_hs.reverse()
        bsequence = T.stack(b_hs, dim=1)
        sequence = T.cat([fsequence, bsequence], dim=-1)
        D = 2 * D
        assert sequence.size() == (N, S, D)
        sequence = self.out_transform(sequence)

        global_state = T.mean(sequence, dim=1)

        aux_loss = None

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": aux_loss}
