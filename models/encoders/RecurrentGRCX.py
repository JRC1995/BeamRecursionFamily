from torch import nn
import torch as T
from models.modules.GRC import GRC


class RecurrentGRCX(nn.Module):

    def __init__(self, config):
        super(RecurrentGRCX, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]

        self.fcell = GRC(hidden_size=self.hidden_dim,
                         cell_hidden_size=4 * self.hidden_dim,
                         dropout=config["dropout"], config=config)

    def forward(self, input, input_mask):

        state = input
        N, S, D = state.size()
        input_mask = input_mask.view(N, S, 1)

        fstate = state
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


        aux_loss = None

        return {"sequence": fsequence,
                "global_state": h,
                "input_mask": input_mask,
                "aux_loss": aux_loss}
