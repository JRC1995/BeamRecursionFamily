from torch import nn
import torch as T
import torch.nn.functional as F
from models.encoders.S4DWrapper2 import S4DWrapper


class S4DStack(nn.Module):
    def __init__(self, config):
        super(S4DStack, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.layers = config["layers"]
        SSMS = [S4DWrapper(config) for _ in range(self.layers)]
        self.SSMS = nn.ModuleList(SSMS)

    def forward(self, input, input_mask):

        state = input
        for l in range(self.layers):
            state = self.SSMS[l](state, input_mask)["sequence"]

        sequence = state
        N, S, D = sequence.size()

        global_state = T.sum(sequence * input_mask.view(N, S, 1), dim=1) \
                       / T.sum(input_mask.view(N, S, 1), dim=1)
        assert global_state.size() == (N, D)


        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": None}
