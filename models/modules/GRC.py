from torch import nn
import torch as T
import torch.nn.functional as F

class GRC(nn.Module):
    def __init__(self, hidden_size, cell_hidden_size, dropout, config=None):
        super(GRC, self).__init__()
        self.hidden_dim = hidden_size
        self.wcell1 = nn.Linear(2 * hidden_size, cell_hidden_size)
        self.wcell2 = nn.Linear(cell_hidden_size, 4 * hidden_size)
        if config and "rvnn_norm" in config:
            self.norm = config["rvnn_norm"]
        else:
            self.norm = "layer"

        if self.norm == "batch":
            self.NT = nn.BatchNorm1d(hidden_size)
        elif self.norm == "skip":
            pass
        else:
            self.NT = nn.LayerNorm(hidden_size)

        self.dropout = dropout

    def normalize(self, state):
        if self.norm == "batch":
            return self.NT(state)
        elif self.norm == "skip":
            return state
        else:
            return self.NT(state)

    def forward(self, left=None, right=None):
        N, D = left.size()
        assert right.size() == left.size()

        concated = T.cat([left, right], dim=-1)

        intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, 4, D)
        gates = T.sigmoid(contents[..., 0:3, :])
        parent = contents[..., 3, :]

        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]

        out = self.normalize(f1 * left + f2 * right + i * parent)

        return out