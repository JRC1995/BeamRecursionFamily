from torch import nn
import torch as T
import torch.nn.functional as F
from models.modules import GAU


class GAU_IN(nn.Module):
    def __init__(self, config):
        super(GAU_IN, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]

        self.content_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                               nn.LayerNorm(self.hidden_dim))

        self.layers = 5
        self.CLS = nn.Parameter(T.randn(self.hidden_dim))
        self.SEP = nn.Parameter(T.randn(self.hidden_dim))
        self.seg1 = nn.Parameter(T.zeros(self.hidden_dim))
        self.seg2 = nn.Parameter(T.zeros(self.hidden_dim))

        GAUS = [GAU(config) for _ in range(self.layers)]
        self.GAUS = nn.ModuleList(GAUS)

        self.energy_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                              nn.GELU(),
                                              nn.Linear(self.hidden_dim, 1))

    def masked_softmax(self, logits, mask, dim):
        if mask is None:
            return F.softmax(logits, dim=dim)

        logits = logits.masked_fill(~mask, float("-inf"))
        logits = F.softmax(logits, dim=dim)
        return logits

    def forward(self, input, input_mask):

        sequence = self.content_transform(input)
        N, S, D = sequence.size()

        CLS = self.CLS.view(1, D).repeat(N, 1)

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


        for l in range(self.layers):
            self.GAUS[l].pos_cache = None
            sequence = self.GAUS[l](sequence=sequence,
                                    attention_mask=attention_mask,
                                    positions=positions)["attended_values"]
            self.GAUS[l].pos_cache = None

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
