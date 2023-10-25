class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1
        self.batch_size = 128
        self.train_batch_size = 128
        self.dev_batch_size = 128
        self.bucket_size_factor = 5
        self.DataParallel = False
        self.num_workers = 6
        self.weight_decay = 1e-2
        self.lr = 1e-3
        self.epochs = 100
        self.early_stop_patience = 5
        self.scheduler = "ReduceLROnPlateau"
        self.scheduler_patience = 2
        self.scheduler_reduce_factor = 0.5
        self.optimizer = "Ranger"
        self.save_by = "accuracy"
        self.metric_direction = +1
        self.different_betas = False
        self.chunk_size = -1
        self.display_metric = "accuracy"


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze = True
        self.initial_transform = False
        self.batch_pair = False
        self.embd_dim = 300
        self.input_size = 300
        self.hidden_size = 300
        self.classifier_hidden_size = 300
        self.global_state_only = True
        self.global_state_return = True
        self.parse_trees = False
        self.classifier_mlp = True


class BiRecurrentGRC_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.4
        self.dropout = 0.2
        self.out_dropout = 0.4
        self.bidirectional = True
        self.encoder_type = "RecurrentGRC"
        self.model_name = "(BiRecurrentGRC)"

class BalancedTreeGRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "BalancedTreeCell"
        self.model_name = "(BalancedTreeGRC)"

class EBT_GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.stochastic = True
        self.beam_size = 5
        self.encoder_type = "EBT_GRC"
        self.model_name = "(EBT-GRC)"

class HEBT_GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.stochastic = True
        self.norm = "skip"
        self.bidirectional = False
        self.pre_SSM = True
        self.s4_dropout = self.dropout
        self.rba_temp = 1
        self.prenorm = False
        self.beam_size = 7
        self.model_chunk_size = 30
        self.RBA = True
        self.RBA_random = False
        self.RBA_advanced = False
        self.encoder_type = "HEBT_GRC"
        self.model_name = "(HEBT_GRC)"

class HGRC_config(HEBT_GRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "HGRC"
        self.model_name = "(HGRC)"

class EGT_GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "EGT_GRC"
        self.model_name = "(EGT-GRC)"

class CRvNN_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.early_stop_patience = 10
        self.encoder_type = "CRvNN"
        self.model_name = "(CRvNN)"

class OM_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.memory_dropout = 0.2
        self.memory_slots = 15
        self.encoder_type = "OrderedMemory"
        self.model_name = "(ordered_memory)"