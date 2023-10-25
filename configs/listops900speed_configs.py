class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1
        self.batch_size = 1
        self.train_batch_size = 1
        self.dev_batch_size = 1
        self.bucket_size_factor = 1
        self.DataParallel = False
        self.num_workers = 6
        self.weight_decay = 1e-2
        self.lr = 1e-3
        self.epochs = 100
        self.early_stop_patience = 4
        self.scheduler = "ReduceLROnPlateau"
        self.scheduler_patience = 2
        self.scheduler_reduce_factor = 0.5
        self.optimizer = "Ranger"
        self.save_by = "loss"
        self.metric_direction = -1
        self.different_betas = False
        self.chunk_size = -1
        self.display_metric = "accuracy"


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze = False
        self.initial_transform = False
        self.batch_pair = False
        self.prenorm = False
        self.embd_dim = 128
        self.input_size = 128
        self.hidden_size = 128
        self.s4_dropout = 0
        self.classifier_hidden_size = 128
        self.global_state_only = True
        self.global_state_return = True
        self.parse_trees = False
        self.classifier_mlp = True


class BiRecurrentGRC_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.1
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.bidirectional = True
        self.encoder_type = "RecurrentGRC"
        self.model_name = "(BiRecurrentGRC)"

class S4DStack_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.optimizer = "AdamW"
        self.weight_decay = 0.05
        self.save_by = "accuracy"
        self.classifier_mlp = False
        self.metric_direction = 1
        self.norm = "batch"
        self.epochs = 40
        self.early_stop_patience = 100
        self.scheduler = "cosine"
        self.max_grad_norm = None
        self.bucket_size_factor = 1
        self.warmup_steps = 1000
        self.training_steps = 40 * 1500
        self.layers = 8
        self.lr = 1e-2
        self.in_dropout = 0
        self.s4_dropout = 0
        self.dropout = 0
        self.out_dropout = 0
        self.encoder_type = "S4DStack"
        self.model_name = "(S4DStack)"


class GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "RecurrentGRCX"
        self.model_name = "(GRC)"

class BalancedTreeGRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "BalancedTreeCell"
        self.model_name = "(BalancedTreeGRC)"

class CRvNN_nohalt_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "CRvNN_nohalt"
        self.model_name = "(CRvNN nohalt)"

class OM_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.memory_dropout = 0.1
        self.in_dropout = 0.1
        self.out_dropout = 0.1
        self.memory_slots = 21
        self.encoder_type = "OrderedMemory"
        self.model_name = "(OM)"

class BT_GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.stochastic = True
        self.beam_size = 5
        self.encoder_type = "BT_GRC"
        self.model_name = "(BT-GRC)"

class EGT_GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "EGT_GRC"
        self.model_name = "(EGT-GRC)"


class EBT_GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.stochastic = True
        self.beam_size = 5
        self.encoder_type = "EBT_GRC"
        self.model_name = "(EBT-GRC)"

class EBT_GRC_noslice_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.stochastic = True
        self.beam_size = 5
        self.encoder_type = "EBT_GRC_noslice"
        self.model_name = "(EBT-GRC-noslice)"

class EBT_GRC512_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.stochastic = True
        self.beam_size = 5
        self.hidden_size = 512
        self.encoder_type = "EBT_GRC"
        self.model_name = "(EBT-GRC-512)"

class EBT_GRC512_noslice_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.stochastic = True
        self.beam_size = 5
        self.hidden_size = 512
        self.encoder_type = "EBT_GRC_noslice"
        self.model_name = "(EBT-GRC-512-noslice)"


class HEBT_GRC_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.norm = "skip"
        self.s4_dropout = self.dropout
        self.pre_SSM = True
        self.rba_temp = 1
        self.prenorm = False
        self.save_by = "accuracy"
        self.metric_direction = 1
        self.RBA = True
        self.RBA_random = False
        self.RBA_advanced = False
        self.stochastic = True
        self.beam_size = 7
        self.model_chunk_size = 30
        self.encoder_type = "HEBT_GRC"
        self.model_name = "(HEBT_GRC)"

class HGRC_config(HEBT_GRC_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "HGRC"
        self.model_name = "(HGRC)"


class MEGA_config(BiRecurrentGRC_config):
    def __init__(self):
        super().__init__()
        self.optimizer = "AdamW"
        self.weight_decay = 0.01
        self.save_by = "accuracy"
        self.classifier_mlp = False
        self.metric_direction = 1
        self.epochs = 60
        self.early_stop_patience = 100
        self.scheduler = "linearWarmup"
        self.max_grad_norm = 1
        self.batch_size = 64
        self.train_batch_size = 64
        self.dev_batch_size = 64
        self.bucket_size_factor = 1
        self.warmup_steps = 3000
        self.warmup_init_lr = 1e-7
        self.end_lr = 0
        self.training_steps = 60 * 1500
        self.lr = 1e-3
        self.betas = (0.99, 0.98)
        self.eps = 1e-8
        self.warmup_power = 1

        self.embd_dim = 80
        self.input_size = 80
        self.hidden_size = 80

        self.hidden_dim = 160
        self.ffn_hidden_dim = 160
        self.embedding_dim = 80
        self.num_encoder_layers = 6
        self.z_dim = 64
        self.n_dim = 16
        self.normalize_before = False
        self.dropout = 0.1
        self.attention_dropout = 0
        self.hidden_dropout = 0
        self.feature_dropout = 0
        self.norm_type = "layernorm"

        self.encoder_type = "MEGA"
        self.model_name = "(MEGA)"