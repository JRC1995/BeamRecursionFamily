import torch as T
import torch
import math
import warnings

from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


#@register_lr_scheduler('linear_decay')
class LinearDecaySchedule(FairseqLRScheduler):
    """Decay the LR on a linear schedule.
    """

    def __init__(self, config, optimizer):
        super().__init__(config, optimizer)
        warmup_end_lr = config["lr"]
        if config["warmup_steps"] < 0:
            raise ValueError('warm up steps cannot be negative.')
        elif config["warmup_steps"] == 0:
            assert config["warmup_init_lr"] < 0
            config["warmup_init_lr"] = warmup_end_lr
        else:
            assert config["warmup_init_lr"] < warmup_end_lr
            if config["warmup_init_lr"] < 0:
                config["warmup_init_lr"] = 0

        # linearly warmup for the first args.warmup_updates
        if config["warmup_steps"] > 0:
            self.warmup_power = config["warmup_power"]
            self.warmup_factor = (warmup_end_lr - config["warmup_init_lr"]) / (config["warmup_steps"] ** self.warmup_power)
            self.lr = config["warmup_init_lr"]
        else:
            self.warmup_power = 1
            self.warmup_factor = 0
            self.lr = warmup_end_lr

        self.end_learning_rate = config["end_lr"]
        self.total_num_update = config["training_steps"]
        self.lr_factor = (warmup_end_lr - self.end_learning_rate) / (self.total_num_update - config["warmup_steps"])

        # initial learning rate

        for i, g in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[i]['lr'] = self.lr

        self.warmup_updates = config["warmup_steps"]
        self.warmup_init_lr = config["warmup_init_lr"]
        

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-power', default=1, type=int, metavar='N', help='the power of warmup')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--end-learning-rate', default=0.0, type=float)
        parser.add_argument('--total-num-update', default=1000000, type=int)

    def state_dict(self):
        return {'lr': self.lr}

    def load_state_dict(self, state_dict):
        if 'lr' in state_dict:
            self.lr = state_dict['lr']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates <= self.warmup_updates:
            self.lr = self.warmup_init_lr + (num_updates ** self.warmup_power) * self.warmup_factor
        elif num_updates >= self.total_num_update:
            self.lr = self.end_learning_rate
        else:
            self.lr = self.lr_factor * (self.total_num_update - num_updates) + self.end_learning_rate

        for i, g in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[i]['lr'] = self.lr

        return self.lr


class CosineWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == self.warmup_step:  # also covers the case where both are 0
            return self.base_lrs
        elif self.last_epoch < self.warmup_step:
            return [base_lr * (self.last_epoch + 1) / self.warmup_step for base_lr in self.base_lrs]
        elif (self.last_epoch - self.warmup_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_step) / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_step - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    _get_closed_form_lr = None



def get_scheduler(config, optimizer):
    if config["scheduler"] is None:
        return None, False
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=config["scheduler_reduce_factor"],
                                                           patience=config["scheduler_patience"])
        return scheduler, True
    elif config["scheduler"].lower() == "cosine":
        scheduler = CosineWarmup(optimizer,
                                 T_max=config["training_steps"],
                                 warmup_step=config["warmup_steps"],
                                 eta_min=1e-6)
        return scheduler, False
    elif config["scheduler"].lower() == "linearwarmup":
        scheduler = LinearDecaySchedule(config, optimizer)
        return scheduler, False
