import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class CosineDecayRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer,
        first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        last_epoch=-1,
    ):
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha

        super(CosineDecayRestarts, self).__init__(optimizer, last_epoch)

    def _calculate_decayed_lr(self, group_lr):
        completed_fraction = (self._step_count - 1) / self.first_decay_steps

        if not self.t_mul == 1.0:
            i_restart = math.floor(
                math.log(1 - completed_fraction * (1 - self.t_mul))
                / math.log(self.t_mul)
            )
            sum_r = (1.0 - self.t_mul ** i_restart) / (1.0 - self.t_mul)
            completed_fraction = (completed_fraction - sum_r) / self.t_mul ** i_restart
        else:
            i_restart = math.floor(completed_fraction)
            completed_fraction = completed_fraction - i_restart

        m_fac = self.m_mul ** i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + math.cos(math.pi * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha

        return group_lr * decayed

    def get_lr(self):
        return [self._calculate_decayed_lr(base_lr) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [self._calculate_decayed_lr(base_lr) for base_lr in self.base_lrs]
