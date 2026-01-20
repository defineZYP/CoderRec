from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from functools import partial
import math

class InverseSquareRootScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super(InverseSquareRootScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return self.base_lrs
        scale_factor = (self.warmup_steps ** 0.5) / (step ** 0.5)
        return [base_lr * scale_factor for base_lr in self.base_lrs]

def fn_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        multipler = 0.5 * (math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1)
        return max(multipler, multipler_min)

def Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps, multipler_min)

def LinearWarmupCosineDecayScheduler(optimizer, warmup_steps, max_steps, min_learning_rate, learning_rate):
    return LambdaLR(
        optimizer,
        Scheduler_LinearWarmup_CosineDecay(
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            multipler_min=min_learning_rate / learning_rate
        )
    )

if __name__ == "__main__":
    warmup_steps = 10000
    base_lr = 0.01
    for step in range(1, 100):
        print((warmup_steps ** 0.5) / (step ** 0.5) * base_lr)