from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        trainer.optimizer.zero_grad()
        # print(trainer.outputs["loss"])
        trainer.outputs["loss"].backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        trainer.optimizer.step()


class EMAUpdateHook(Hook):
    def __init__(self, alpha=0.999):
        # alpha: ema decay
        self.alpha = alpha

    def after_train_iter(self, trainer):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (trainer.global_step + 1), self.alpha)
        for ema_param, param in zip(trainer.ema_model.parameters(), trainer.model.parameters()):
            # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
