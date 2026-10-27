from core import signals
from models.ials import IALS


class CustomTargetIALS(IALS):
    name = "iALS + custom target"

    def __init__(self, factors=64, reg=1.0, iters=15, cg_steps=4, seed=42, **signal_params):
        self.cfg = signals.SignalConfig(**signal_params)
        super().__init__(factors, reg, self.cfg.alpha, iters, cg_steps, seed)

    def signal(self, history):
        rows, cols, p, c, self.diagnostics = signals.build(history, self.cfg)
        return rows, cols, p, c
