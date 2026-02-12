from typing import Optional

import torch
import torch.optim as optim
import torch.nn.functional as F

from ..utils import log
from .step import Step
from .model import Net


class Devign(Step):
    def __init__(
        self,
        path: str,
        device: str,
        model: dict,
        learning_rate: float,
        weight_decay: float,
        loss_lambda: float,
        pos_weight: float,
        scheduler_cfg: Optional[dict] = None,
    ):
        self.path = path
        self.lr = learning_rate
        self.wd = weight_decay
        self.ll = loss_lambda
        self.device = device
        self.pos_weight = torch.tensor([pos_weight], device=device, dtype=torch.float32)
        self.scheduler_cfg = scheduler_cfg or {}
        log.log_info(
            "devign",
            f"LR: {self.lr}; WD: {self.wd}; LL: {self.ll}; POS_W: {pos_weight};",
        )
        _model = Net(**model, device=device)

        def loss_fn(logits, targets):
            probs = torch.sigmoid(logits)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )
            l1 = F.l1_loss(probs, targets)
            return bce + l1 * self.ll

        optimizer = optim.Adam(_model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_cfg.get("factor", 0.5),
            patience=self.scheduler_cfg.get("patience", 2),
            min_lr=self.scheduler_cfg.get("min_lr", 1e-6),
        )

        super().__init__(
            model=_model,
            loss_function=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.count_parameters()

    def load(self):
        self.model.load(self.path)

    def save(self):
        self.model.save(self.path)

    def count_parameters(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")
