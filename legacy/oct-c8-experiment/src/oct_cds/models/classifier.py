"""LightningModule wrapping a timm backbone for 8-class OCT classification."""

from __future__ import annotations

from typing import Any

from oct_cds.data.label_map import load_label_map
from oct_cds.models.backbones import (
    build_backbone,
    set_backbone_frozen,
    split_param_groups,
)
from oct_cds.models.losses import build_loss


# allow `import` without lightning for lightweight tests (either the
# `lightning` or `pytorch_lightning` distribution works)
from oct_cds.common.lightning_compat import pl as _pl

_Base = _pl.LightningModule if _pl is not None else object


class OCTClassifier(_Base):
    def __init__(self, model_cfg: Any, training_cfg: Any, class_weights=None):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.training_cfg = dict(training_cfg)
        self.num_classes = int(model_cfg["num_classes"])
        self.label_map = load_label_map()

        self.net = build_backbone(model_cfg)
        self.criterion = build_loss(training_cfg.get("loss", {}), class_weights)

        self._freeze_epochs = int(training_cfg.get("freeze_epochs", 0))
        if model_cfg.get("freeze_backbone_initially") and self._freeze_epochs > 0:
            set_backbone_frozen(self.net, True)

        self._build_metrics()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["class_weights"])

    # -- metrics ------------------------------------------------------
    def _build_metrics(self):
        try:
            import torchmetrics as tm
        except ImportError:  # pragma: no cover
            self._tm = None
            return
        mk = lambda: tm.MetricCollection(  # noqa: E731
            {
                "acc": tm.Accuracy(task="multiclass", num_classes=self.num_classes),
                "macro_f1": tm.F1Score(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                "auroc": tm.AUROC(task="multiclass", num_classes=self.num_classes),
            }
        )
        self._tm = {"val": mk(), "test": mk()}

    # -- forward ----------------------------------------------------
    def forward(self, x):
        return self.net(x)

    # -- steps ----------------------------------------------------
    def _step(self, batch, stage: str):
        import torch

        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"))
        if self._tm and stage in self._tm:
            probs = torch.softmax(logits, dim=1)
            self._tm[stage].to(logits.device).update(probs, y)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def _epoch_end(self, stage: str):
        if self._tm and stage in self._tm:
            out = self._tm[stage].compute()
            self.log_dict({f"{stage}/{k}": v for k, v in out.items()}, prog_bar=True)
            self._tm[stage].reset()

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def on_train_epoch_start(self):
        # `>=` (not `==`) so a run resumed from a checkpoint past the warmup
        # epoch still unfreezes the backbone. Idempotent.
        if self._freeze_epochs and self.current_epoch >= self._freeze_epochs:
            set_backbone_frozen(self.net, False)

    # -- optim ----------------------------------------------------
    def configure_optimizers(self):
        import torch

        opt_cfg = self.training_cfg.get("optimizer", {})
        base_lr = float(opt_cfg.get("lr", 3e-4))
        groups = split_param_groups(
            self.net, base_lr, float(self.model_cfg.get("head_lr_mult", 1.0))
        )
        optim = torch.optim.AdamW(
            groups, lr=base_lr, weight_decay=float(opt_cfg.get("weight_decay", 1e-4))
        )
        sched_cfg = self.training_cfg.get("scheduler", {})
        if sched_cfg.get("name") == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=int(self.training_cfg.get("max_epochs", 30)),
                eta_min=float(sched_cfg.get("min_lr", 1e-6)),
            )
            return {"optimizer": optim, "lr_scheduler": sched}
        return optim
