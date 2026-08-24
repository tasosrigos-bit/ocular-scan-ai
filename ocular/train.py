"""Training loop for the ocular project.

A single :func:`train` function fits a model on a training loader while tracking
validation accuracy each epoch. It is deliberately small, because model selection
and the richer evaluation metrics live in :mod:`ocular.eval` rather than here.

Kermany validation accuracy is not a reliable guide to transfer, so the loop runs
for a fixed number of epochs and returns the final model rather than selecting a
checkpoint on validation. The per-epoch validation accuracy is returned in the
history only for monitoring.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> torch.device:
    """Return the best available torch device.

    Returns
    -------
    torch.device
        ``cuda`` if available, otherwise ``mps`` on Apple silicon, otherwise
        ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _val_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return the top-1 accuracy of a model over a loader."""
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(1).cpu()
        correct += int((pred == yb).sum())
        total += len(yb)
    return correct / max(total, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    weights: torch.Tensor | None = None,
    epochs: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    ckpt: str | Path | None = None,
    progress: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Fit a model and return it together with a training history.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train. It is moved to ``device`` in place.
    train_loader : torch.utils.data.DataLoader
        Training loader.
    val_loader : torch.utils.data.DataLoader, optional
        Validation loader used to record accuracy each epoch. No checkpoint is
        selected from it.
    weights : torch.Tensor, optional
        Per-class weights for the cross-entropy loss, addressing the class
        imbalance.
    epochs : int, optional
        Number of training epochs.
    lr : float, optional
        Learning rate for the AdamW optimiser.
    weight_decay : float, optional
        Weight decay for the optimiser.
    device : torch.device, optional
        Device to train on. Defaults to :func:`get_device`.
    ckpt : str or pathlib.Path, optional
        If given, the final model state dictionary is written here.
    progress : bool, optional
        Whether to show a per-epoch progress bar.

    Returns
    -------
    tuple of (torch.nn.Module, list of dict)
        The trained model and a list of per-epoch records, each with the keys
        ``epoch``, ``train_loss``, and ``val_acc``.
    """
    device = device or get_device()
    model = model.to(device)
    weight = weights.to(device) if weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = train_loader
        if progress:
            batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)

        train_loss = running / len(train_loader.dataset)
        val_acc = _val_accuracy(model, val_loader, device) if val_loader is not None else None
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        if progress:
            msg = f"epoch {epoch}/{epochs}  train_loss {train_loss:.4f}"
            if val_acc is not None:
                msg += f"  val_acc {val_acc:.4f}"
            print(msg)

    if ckpt is not None:
        ckpt = Path(ckpt)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
    return model, history
