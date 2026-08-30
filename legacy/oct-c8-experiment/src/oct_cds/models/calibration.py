"""Temperature scaling. Fit on the validation split AFTER training; the CDS
layer consumes calibrated probabilities only."""

from __future__ import annotations

from pathlib import Path


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits, labels, max_iter: int = 100):
        import torch

        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)
        nll = torch.nn.CrossEntropyLoss()

        def _closure():
            opt.zero_grad()
            loss = nll(logits / log_t.exp(), labels)
            loss.backward()
            return loss

        opt.step(_closure)
        self.temperature = float(log_t.exp().item())
        return self

    def transform(self, logits):
        import torch

        t = torch.as_tensor(logits, dtype=torch.float32) / self.temperature
        return torch.softmax(t, dim=-1)

    # -- io -----------------------------------------------------------
    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'{{"temperature": {self.temperature}}}', encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        import json

        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["temperature"])
