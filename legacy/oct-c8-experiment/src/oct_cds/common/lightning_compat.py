"""Import shim for PyTorch Lightning.

Works with either distribution:

* ``lightning``           (>= 2.0, the unified package)  -> ``lightning.pytorch``
* ``pytorch_lightning``   (the standalone package)       -> ``pytorch_lightning``

Import ``pl`` and the callbacks from here instead of hard-coding one name.
``HAS_LIGHTNING`` is False when neither is installed (lets tests import modules
that only *optionally* use Lightning).
"""

from __future__ import annotations

pl = None
HAS_LIGHTNING = False
ModelCheckpoint = EarlyStopping = LearningRateMonitor = None

try:
    import lightning.pytorch as pl  # type: ignore
    from lightning.pytorch.callbacks import (  # type: ignore
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    HAS_LIGHTNING = True
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl  # type: ignore
        from pytorch_lightning.callbacks import (  # type: ignore
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )

        HAS_LIGHTNING = True
    except ModuleNotFoundError:
        pass


def require_lightning() -> "pl":  # noqa: F821 - string annotation on purpose
    if not HAS_LIGHTNING:
        raise ModuleNotFoundError(
            "PyTorch Lightning is not installed. Install the project with its "
            "deps (`pip install -e .`) or `pip install lightning`."
        )
    return pl
