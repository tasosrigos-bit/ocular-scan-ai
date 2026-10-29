"""The assistant's tool and the classifier the application runs on an uploaded scan.

``search_corpus`` is the one tool the language model can call, built as a LangChain
tool that LangGraph can drive. It runs the selected retrieval configuration over an
index and returns the passages it found. It is created by a factory bound to an
index, so the same code serves the evaluation and the application, and takes an
optional collector that records every passage it returns, so a caller can recover
the contexts the model used.

:func:`predict_scan` runs the trained OCT classifier on one B-scan. The application
shows its prediction next to the scan, and the assistant never sees it.
"""
from __future__ import annotations

import numpy as np
import torch
from langchain_core.tools import BaseTool, tool

from ocular import config
from ocular.classifier import model as cnn
from ocular.classifier import train
from ocular.classifier.data import OCTDataset, PreConfig
from ocular.rag import rerank, retrieve
from ocular.rag.index import Index
from ocular.rag.retrieve import Hit

# ---------------------------------------------------------------------------
# search_corpus
# ---------------------------------------------------------------------------


def format_passages(hits: list[Hit]) -> str:
    """Number the retrieved passages and label each with its source."""
    blocks = []
    for i, h in enumerate(hits, 1):
        source = f"{h.chunk.title} - {h.chunk.section}" if h.chunk.section else h.chunk.title
        blocks.append(f"[{i}] ({source})\n{h.chunk.text}")
    return "\n\n".join(blocks) if blocks else "No relevant passages found."


def make_search_corpus(
    idx: Index,
    *,
    mode: str = "hybrid",
    k: int = 5,
    candidates: int = 50,
    rerank_model: str | None = rerank.DEFAULT_RERANKER,
    collector: list[Hit] | None = None,
) -> BaseTool:
    """Build a ``search_corpus`` tool bound to an index and retrieval settings.

    Parameters
    ----------
    idx : Index
        The index to search.
    mode, k, candidates, rerank_model
        The retrieval configuration. ``rerank_model`` of ``None`` skips reranking.
    collector : list of Hit, optional
        If given, every passage the tool returns is appended here.

    Returns
    -------
    langchain_core.tools.BaseTool
        The ``search_corpus`` tool.
    """

    @tool
    def search_corpus(query: str) -> str:
        """Search the ophthalmology literature for passages relevant to a query.

        Use this to find evidence before answering a clinical question. Returns
        numbered passages, each with its source; cite them by number in the answer.
        """
        hits = retrieve.retrieve(idx, query, k=candidates, mode=mode, candidates=candidates)
        if rerank_model:
            hits = rerank.rerank(query, hits, model_name=rerank_model, k=k)
        else:
            hits = hits[:k]
        if collector is not None:
            collector.extend(hits)
        return format_passages(hits)

    return search_corpus


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

# The preprocessing selected for the final model, and the model of record. Public
# because the application reads it too, so the scan that is explained is always the
# same one that was classified.
PREPROCESS = PreConfig(384, 256, crop=True, curvature=True)
DEFAULT_CKPT = config.CKPT

_CLASSIFIERS: dict[str, tuple] = {}


def load_classifier(ckpt=DEFAULT_CKPT, arch: str = "convnext_tiny", device=None):
    """Load the trained classifier once and cache it, returning it with its device."""
    device = device or train.get_device()
    key = f"{ckpt}:{device}"
    if key not in _CLASSIFIERS:
        net = cnn.build_model(arch, pretrained=False)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        _CLASSIFIERS[key] = (net.to(device).eval(), device)
    return _CLASSIFIERS[key]


def predict_scan(path, *, ckpt=DEFAULT_CKPT, device=None) -> tuple[str, float, dict[str, float]]:
    """Classify one OCT B-scan file.

    The image is preprocessed and normalised exactly as during training, so the
    input the model sees at inference matches what it was trained on.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the B-scan image.

    Returns
    -------
    tuple
        The predicted class, its probability, and the full probability per class.
    """
    net, device = load_classifier(ckpt=ckpt, device=device)
    img = (PREPROCESS.apply(path) * 255).astype(np.uint8)
    tensor, _ = OCTDataset(img[None], [0], augment=False)[0]
    with torch.no_grad():
        probs = torch.softmax(net(tensor.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return config.CLASSES[idx], float(probs[idx]), dict(zip(config.CLASSES, map(float, probs)))
