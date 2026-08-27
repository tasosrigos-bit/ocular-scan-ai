"""Tools the agent can call.

The agentic pipeline gives the language model a set of tools and lets it decide
when to use them. This module builds those tools as LangChain tools that LangGraph
can drive. There are two, and together they are the seam between the retrieval
system and the vision model.

``search_corpus`` runs the selected retrieval configuration over an index and
returns the passages it found. ``classify_scan`` runs the trained OCT classifier
on the scan the user has uploaded and returns its prediction. Both are created by
factories bound to their state, an index for the first and a way to reach the
uploaded image for the second, so the same code serves the evaluation and the
application. The retrieval tool also takes an optional collector that records every
passage it returns, so a caller can recover the contexts the agent used.
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


def _format(hits: list[Hit]) -> str:
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
        return _format(hits)

    return search_corpus


# ---------------------------------------------------------------------------
# classify_scan
# ---------------------------------------------------------------------------

# The preprocessing selected for the final model, and the model of record.
_PREPROCESS = PreConfig(384, 256, crop=True, curvature=True)
DEFAULT_CKPT = config.ROOT / "experiments" / "convnext_final_e1.pt"

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
    img = (_PREPROCESS.apply(path) * 255).astype(np.uint8)
    tensor, _ = OCTDataset(img[None], [0], augment=False)[0]
    with torch.no_grad():
        probs = torch.softmax(net(tensor.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return config.CLASSES[idx], float(probs[idx]), dict(zip(config.CLASSES, map(float, probs)))


def make_classify_scan(get_image_path, *, ckpt=DEFAULT_CKPT, device=None) -> BaseTool:
    """Build a ``classify_scan`` tool bound to a source for the uploaded image.

    Parameters
    ----------
    get_image_path : callable
        Returns the path of the currently uploaded scan, or a falsy value if none
        has been uploaded. The tool takes no argument, so the model does not pass
        image data; it acts on whatever scan the application holds.
    ckpt, device
        The checkpoint and device for the classifier.

    Returns
    -------
    langchain_core.tools.BaseTool
        The ``classify_scan`` tool.
    """

    @tool
    def classify_scan() -> str:
        """Classify the OCT B-scan the user has uploaded into CNV, DME, DRUSEN or NORMAL.

        Call this whenever the question concerns the uploaded scan or image, for
        example to identify the finding or to explain what the scan shows.
        """
        path = get_image_path()
        if not path:
            return "No OCT scan has been uploaded, so there is nothing to classify."
        label, confidence, probs = predict_scan(path, ckpt=ckpt, device=device)
        breakdown = ", ".join(f"{c} {p:.0%}" for c, p in probs.items())
        return (
            f"The uploaded B-scan is classified as {label} with {confidence:.0%} confidence. "
            f"Class probabilities: {breakdown}."
        )

    return classify_scan
