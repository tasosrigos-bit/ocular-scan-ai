"""The OCT B-scan classifier.

This subpackage holds the image side of the project: the preprocessing pipeline
(:mod:`~ocular.classifier.preprocess`), the data caches and loaders
(:mod:`~ocular.classifier.data`), the model interface
(:mod:`~ocular.classifier.model`), the training loop
(:mod:`~ocular.classifier.train`), the metrics (:mod:`~ocular.classifier.eval`)
and the Grad-CAM explainability (:mod:`~ocular.classifier.explain`). Paths, the
class vocabulary and the seed live one level up in :mod:`ocular.config`, shared
with the retrieval assistant in :mod:`ocular.rag`.
"""
