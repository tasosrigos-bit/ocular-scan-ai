"""Retrieval-augmented generation layer for the ocular project.

This subpackage builds a clinical corpus, indexes it, retrieves over it, and
generates grounded answers with a language model. It sits *on top of* the vision
package:
the agent calls the trained classifier (``ocular.classifier.model`` / ``ocular.classifier.explain``)
and the retriever defined here as tools, and the dependency runs one way only -
``ocular.rag`` imports ``ocular``, never the reverse.
"""
