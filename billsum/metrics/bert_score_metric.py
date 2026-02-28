"""
BERTScore metric for semantic similarity evaluation.

This module provides the BERTScoreMetric class for computing semantic
similarity between generated summaries and reference summaries.
"""

import torch
from bert_score import score
from .base_metric import BaseMetric


class BERTScoreMetric(BaseMetric):
    """
    BERTScore metric for evaluating semantic similarity.

    Uses DeBERTa-XLarge-MNLI model to compute contextualized embeddings
    and measures semantic similarity between generated and reference texts.

    Attributes:
        model_type (str): HuggingFace model to use for BERTScore
        batch_size (int): Batch size for processing
        device (str): Device to run computation on
    """

    def __init__(
        self,
        model_type="microsoft/deberta-xlarge-mnli",
        batch_size=16,
        device="cuda:0",
        verbose=False,
    ):
        """
        Initialize BERTScore metric.

        Args:
            model_type (str): HuggingFace model to use for embeddings
            batch_size (int): Batch size for processing
            device (str): Device to run on ('cuda:0', 'cpu', etc.)
            verbose (bool): If True, print progress information
        """
        super().__init__(verbose)
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device

    def compute_metric(self, texts, references=None, **kwargs):
        """
        Compute BERTScore F1 for given texts against references.

        Args:
            texts (list): List of generated texts to evaluate
            references (list): List of reference summaries
            **kwargs: Additional arguments (unused)

        Returns:
            list: List of BERTScore F1 scores (floats) for each text

        Raises:
            ValueError: If references are not provided
        """
        if references is None:
            raise ValueError("BERTScore requires reference texts")

        if len(texts) != len(references):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of references ({len(references)})"
            )

        _, _, f1_scores = score(
            texts,
            references,
            model_type=self.model_type,
            lang="en",
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
        )

        return f1_scores.tolist()
