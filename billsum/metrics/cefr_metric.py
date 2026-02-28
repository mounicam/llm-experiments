"""
CEFR metric for readability level classification.

This module provides the CEFRMetric class for computing readability
level scores based on the Common European Framework of Reference (CEFR).
"""

import torch
from tqdm import tqdm
from transformers import pipeline
from .base_metric import BaseMetric

CEFR_LABELS = {"beginner": "A", "intermediate": "B", "advanced": "C"}


class CEFRMetric(BaseMetric):
    """
    CEFR metric for evaluating readability level.

    Uses a fine-tuned BERT classifier to predict CEFR reading levels
    and returns probability scores for target levels.

    Attributes:
        classifier: HuggingFace pipeline for CEFR classification
        batch_size (int): Batch size for processing
    """

    def __init__(
        self,
        model_name="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
        batch_size=16,
        device=0,
        verbose=False,
    ):
        """
        Initialize CEFR metric.

        Args:
            model_name (str): HuggingFace model for CEFR classification
            batch_size (int): Batch size for processing
            device (int): Device ID to run on (0 for cuda:0, -1 for CPU)
            verbose (bool): If True, print progress information
        """
        super().__init__(verbose)
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            batch_size=batch_size,
            top_k=None,
            torch_dtype=torch.float16,
        )
        self.batch_size = batch_size

    def compute_metric(self, texts, references=None, target_labels=None, **kwargs):
        """
        Compute CEFR probability scores for given texts.

        For each text, computes the probability that it matches the target
        CEFR level. The score is the sum of probabilities for level1 and level2
        (e.g., for target "A", returns P(A1) + P(A2)).

        Args:
            texts (list): List of generated texts to evaluate
            references (list, optional): Not used for CEFR metric
            target_labels (list, optional): List of target CEFR labels (e.g., ["A", "B", "A"])
                If None, returns full probability distributions
            **kwargs: Additional arguments (unused)

        Returns:
            list: If target_labels provided, list of probability scores (floats)
                  If target_labels is None, list of probability distributions (dicts)
        """
        if self.verbose:
            print(f"Computing CEFR scores for {len(texts)} texts...")
            predictions = list(tqdm(self.classifier(texts), total=len(texts)))
        else:
            predictions = list(self.classifier(texts))

        # If no target labels specified, return full distributions
        if target_labels is None:
            return predictions

        # Otherwise, extract probability for target label
        if len(texts) != len(target_labels):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of target_labels ({len(target_labels)})"
            )

        scores = []
        for pred, target_label in zip(predictions, target_labels):
            probs_dict = {p["label"]: p["score"] for p in pred}
            # Sum probabilities for level1 and level2 (e.g., A1 + A2)
            target_label = CEFR_LABELS[target_label]
            score = probs_dict.get(target_label + "1", 0.0) + probs_dict.get(
                target_label + "2", 0.0
            )
            scores.append(score)

        return scores
