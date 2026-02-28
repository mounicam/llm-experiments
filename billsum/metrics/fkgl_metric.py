"""
FKGL metric for readability assessment.

This module provides the FKGLMetric class for computing Flesch-Kincaid
Grade Level scores to assess text readability.
"""

import textstat
from .base_metric import BaseMetric


class FKGLMetric(BaseMetric):
    """
    FKGL (Flesch-Kincaid Grade Level) metric for evaluating text readability.

    Computes the Flesch-Kincaid Grade Level score which estimates the
    U.S. grade level required to understand the text. Lower scores indicate
    easier readability.

    The formula considers:
    - Average sentence length (words per sentence)
    - Average syllables per word

    Score interpretation:
    - 1-6: Elementary school level
    - 7-8: Middle school level
    - 9-12: High school level
    - 13+: College level and above
    """

    def __init__(self, verbose=False):
        """
        Initialize FKGL metric.

        Args:
            verbose (bool): If True, print progress information
        """
        super().__init__(verbose)

    def compute_metric(self, texts, references=None, **kwargs):
        """
        Compute Flesch-Kincaid Grade Level for given texts.

        Args:
            texts (list): List of generated texts to evaluate
            references (list, optional): Not used for FKGL metric
            **kwargs: Additional arguments (unused)

        Returns:
            list: List of FKGL scores (floats) for each text
        """
        if self.verbose:
            print(f"Computing FKGL scores for {len(texts)} texts...")

        scores = []
        for text in texts:
            try:
                score = textstat.flesch_kincaid_grade(text)
                scores.append(score)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to compute FKGL for text: {e}")
                # Default to a neutral score if computation fails
                scores.append(0.0)

        return scores
