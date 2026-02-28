"""
Base metric class for evaluation metrics.

This module provides the abstract base class that all metrics inherit from.
"""

from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    All metric classes should inherit from this class and implement
    the compute_metric method.

    Attributes:
        verbose (bool): Whether to print progress/debug information
    """

    def __init__(self, verbose=False):
        """
        Initialize the metric.

        Args:
            verbose (bool): If True, print progress information during computation
        """
        self.verbose = verbose

    @abstractmethod
    def compute_metric(self, texts, references=None, **kwargs):
        """
        Compute the metric for given texts.

        Args:
            texts (list): List of generated texts to evaluate
            references (list, optional): List of reference texts for comparison
            **kwargs: Additional metric-specific arguments

        Returns:
            list: List of metric scores corresponding to each input text
        """
        pass
