"""
Evaluation module for computing metrics on generated summaries.

This module provides the Evaluator class for computing CEFR classification
scores, BERTScore, and FKGL metrics on generated text summaries. These metrics
are used as rewards for reinforcement learning (DPO/GRPO) training.
"""

import gc
import torch
import numpy as np
from prompts import READABILTIY_LABELS
from metrics import BERTScoreMetric, CEFRMetric, FKGLMetric


class Evaluator:
    """
    Evaluator for computing CEFR and BERTScore metrics on generated summaries.

    This class handles batch evaluation of predictions across multiple CEFR
    levels, computing both reading level classification scores and semantic
    similarity scores against reference summaries.

    Attributes:
        cefr_metric (CEFRMetric): Metric for CEFR level classification
        bert_metric (BERTScoreMetric): Metric for semantic similarity
        fkgl_metric (FKGLMetric): Metric for Flesch-Kincaid Grade Level
        verbose (bool): Whether to print metrics after computation
    """

    def __init__(self, verbose):
        """
        Initialize evaluator with metric instances.

        Args:
            verbose (bool): If True, print aggregate metrics after computation
        """
        self.cefr_metric = CEFRMetric(
            model_name="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
            batch_size=16,
            device=0,
            verbose=verbose,
        )
        self.bert_metric = BERTScoreMetric(
            model_type="microsoft/deberta-large-mnli",
            batch_size=2,
            device=0,
            verbose=True,
        )
        self.fkgl_metric = FKGLMetric(verbose=verbose)
        self.verbose = verbose

    def _flatten_dataset(self, dataset):
        """
        Flatten dataset predictions into lists for batch processing.

        Converts nested structure (dataset -> items -> labels -> candidates)
        into flat lists suitable for batch metric computation.

        Args:
            dataset (list): Dataset with predictions structure

        Returns:
            tuple: (all_texts, all_references, metadata) where:
                - all_texts: List of generated texts
                - all_references: List of reference summaries
                - metadata: List of (item_idx, label, candidate_idx) tuples
        """
        all_texts = []
        all_references = []
        metadata = []  # To keep track of (item_index, label, candidate_index)

        for i, item in enumerate(dataset):
            for label in READABILTIY_LABELS:
                preds = item["predictions"][label]
                # Ensure it's a list even if it's a single string by mistake
                if isinstance(preds, str):
                    raise ValueError(
                        f"Predictions for item {i}, label '{label}' must be a list, "
                        f"not a string. Got: {type(preds).__name__}"
                    )

                for k, cand_text in enumerate(preds):
                    all_texts.append(cand_text["generation"])
                    all_references.append(item["summary"])
                    metadata.append((i, label, k))
        return all_texts, all_references, metadata

    def _run_metrics(self, all_texts, all_references, target_labels):
        """
        Compute CEFR, BERTScore, and FKGL metrics for all texts.

        Args:
            all_texts (list): List of generated texts to evaluate
            all_references (list): List of reference summaries
            target_labels (list): List of target CEFR labels for each text

        Returns:
            tuple: (cefr_scores_flat, bert_scores_flat, fkgl_scores_flat) where:
                - cefr_scores_flat: List of CEFR probability scores
                - bert_scores_flat: List of BERTScore F1 scores
                - fkgl_scores_flat: List of FKGL readability scores
        """

        print(f"Scoring {len(all_texts)} total candidates...")

        # Compute CEFR probabilities for target labels
        cefr_scores_flat = self.cefr_metric.compute_metric(
            all_texts, target_labels=target_labels
        )
        del self.cefr_metric.classifier
        gc.collect()
        torch.cuda.empty_cache()

        # Compute BERTScore F1 scores
        bert_scores_flat = self.bert_metric.compute_metric(all_texts, all_references)

        # Compute FKGL readability scores
        fkgl_scores_flat = self.fkgl_metric.compute_metric(all_texts)

        return cefr_scores_flat, bert_scores_flat, fkgl_scores_flat

    def compute_metrics(self, dataset):
        """
        Compute and attach metrics to dataset predictions in-place.

        For each prediction, adds 'cefr_prob', 'bert_score', and 'fkgl_score' fields
        based on the target CEFR level and reference summary.

        Args:
            dataset (list): Dataset with predictions to augment with metrics
        """

        all_texts, all_references, metadata = self._flatten_dataset(dataset)

        # Extract target labels for each text
        target_labels = [label for _, label, _ in metadata]

        cefr_scores_flat, bert_scores_flat, fkgl_scores_flat = self._run_metrics(
            all_texts, all_references, target_labels
        )

        for idx, (item_idx, label, cand_idx) in enumerate(metadata):
            dataset[item_idx]["predictions"][label][cand_idx]["cefr_prob"] = (
                cefr_scores_flat[idx]
            )
            dataset[item_idx]["predictions"][label][cand_idx]["bert_score"] = (
                bert_scores_flat[idx]
            )
            dataset[item_idx]["predictions"][label][cand_idx]["fkgl_score"] = (
                fkgl_scores_flat[idx]
            )

        if self.verbose:
            self.print_metrics(dataset)

    def get_metrics(self, dataset):
        """
        Compute aggregate metrics for each readability level.

        Computes mean and standard deviation for BERTScore, CEFR probability,
        and FKGL score across all predictions for each reading level.

        Args:
            dataset (list): Dataset with computed metrics

        Returns:
            dict: Dictionary containing metrics for each level and overall statistics
        """
        metrics = {}

        # Collect metrics for each level
        all_bscores = []
        all_cefr_scores = []
        all_fkgl_scores = []

        for label in READABILTIY_LABELS:

            bscores = []
            cefr_scores = []
            fkgl_scores = []
            for instance in dataset:
                preds = instance["predictions"][label]
                bscores.extend([pred["bert_score"] for pred in preds])
                cefr_scores.extend([pred["cefr_prob"] for pred in preds])
                fkgl_scores.extend([pred["fkgl_score"] for pred in preds])

            # Calculate statistics
            metrics[label] = {
                "bert_score": {
                    "mean": float(np.mean(bscores) * 100.0),
                    "std": float(np.std(bscores) * 100.0),
                },
                "cefr_prob": {
                    "mean": float(np.mean(cefr_scores) * 100.0),
                    "std": float(np.std(cefr_scores) * 100.0),
                },
                "fkgl": {
                    "mean": float(np.mean(fkgl_scores)),
                    "std": float(np.std(fkgl_scores)),
                },
            }

            # Collect for overall statistics
            all_bscores.extend(bscores)
            all_cefr_scores.extend(cefr_scores)
            all_fkgl_scores.extend(fkgl_scores)

        # Add overall statistics
        metrics["overall"] = {
            "bert_score": {
                "mean": float(np.mean(all_bscores) * 100.0),
                "std": float(np.std(all_bscores) * 100.0),
            },
            "cefr_prob": {
                "mean": float(np.mean(all_cefr_scores) * 100.0),
                "std": float(np.std(all_cefr_scores) * 100.0),
            },
            "fkgl": {
                "mean": float(np.mean(all_fkgl_scores)),
                "std": float(np.std(all_fkgl_scores)),
            },
        }

        return metrics

    def print_metrics(self, dataset):
        """
        Print aggregate metrics for each readability level in a formatted table.

        Computes and displays mean BERTScore, CEFR probability, and FKGL score
        across all predictions for each reading level, along with standard deviations.

        Args:
            dataset (list): Dataset with computed metrics
        """
        # Get metrics data
        metrics = self.get_metrics(dataset)

        print("\n" + "=" * 90)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 90)

        # Print table header
        print(f"\n{'Level':<15} {'BERTScore':<20} {'CEFR Prob':<20} {'FKGL':<20}")
        print(
            f"{'':<15} {'(mean ± std)':<20} {'(mean ± std)':<20} {'(mean ± std)':<20}"
        )
        print("-" * 90)

        # Print metrics for each level
        for label in READABILTIY_LABELS:
            m = metrics[label]
            print(
                f"{label:<15} "
                f"{m['bert_score']['mean']:5.2f} ± {m['bert_score']['std']:4.2f}{'':>6} "
                f"{m['cefr_prob']['mean']:5.2f} ± {m['cefr_prob']['std']:4.2f}{'':>6} "
                f"{m['fkgl']['mean']:5.2f} ± {m['fkgl']['std']:4.2f}"
            )

        # Print overall statistics
        print("-" * 90)
        m = metrics["overall"]
        print(
            f"{'Overall':<15} "
            f"{m['bert_score']['mean']:5.2f} ± {m['bert_score']['std']:4.2f}{'':>6} "
            f"{m['cefr_prob']['mean']:5.2f} ± {m['cefr_prob']['std']:4.2f}{'':>6} "
            f"{m['fkgl']['mean']:5.2f} ± {m['fkgl']['std']:4.2f}"
        )
        print("=" * 90 + "\n")
