"""
Evaluation module for computing metrics on generated summaries.

This module provides the Evaluator class for computing CEFR classification
scores and BERTScore metrics on generated text summaries. These metrics
are used as rewards for reinforcement learning (DPO/GRPO) training.
"""

import sys
import numpy as np
from tqdm import tqdm
from bert_score import score
from prompts import CEFR_LABELS
from transformers import pipeline


class Evaluator:
    """
    Evaluator for computing CEFR and BERTScore metrics on generated summaries.

    This class handles batch evaluation of predictions across multiple CEFR
    levels, computing both reading level classification scores and semantic
    similarity scores against reference summaries.

    Attributes:
        classifier: HuggingFace pipeline for CEFR level classification
        verbose (bool): Whether to print metrics after computation
    """

    def __init__(self, verbose):
        """
        Initialize evaluator with CEFR classifier.

        Args:
            verbose (bool): If True, print aggregate metrics after computation
        """
        self.classifier = pipeline(
            "text-classification",
            model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
            device=0,
            batch_size=16,
            top_k=None,
        )
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
            for label in CEFR_LABELS:
                preds = item["predictions"][label]
                # Ensure it's a list even if it's a single string by mistake
                if isinstance(preds, str):
                    preds = [preds]

                for k, cand_text in enumerate(preds):
                    all_texts.append(cand_text["generation"])
                    all_references.append(item["summary"])
                    metadata.append((i, label, k))
        return all_texts, all_references, metadata

    def _run_metrics(self, all_texts, all_references):
        """
        Compute CEFR and BERTScore metrics for all texts.

        Args:
            all_texts (list): List of generated texts to evaluate
            all_references (list): List of reference summaries

        Returns:
            tuple: (cefr_probs_flat, bert_scores_flat) where:
                - cefr_probs_flat: List of CEFR probability distributions
                - bert_scores_flat: List of BERTScore F1 scores
        """

        print(f"Scoring {len(all_texts)} total candidates...")
        cefr_probs_flat = list(tqdm(self.classifier(all_texts), total=len(all_texts)))

        _, _, f1_flat = score(
            all_texts,
            all_references,
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            batch_size=16,
            device="cuda:0",
            verbose=True,
        )
        bert_scores_flat = f1_flat.tolist()

        return cefr_probs_flat, bert_scores_flat

    def compute_metrics(self, dataset):
        """
        Compute and attach metrics to dataset predictions in-place.

        For each prediction, adds 'cefr_prob' and 'bert_score' fields
        based on the target CEFR level and reference summary.

        Args:
            dataset (list): Dataset with predictions to augment with metrics
        """

        all_texts, all_references, metadata = self._flatten_dataset(dataset)
        cefr_probs_flat, bert_scores_flat = self._run_metrics(all_texts, all_references)

        for idx, (item_idx, label, cand_idx) in enumerate(metadata):
            # Extract CEFR prob for the specific label we requested
            probs_dict = {p["label"]: p["score"] for p in cefr_probs_flat[idx]}
            c_score = probs_dict[label + "1"] + probs_dict[label + "2"]

            # Store metrics
            dataset[item_idx]["predictions"][label][cand_idx]["cefr_prob"] = c_score
            dataset[item_idx]["predictions"][label][cand_idx]["bert_score"] = (
                bert_scores_flat[idx]
            )

        if self.verbose:
            self.print_metrics(dataset)

    def print_metrics(self, dataset):
        """
        Print aggregate metrics for each CEFR level.

        Computes and displays mean BERTScore and CEFR probability
        across all predictions for each reading level.

        Args:
            dataset (list): Dataset with computed metrics
        """
        for label in CEFR_LABELS:
            bscores = []
            cefr_scores = []
            for instance in dataset:
                preds = instance["predictions"][label]
                bscores.extend([pred["bert_score"] for pred in preds])
                cefr_scores.extend([pred["cefr_prob"] for pred in preds])
            final_bscore = round(np.mean(bscores) * 100.0, 2)
            final_cefr = round(np.mean(cefr_scores) * 100.0, 2)
            print(f"Metrics for {label}: BERTScore {final_bscore}, CEFR {final_cefr}")
