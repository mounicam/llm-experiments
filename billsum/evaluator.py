import sys
import numpy as np
from tqdm import tqdm
from bert_score import score
from prompts import CEFR_LABELS
from transformers import pipeline


class Evaluator:

    def __init__(self, verbose):
        self.classifier = pipeline(
            "text-classification",
            model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
            device=0,
            batch_size=16,
            top_k=None,
        )
        self.verbose = verbose

    def _flatten_dataset(self, dataset):
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

        print(f"Scoring {len(all_texts)} total candidates...")
        cefr_probs_flat = list(tqdm(self.classifier(all_texts), total=len(all_texts))) 
        
        _, _, f1_flat = score(
            all_texts,
            all_references,
            model_type="roberta-large", #"microsoft/deberta-xlarge-mnli",
            lang="en",
            batch_size=16,
            device="cuda:0",
            verbose=True,
        )
        bert_scores_flat = f1_flat.tolist()

        return cefr_probs_flat, bert_scores_flat

    def compute_metrics(self, dataset):

        all_texts, all_references, metadata = self._flatten_dataset(dataset)
        cefr_probs_flat, bert_scores_flat = self._run_metrics(all_texts, all_references)

        for idx, (item_idx, label, cand_idx) in enumerate(metadata):
            # Extract CEFR prob for the specific label we requested
            probs_dict = {p["label"]: p["score"] for p in cefr_probs_flat[idx]}
            c_score = probs_dict.get(label, 0.0)

            # Store metrics
            dataset[item_idx]["predictions"][label][cand_idx]["cefr_prob"] = c_score
            dataset[item_idx]["predictions"][label][cand_idx]["bert_score"] = (
                bert_scores_flat[idx]
            )

        if self.verbose:
            self.print_metrics(dataset)

    def print_metrics(self, dataset):
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
