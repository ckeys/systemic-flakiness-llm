"""
Evaluator for RQ1 Experiment

This module implements the evaluation metrics for comparing LLM diagnoses
with human-annotated ground truth (Q3 answers).

Evaluation Dimensions (as per RQ1.md):
1. Category Accuracy (Automated): Precision, Recall, F1, Macro-F1
2. Semantic Alignment (Automated): BERTScore, Cosine Similarity
3. Diagnosis Quality (LLM-based): Semantic similarity scoring
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

from llm_client import LLMClient
from config import SIMILARITY_SCALE, ROOT_CAUSE_CATEGORIES

logger = logging.getLogger(__name__)

# Try to import optional dependencies for automated semantic evaluation
SENTENCE_TRANSFORMERS_AVAILABLE = False
BERT_SCORE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, AttributeError, Exception) as e:
    logger.warning(f"sentence-transformers not available: {e}")

try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except (ImportError, AttributeError, Exception) as e:
    logger.warning(f"bert-score not available: {e}")


@dataclass
class EvaluationResult:
    """Result of evaluating a diagnosis against ground truth."""
    cluster_project: str
    cluster_id: int
    method: str  # "individual" or "collective"
    
    # Dimension 1: Category Accuracy
    predicted_category: Optional[str] = None
    actual_category: Optional[str] = None
    category_match: bool = False
    
    # Dimension 2: Semantic Alignment (Automated)
    cosine_similarity: float = 0.0  # Sentence embedding similarity
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0
    bert_score_f1: float = 0.0
    is_semantically_aligned: bool = False  # cosine >= 0.7
    
    # Dimension 3: LLM-based Diagnosis Quality
    similarity_score: int = 0  # 1-5 scale
    similarity_explanation: str = ""
    is_correct: str = "Partial"  # "Yes", "No", or "Partial"
    
    # Specificity score (1-3): 1=symptom-level, 2=intermediate, 3=root-cause-level
    specificity_score: int = 0
    
    # Raw data
    llm_diagnosis: str = ""
    ground_truth: str = ""


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

SEMANTIC_SIMILARITY_PROMPT = """Compare these two root cause diagnoses for flaky tests:

## Diagnosis A (LLM Generated)
{llm_diagnosis}

## Diagnosis B (Human Expert Ground Truth)
{ground_truth}

## Task
Rate the semantic similarity between these diagnoses on a scale of 1-5:

1 = Completely different root causes (e.g., one says "network issue", other says "memory leak")
2 = Related but different specific causes (e.g., both mention network, but different issues)
3 = Same general category, different details (e.g., both identify DNS issues, but different aspects)
4 = Same root cause, minor wording differences (e.g., both identify DNS resolution failure)
5 = Essentially identical diagnosis (same root cause with same level of detail)

Also determine if Diagnosis A correctly identifies the root cause: Yes/No/Partial

Respond in the following format:
SIMILARITY_SCORE: [1-5]
CORRECTNESS: [Yes/No/Partial]
EXPLANATION: [Brief explanation of your rating]"""


CATEGORY_CLASSIFICATION_PROMPT = """Classify the following root cause diagnosis into one of these categories:

Categories:
- Networking: DNS issues, connection problems, socket errors, HTTP failures
- Resource: Memory issues, thread exhaustion, file handle limits
- Configuration: Version mismatch, missing config, environment issues
- External Dependency: Third-party service failures, API issues
- Concurrency: Race conditions, deadlocks, synchronization issues
- Timing: Timeouts, delays, timing-dependent behavior
- Filesystem: File locks, path issues, permission problems
- Other: Doesn't fit other categories

## Diagnosis
{diagnosis}

Respond with just the category name (one of: Networking, Resource, Configuration, External Dependency, Concurrency, Timing, Filesystem, Other)"""


SPECIFICITY_PROMPT = """Rate the specificity of this root cause diagnosis on a scale of 1-3:

## Diagnosis
{diagnosis}

Rating scale:
1 = Symptom-level: Only describes what failed (e.g., "test timed out", "connection failed")
2 = Intermediate: Identifies the general area but lacks specific root cause (e.g., "network issues")
3 = Root-cause-level: Identifies the specific underlying issue (e.g., "DNS server overloaded causing resolution failures")

Respond with just the number (1, 2, or 3)"""


# ============================================================================
# SEMANTIC EVALUATOR (Automated - BERTScore, Cosine Similarity)
# ============================================================================

class SemanticEvaluator:
    """
    Automated semantic similarity evaluation using:
    - Sentence embeddings (Cosine Similarity)
    - BERTScore
    
    Literature support:
    - BERTScore: Zhang et al., ICLR 2020
    - Sentence-BERT: Reimers & Gurevych, EMNLP 2019
    """
    
    def __init__(self):
        """Initialize the semantic evaluator."""
        self._sentence_model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of models."""
        if self._initialized:
            return
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded Sentence-BERT model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load Sentence-BERT model: {e}")
                self._sentence_model = None
        
        self._initialized = True
    
    def compute_cosine_similarity(self, prediction: str, ground_truth: str) -> float:
        """
        Compute cosine similarity between sentence embeddings.
        
        Args:
            prediction: LLM diagnosis
            ground_truth: Human annotation
            
        Returns:
            Cosine similarity score (0-1)
        """
        self._ensure_initialized()
        
        if self._sentence_model is None:
            return 0.0
        
        try:
            embeddings = self._sentence_model.encode([prediction, ground_truth])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Cosine similarity computation failed: {e}")
            return 0.0
    
    def compute_bert_score(
        self, 
        predictions: List[str], 
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore for predictions vs ground truths.
        
        Args:
            predictions: List of LLM diagnoses
            ground_truths: List of human annotations
            
        Returns:
            Dict with precision, recall, f1 means
        """
        if not BERT_SCORE_AVAILABLE:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            P, R, F1 = bert_score_func(
                predictions, ground_truths, 
                lang="en", 
                verbose=False,
                rescale_with_baseline=True
            )
            return {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item()
            }
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def compute_single_bert_score(
        self, 
        prediction: str, 
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Compute BERTScore for a single prediction.
        
        Args:
            prediction: LLM diagnosis
            ground_truth: Human annotation
            
        Returns:
            Dict with precision, recall, f1
        """
        if not BERT_SCORE_AVAILABLE:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            P, R, F1 = bert_score_func(
                [prediction], [ground_truth], 
                lang="en", 
                verbose=False,
                rescale_with_baseline=True
            )
            return {
                "precision": P[0].item(),
                "recall": R[0].item(),
                "f1": F1[0].item()
            }
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# Global semantic evaluator instance (lazy loaded)
_semantic_evaluator = None

def get_semantic_evaluator() -> SemanticEvaluator:
    """Get the global semantic evaluator instance."""
    global _semantic_evaluator
    if _semantic_evaluator is None:
        _semantic_evaluator = SemanticEvaluator()
    return _semantic_evaluator


class Evaluator:
    """
    Evaluator for comparing LLM diagnoses with ground truth.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the evaluator.
        
        Args:
            llm_client: LLM client for semantic evaluation
        """
        self.llm_client = llm_client
    
    def evaluate_semantic_similarity(
        self, 
        llm_diagnosis: str, 
        ground_truth: str
    ) -> tuple[int, str, str]:
        """
        Evaluate semantic similarity between LLM diagnosis and ground truth.
        
        Args:
            llm_diagnosis: LLM-generated diagnosis
            ground_truth: Human-annotated ground truth (Q3 answer)
            
        Returns:
            Tuple of (similarity_score, correctness, explanation)
        """
        prompt = SEMANTIC_SIMILARITY_PROMPT.format(
            llm_diagnosis=llm_diagnosis,
            ground_truth=ground_truth
        )
        
        response = self.llm_client.generate(prompt)
        
        # Parse the response
        similarity_score = 3  # Default
        correctness = "Partial"
        explanation = ""
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("SIMILARITY_SCORE:"):
                try:
                    score_str = line.replace("SIMILARITY_SCORE:", "").strip()
                    similarity_score = int(score_str)
                    similarity_score = max(1, min(5, similarity_score))  # Clamp to 1-5
                except ValueError:
                    pass
            elif line.startswith("CORRECTNESS:"):
                correctness = line.replace("CORRECTNESS:", "").strip()
                if correctness not in ["Yes", "No", "Partial"]:
                    correctness = "Partial"
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
        
        return similarity_score, correctness, explanation
    
    def classify_category(self, diagnosis: str) -> str:
        """
        Classify the root cause category of a diagnosis.
        
        Args:
            diagnosis: Root cause diagnosis text
            
        Returns:
            Category name
        """
        prompt = CATEGORY_CLASSIFICATION_PROMPT.format(diagnosis=diagnosis)
        response = self.llm_client.generate(prompt).strip()
        
        # Validate category
        for category in ROOT_CAUSE_CATEGORIES:
            if category.lower() in response.lower():
                return category
        
        return "Other"
    
    def evaluate_specificity(self, diagnosis: str) -> int:
        """
        Evaluate the specificity level of a diagnosis.
        
        Args:
            diagnosis: Root cause diagnosis text
            
        Returns:
            Specificity score (1-3)
        """
        prompt = SPECIFICITY_PROMPT.format(diagnosis=diagnosis)
        response = self.llm_client.generate(prompt).strip()
        
        try:
            score = int(response[0])  # Get first character
            return max(1, min(3, score))
        except (ValueError, IndexError):
            return 2  # Default to intermediate
    
    def evaluate(
        self,
        llm_diagnosis: str,
        ground_truth: str,
        cluster_project: str,
        cluster_id: int,
        method: str
    ) -> EvaluationResult:
        """
        Perform full evaluation of a diagnosis.
        
        Evaluation Dimensions:
        1. Category Accuracy: Classify and compare categories
        2. Semantic Alignment: BERTScore + Cosine Similarity (automated)
        3. Diagnosis Quality: LLM-based semantic similarity (1-5 scale)
        
        Args:
            llm_diagnosis: LLM-generated diagnosis
            ground_truth: Human-annotated ground truth
            cluster_project: Project name
            cluster_id: Cluster ID
            method: Analysis method ("individual" or "collective")
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating {method} diagnosis for {cluster_project}/cluster{cluster_id}")
        
        # Dimension 1: Category Accuracy
        predicted_category = self.classify_category(llm_diagnosis)
        actual_category = self.classify_category(ground_truth)
        category_match = predicted_category == actual_category
        
        # Dimension 2: Semantic Alignment (Automated)
        semantic_eval = get_semantic_evaluator()
        cosine_sim = semantic_eval.compute_cosine_similarity(llm_diagnosis, ground_truth)
        bert_scores = semantic_eval.compute_single_bert_score(llm_diagnosis, ground_truth)
        is_semantically_aligned = cosine_sim >= 0.7  # Threshold from RQ1.md
        
        # Dimension 3: LLM-based Diagnosis Quality
        similarity_score, correctness, explanation = self.evaluate_semantic_similarity(
            llm_diagnosis, ground_truth
        )
        specificity_score = self.evaluate_specificity(llm_diagnosis)
        
        return EvaluationResult(
            cluster_project=cluster_project,
            cluster_id=cluster_id,
            method=method,
            # Dimension 1
            predicted_category=predicted_category,
            actual_category=actual_category,
            category_match=category_match,
            # Dimension 2
            cosine_similarity=cosine_sim,
            bert_score_precision=bert_scores["precision"],
            bert_score_recall=bert_scores["recall"],
            bert_score_f1=bert_scores["f1"],
            is_semantically_aligned=is_semantically_aligned,
            # Dimension 3
            similarity_score=similarity_score,
            similarity_explanation=explanation,
            is_correct=correctness,
            specificity_score=specificity_score,
            # Raw data
            llm_diagnosis=llm_diagnosis,
            ground_truth=ground_truth
        )


def compute_category_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Compute category classification metrics (Precision, Recall, F1, Macro-F1).
    
    Args:
        results: List of EvaluationResult objects
        
    Returns:
        Dictionary with category metrics
    """
    if not results:
        return {}
    
    # Get all categories
    all_categories = list(ROOT_CAUSE_CATEGORIES) + ["Other"]
    
    # Count per category
    category_metrics = {}
    for category in all_categories:
        tp = sum(1 for r in results if r.predicted_category == category and r.actual_category == category)
        fp = sum(1 for r in results if r.predicted_category == category and r.actual_category != category)
        fn = sum(1 for r in results if r.predicted_category != category and r.actual_category == category)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        category_metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn  # Number of actual instances
        }
    
    # Compute macro-F1
    f1_scores = [m["f1"] for m in category_metrics.values() if m["support"] > 0]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    # Overall accuracy
    accuracy = sum(1 for r in results if r.category_match) / len(results)
    
    return {
        "per_category": category_metrics,
        "macro_f1": macro_f1,
        "accuracy": accuracy
    }


def compute_aggregate_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from a list of evaluation results.
    
    Metrics computed:
    1. Category Accuracy: Precision, Recall, F1, Macro-F1, Accuracy
    2. Semantic Alignment: Mean BERTScore-F1, Mean Cosine Similarity
    3. Diagnosis Quality: Mean Similarity Score (1-5), Correctness Rate
    
    Args:
        results: List of EvaluationResult objects
        
    Returns:
        Dictionary with aggregate metrics
    """
    if not results:
        return {}
    
    # Separate by method
    individual_results = [r for r in results if r.method == "individual"]
    collective_results = [r for r in results if r.method == "collective"]
    
    def compute_for_group(group: List[EvaluationResult], name: str) -> Dict[str, Any]:
        if not group:
            return {}
        
        # Dimension 1: Category Accuracy
        category_metrics = compute_category_metrics(group)
        
        # Dimension 2: Semantic Alignment (Automated)
        cosine_scores = [r.cosine_similarity for r in group]
        bert_f1_scores = [r.bert_score_f1 for r in group]
        semantically_aligned = sum(1 for r in group if r.is_semantically_aligned)
        
        # Dimension 3: Diagnosis Quality (LLM-based)
        similarity_scores = [r.similarity_score for r in group if r.similarity_score > 0]
        correctness_counts = {
            "Yes": sum(1 for r in group if r.is_correct == "Yes"),
            "Partial": sum(1 for r in group if r.is_correct == "Partial"),
            "No": sum(1 for r in group if r.is_correct == "No")
        }
        specificity_scores = [r.specificity_score for r in group if r.specificity_score > 0]
        
        metrics = {
            f"{name}_count": len(group),
            # Dimension 1: Category Accuracy
            f"{name}_category_accuracy": category_metrics.get("accuracy", 0),
            f"{name}_category_macro_f1": category_metrics.get("macro_f1", 0),
            # Dimension 2: Semantic Alignment
            f"{name}_mean_cosine_similarity": np.mean(cosine_scores) if cosine_scores else 0,
            f"{name}_std_cosine_similarity": np.std(cosine_scores) if cosine_scores else 0,
            f"{name}_mean_bert_score_f1": np.mean(bert_f1_scores) if bert_f1_scores else 0,
            f"{name}_std_bert_score_f1": np.std(bert_f1_scores) if bert_f1_scores else 0,
            f"{name}_semantic_alignment_rate": semantically_aligned / len(group),
            # Dimension 3: Diagnosis Quality
            f"{name}_mean_similarity": np.mean(similarity_scores) if similarity_scores else 0,
            f"{name}_std_similarity": np.std(similarity_scores) if similarity_scores else 0,
            f"{name}_median_similarity": np.median(similarity_scores) if similarity_scores else 0,
            f"{name}_correct_rate": correctness_counts["Yes"] / len(group),
            f"{name}_partial_rate": correctness_counts["Partial"] / len(group),
            f"{name}_incorrect_rate": correctness_counts["No"] / len(group),
            f"{name}_mean_specificity": np.mean(specificity_scores) if specificity_scores else 0,
        }
        
        return metrics
    
    metrics = {}
    metrics.update(compute_for_group(individual_results, "individual"))
    metrics.update(compute_for_group(collective_results, "collective"))
    
    # Comparison metrics (Collective vs Individual)
    if individual_results and collective_results:
        # Sort to align by cluster
        ind_sorted = sorted(individual_results, key=lambda x: (x.cluster_project, x.cluster_id))
        col_sorted = sorted(collective_results, key=lambda x: (x.cluster_project, x.cluster_id))
        
        # Similarity score comparison
        ind_mean = metrics.get("individual_mean_similarity", 0)
        col_mean = metrics.get("collective_mean_similarity", 0)
        metrics["similarity_improvement"] = col_mean - ind_mean
        
        # Win/Loss analysis
        metrics["collective_wins_similarity"] = sum(
            1 for i, c in zip(ind_sorted, col_sorted) 
            if c.similarity_score > i.similarity_score
        )
        metrics["individual_wins_similarity"] = sum(
            1 for i, c in zip(ind_sorted, col_sorted) 
            if i.similarity_score > c.similarity_score
        )
        metrics["ties_similarity"] = len(ind_sorted) - metrics["collective_wins_similarity"] - metrics["individual_wins_similarity"]
        
        # Cosine similarity comparison
        metrics["cosine_improvement"] = (
            metrics.get("collective_mean_cosine_similarity", 0) - 
            metrics.get("individual_mean_cosine_similarity", 0)
        )
        
        # BERTScore comparison
        metrics["bert_score_improvement"] = (
            metrics.get("collective_mean_bert_score_f1", 0) - 
            metrics.get("individual_mean_bert_score_f1", 0)
        )
        
        # Category accuracy comparison
        metrics["category_accuracy_improvement"] = (
            metrics.get("collective_category_accuracy", 0) - 
            metrics.get("individual_category_accuracy", 0)
        )
    
    return metrics


def format_results_table(results: List[EvaluationResult]) -> str:
    """
    Format evaluation results as a table.
    
    Args:
        results: List of EvaluationResult objects
        
    Returns:
        Formatted table string
    """
    headers = ["Project", "Cluster", "Method", "Sim", "Cosine", "BERT-F1", "Cat", "Correct"]
    rows = []
    
    for r in sorted(results, key=lambda x: (x.cluster_project, x.cluster_id, x.method)):
        rows.append([
            r.cluster_project[:15],  # Truncate long project names
            str(r.cluster_id),
            r.method[:4],  # "indi" or "coll"
            str(r.similarity_score),
            f"{r.cosine_similarity:.2f}",
            f"{r.bert_score_f1:.2f}",
            "✓" if r.category_match else "✗",
            r.is_correct[:1]  # "Y", "N", or "P"
        ])
    
    # Calculate column widths
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    # Format table
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)
    lines.append(header_line)
    lines.append(separator)
    
    for row in rows:
        lines.append(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    
    return "\n".join(lines)


def format_summary_report(results: List[EvaluationResult], metrics: Dict[str, Any]) -> str:
    """
    Format a comprehensive summary report.
    
    Args:
        results: List of EvaluationResult objects
        metrics: Aggregate metrics dictionary
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("RQ1 EXPERIMENT RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    # Dataset info
    ind_count = metrics.get("individual_count", 0)
    col_count = metrics.get("collective_count", 0)
    lines.append(f"Total Clusters Evaluated: {max(ind_count, col_count)}")
    lines.append("")
    
    # Dimension 1: Category Accuracy
    lines.append("-" * 70)
    lines.append("DIMENSION 1: CATEGORY ACCURACY")
    lines.append("-" * 70)
    lines.append(f"{'Method':<15} {'Accuracy':<12} {'Macro-F1':<12}")
    lines.append(f"{'Individual':<15} {metrics.get('individual_category_accuracy', 0):.3f}        {metrics.get('individual_category_macro_f1', 0):.3f}")
    lines.append(f"{'Collective':<15} {metrics.get('collective_category_accuracy', 0):.3f}        {metrics.get('collective_category_macro_f1', 0):.3f}")
    lines.append(f"{'Improvement':<15} {metrics.get('category_accuracy_improvement', 0):+.3f}")
    lines.append("")
    
    # Dimension 2: Semantic Alignment
    lines.append("-" * 70)
    lines.append("DIMENSION 2: SEMANTIC ALIGNMENT (Automated)")
    lines.append("-" * 70)
    lines.append(f"{'Method':<15} {'Cosine Sim':<15} {'BERTScore-F1':<15} {'Aligned Rate':<12}")
    lines.append(f"{'Individual':<15} {metrics.get('individual_mean_cosine_similarity', 0):.3f} ± {metrics.get('individual_std_cosine_similarity', 0):.3f}    {metrics.get('individual_mean_bert_score_f1', 0):.3f} ± {metrics.get('individual_std_bert_score_f1', 0):.3f}    {metrics.get('individual_semantic_alignment_rate', 0):.3f}")
    lines.append(f"{'Collective':<15} {metrics.get('collective_mean_cosine_similarity', 0):.3f} ± {metrics.get('collective_std_cosine_similarity', 0):.3f}    {metrics.get('collective_mean_bert_score_f1', 0):.3f} ± {metrics.get('collective_std_bert_score_f1', 0):.3f}    {metrics.get('collective_semantic_alignment_rate', 0):.3f}")
    lines.append(f"{'Improvement':<15} {metrics.get('cosine_improvement', 0):+.3f}            {metrics.get('bert_score_improvement', 0):+.3f}")
    lines.append("")
    
    # Dimension 3: Diagnosis Quality
    lines.append("-" * 70)
    lines.append("DIMENSION 3: DIAGNOSIS QUALITY (LLM-based)")
    lines.append("-" * 70)
    lines.append(f"{'Method':<15} {'Similarity':<15} {'Correct':<10} {'Partial':<10} {'Incorrect':<10}")
    lines.append(f"{'Individual':<15} {metrics.get('individual_mean_similarity', 0):.2f} ± {metrics.get('individual_std_similarity', 0):.2f}      {metrics.get('individual_correct_rate', 0):.1%}      {metrics.get('individual_partial_rate', 0):.1%}      {metrics.get('individual_incorrect_rate', 0):.1%}")
    lines.append(f"{'Collective':<15} {metrics.get('collective_mean_similarity', 0):.2f} ± {metrics.get('collective_std_similarity', 0):.2f}      {metrics.get('collective_correct_rate', 0):.1%}      {metrics.get('collective_partial_rate', 0):.1%}      {metrics.get('collective_incorrect_rate', 0):.1%}")
    lines.append(f"{'Improvement':<15} {metrics.get('similarity_improvement', 0):+.2f}")
    lines.append("")
    
    # Win/Loss Analysis
    lines.append("-" * 70)
    lines.append("WIN/LOSS ANALYSIS (Collective vs Individual)")
    lines.append("-" * 70)
    lines.append(f"Collective Wins: {metrics.get('collective_wins_similarity', 0)}")
    lines.append(f"Individual Wins: {metrics.get('individual_wins_similarity', 0)}")
    lines.append(f"Ties: {metrics.get('ties_similarity', 0)}")
    lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test evaluation with sample data
    print("Evaluator module loaded successfully.")
    print(f"\nSimilarity scale:")
    for score, description in SIMILARITY_SCALE.items():
        print(f"  {score}: {description}")
    print(f"\nRoot cause categories: {ROOT_CAUSE_CATEGORIES}")

