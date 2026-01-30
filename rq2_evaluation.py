"""
RQ2 Evaluation: Clustering Quality Metrics

This module implements evaluation metrics for comparing predicted clusters
against ground truth clusters. Based on SE literature survey, we use the 
most common metrics in software engineering clustering research:

Core Metrics (Required):
- Adjusted Rand Index (ARI): Standard metric in SE clustering papers
- Normalized Mutual Information (NMI): Standard metric in SE clustering papers

Recommended Metrics:
- Cluster Purity: Intuitive and widely used
- V-Measure: Harmonic mean of homogeneity and completeness

References:
- LLM-Guided Crowdsourced Test Report Clustering (IEEE Access 2025)
- Semi-supervised Crowdsourced Test Report Clustering (TSE 2024)
- Contrastive Log Embeddings for Clustering Non-Deterministic Test Failures
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from rq2_clustering import ClusteringResult

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EvaluationResult:
    """
    Evaluation result for a clustering method.
    
    Contains only the essential metrics based on SE literature survey:
    - ARI and NMI: Core metrics used in almost all SE clustering papers
    - Purity: Widely used supplementary metric
    - V-Measure: Comprehensive metric combining homogeneity and completeness
    """
    method_name: str
    
    # Core metrics (Required - used in almost all SE clustering papers)
    ari: float = 0.0                    # Adjusted Rand Index [-1, 1]
    nmi: float = 0.0                    # Normalized Mutual Information [0, 1]
    
    # Recommended metrics
    purity: float = 0.0                 # Cluster Purity [0, 1]
    v_measure: float = 0.0              # Harmonic mean of homogeneity and completeness [0, 1]
    
    # Cluster statistics
    num_predicted_clusters: int = 0
    num_ground_truth_clusters: int = 0
    num_tests: int = 0
    
    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# LABEL CONVERSION
# ============================================================================

def clustering_to_labels(
    clustering_result: ClusteringResult,
    all_tests: List[str]
) -> List[int]:
    """
    Convert ClusteringResult to a list of cluster labels.
    
    Args:
        clustering_result: The clustering result
        all_tests: Ordered list of all test names
        
    Returns:
        List of cluster labels (one per test)
    """
    labels = []
    for test in all_tests:
        label = clustering_result.test_to_cluster.get(test, -1)
        labels.append(label)
    return labels


def ground_truth_to_labels(
    ground_truth_clusters: List[List[str]],
    all_tests: List[str]
) -> List[int]:
    """
    Convert ground truth clusters to a list of cluster labels.
    
    Args:
        ground_truth_clusters: List of lists, each inner list contains test names in a cluster
        all_tests: Ordered list of all test names
        
    Returns:
        List of cluster labels (one per test)
    """
    test_to_cluster = {}
    for cluster_id, tests in enumerate(ground_truth_clusters):
        for test in tests:
            test_to_cluster[test] = cluster_id
    
    labels = []
    for test in all_tests:
        label = test_to_cluster.get(test, -1)
        labels.append(label)
    return labels


# ============================================================================
# CORE METRICS IMPLEMENTATION
# ============================================================================

def compute_ari(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Compute Adjusted Rand Index.
    
    ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    
    Range: [-1, 1], where 1 = perfect clustering, 0 = random clustering
    """
    try:
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(true_labels, pred_labels)
    except ImportError:
        return _compute_ari_manual(true_labels, pred_labels)


def _compute_ari_manual(true_labels: List[int], pred_labels: List[int]) -> float:
    """Manual implementation of ARI without sklearn."""
    n = len(true_labels)
    if n == 0:
        return 0.0
    
    # Build contingency table
    contingency = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_labels, pred_labels):
        contingency[t][p] += 1
    
    # Compute row sums, column sums
    row_sums = {}
    col_sums = defaultdict(int)
    
    for t, row in contingency.items():
        row_sums[t] = sum(row.values())
        for p, count in row.items():
            col_sums[p] += count
    
    # Compute index
    def comb2(n):
        return n * (n - 1) // 2
    
    sum_comb = sum(comb2(count) for row in contingency.values() for count in row.values())
    sum_row_comb = sum(comb2(s) for s in row_sums.values())
    sum_col_comb = sum(comb2(s) for s in col_sums.values())
    
    n_comb = comb2(n)
    if n_comb == 0:
        return 0.0
    
    expected = sum_row_comb * sum_col_comb / n_comb
    max_index = (sum_row_comb + sum_col_comb) / 2
    
    if max_index == expected:
        return 1.0 if sum_comb == expected else 0.0
    
    return (sum_comb - expected) / (max_index - expected)


def compute_nmi(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Compute Normalized Mutual Information.
    
    NMI = 2 * I(Y; C) / (H(Y) + H(C))
    
    Range: [0, 1], where 1 = perfect clustering
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return normalized_mutual_info_score(true_labels, pred_labels)
    except ImportError:
        return _compute_nmi_manual(true_labels, pred_labels)


def _compute_nmi_manual(true_labels: List[int], pred_labels: List[int]) -> float:
    """Manual implementation of NMI without sklearn."""
    import math
    
    n = len(true_labels)
    if n == 0:
        return 0.0
    
    # Count occurrences
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    joint_counts = defaultdict(lambda: defaultdict(int))
    
    for t, p in zip(true_labels, pred_labels):
        true_counts[t] += 1
        pred_counts[p] += 1
        joint_counts[t][p] += 1
    
    # Compute entropies
    def entropy(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    
    h_true = entropy(true_counts)
    h_pred = entropy(pred_counts)
    
    # Compute mutual information
    mi = 0.0
    for t, t_count in true_counts.items():
        for p, p_count in pred_counts.items():
            joint = joint_counts[t].get(p, 0)
            if joint > 0:
                mi += (joint / n) * math.log((joint * n) / (t_count * p_count))
    
    # Normalize
    if h_true + h_pred == 0:
        return 0.0
    
    return 2 * mi / (h_true + h_pred)


def compute_v_measure(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Compute V-Measure.
    
    V-Measure is the harmonic mean of homogeneity and completeness:
    - Homogeneity: each cluster contains only members of a single class
    - Completeness: all members of a class are assigned to the same cluster
    
    Range: [0, 1], where 1 = perfect clustering
    
    Returns: v_measure score
    """
    try:
        from sklearn.metrics import v_measure_score
        return v_measure_score(true_labels, pred_labels)
    except ImportError:
        _, _, v_measure = _compute_v_measure_manual(true_labels, pred_labels)
        return v_measure


def _compute_v_measure_manual(true_labels: List[int], pred_labels: List[int]) -> Tuple[float, float, float]:
    """Manual implementation of V-Measure without sklearn."""
    import math
    
    n = len(true_labels)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Build contingency table
    contingency = defaultdict(lambda: defaultdict(int))
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    
    for t, p in zip(true_labels, pred_labels):
        contingency[t][p] += 1
        true_counts[t] += 1
        pred_counts[p] += 1
    
    # Compute entropies
    def entropy(counts, total):
        return -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    
    h_true = entropy(true_counts, n)
    h_pred = entropy(pred_counts, n)
    
    # Compute conditional entropies
    h_true_given_pred = 0.0
    for p, p_count in pred_counts.items():
        for t in true_counts:
            joint = contingency[t].get(p, 0)
            if joint > 0:
                h_true_given_pred -= (joint / n) * math.log(joint / p_count)
    
    h_pred_given_true = 0.0
    for t, t_count in true_counts.items():
        for p in pred_counts:
            joint = contingency[t].get(p, 0)
            if joint > 0:
                h_pred_given_true -= (joint / n) * math.log(joint / t_count)
    
    # Compute homogeneity and completeness
    homogeneity = 1.0 - (h_true_given_pred / h_true) if h_true > 0 else 1.0
    completeness = 1.0 - (h_pred_given_true / h_pred) if h_pred > 0 else 1.0
    
    # V-Measure
    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = 2 * homogeneity * completeness / (homogeneity + completeness)
    
    return homogeneity, completeness, v_measure


def compute_purity(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Compute cluster purity.
    
    Purity = (1/N) * sum_k max_j |cluster_k ∩ class_j|
    
    For each predicted cluster, find the most common true label and sum.
    
    Range: [0, 1], where 1 = perfect purity
    """
    n = len(true_labels)
    if n == 0:
        return 0.0
    
    # Group by predicted cluster
    pred_to_true = defaultdict(list)
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        pred_to_true[p].append(t)
    
    # Sum max frequencies
    total_correct = 0
    for true_labels_in_cluster in pred_to_true.values():
        # Count true label frequencies
        freq = defaultdict(int)
        for t in true_labels_in_cluster:
            freq[t] += 1
        # Add max frequency
        total_correct += max(freq.values()) if freq else 0
    
    return total_correct / n


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_clustering(
    predicted: ClusteringResult,
    ground_truth_clusters: List[List[str]],
    all_tests: Optional[List[str]] = None
) -> EvaluationResult:
    """
    Evaluate a clustering result against ground truth.
    
    Args:
        predicted: The predicted clustering result
        ground_truth_clusters: List of lists, each containing test names in a cluster
        all_tests: Optional list of all test names (inferred if not provided)
        
    Returns:
        EvaluationResult with all metrics
    """
    # Get all tests
    if all_tests is None:
        all_tests = list(predicted.test_to_cluster.keys())
        # Add any tests from ground truth not in predicted
        for cluster in ground_truth_clusters:
            for test in cluster:
                if test not in all_tests:
                    all_tests.append(test)
    
    # Convert to labels
    pred_labels = clustering_to_labels(predicted, all_tests)
    true_labels = ground_truth_to_labels(ground_truth_clusters, all_tests)
    
    # Filter out tests not in both
    valid_indices = [
        i for i in range(len(all_tests))
        if pred_labels[i] >= 0 and true_labels[i] >= 0
    ]
    
    if not valid_indices:
        logger.warning("No common tests between predicted and ground truth")
        return EvaluationResult(
            method_name=predicted.method_name,
            metadata={"error": "No common tests"}
        )
    
    pred_labels = [pred_labels[i] for i in valid_indices]
    true_labels = [true_labels[i] for i in valid_indices]
    
    # Compute essential metrics (based on SE literature survey)
    # Core metrics: ARI and NMI
    ari = compute_ari(true_labels, pred_labels)
    nmi = compute_nmi(true_labels, pred_labels)
    
    # Recommended metrics: Purity and V-measure
    purity = compute_purity(true_labels, pred_labels)
    v_measure = compute_v_measure(true_labels, pred_labels)
    
    return EvaluationResult(
        method_name=predicted.method_name,
        ari=ari,
        nmi=nmi,
        purity=purity,
        v_measure=v_measure,
        num_predicted_clusters=predicted.num_clusters,
        num_ground_truth_clusters=len(ground_truth_clusters),
        num_tests=len(valid_indices),
        metadata={
            "num_tests_evaluated": len(valid_indices),
            "num_unique_true_labels": len(set(true_labels)),
            "num_unique_pred_labels": len(set(pred_labels))
        }
    )


def evaluate_all_methods(
    results: Dict[str, ClusteringResult],
    ground_truth_clusters: List[List[str]],
    all_tests: Optional[List[str]] = None
) -> Dict[str, EvaluationResult]:
    """
    Evaluate all clustering methods against ground truth.
    
    Args:
        results: Dict mapping method name to ClusteringResult
        ground_truth_clusters: Ground truth clusters
        all_tests: Optional list of all test names
        
    Returns:
        Dict mapping method name to EvaluationResult
    """
    evaluations = {}
    
    for method_name, clustering in results.items():
        logger.info(f"Evaluating {method_name}...")
        eval_result = evaluate_clustering(clustering, ground_truth_clusters, all_tests)
        evaluations[method_name] = eval_result
    
    return evaluations


# ============================================================================
# RESULT FORMATTING
# ============================================================================

def format_evaluation_table(evaluations: Dict[str, EvaluationResult]) -> str:
    """
    Format evaluation results as a markdown table.
    
    Displays the 4 essential metrics based on SE literature survey:
    - ARI and NMI (core metrics)
    - Purity and V-Measure (recommended metrics)
    
    Args:
        evaluations: Dict mapping method name to EvaluationResult
        
    Returns:
        Markdown table string
    """
    lines = [
        "| Method | ARI | NMI | Purity | V-Measure | #Clusters |",
        "|--------|-----|-----|--------|-----------|-----------|"
    ]
    
    for method_name, result in sorted(evaluations.items()):
        line = (
            f"| {method_name} | "
            f"{result.ari:.3f} | "
            f"{result.nmi:.3f} | "
            f"{result.purity:.3f} | "
            f"{result.v_measure:.3f} | "
            f"{result.num_predicted_clusters} |"
        )
        lines.append(line)
    
    return "\n".join(lines)


def format_detailed_evaluation(result: EvaluationResult) -> str:
    """
    Format a detailed evaluation result.
    
    Args:
        result: EvaluationResult object
        
    Returns:
        Formatted string
    """
    return f"""
## {result.method_name}

### Core Metrics (Standard in SE Clustering Research)
- Adjusted Rand Index (ARI): {result.ari:.4f}
- Normalized Mutual Information (NMI): {result.nmi:.4f}

### Recommended Metrics
- Purity: {result.purity:.4f}
- V-Measure: {result.v_measure:.4f}

### Cluster Statistics
- Number of Predicted Clusters: {result.num_predicted_clusters}
- Number of Ground Truth Clusters: {result.num_ground_truth_clusters}
- Number of Tests Evaluated: {result.num_tests}
"""


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with simple example
    print("Testing Evaluation Metrics:")
    print("=" * 60)
    
    # Ground truth: 3 clusters
    ground_truth = [
        ["test1", "test2", "test3"],
        ["test4", "test5"],
        ["test6", "test7", "test8"]
    ]
    
    all_tests = ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8"]
    
    # Perfect prediction
    from rq2_clustering import ClusteringResult, PredictedCluster
    
    perfect_result = ClusteringResult(
        method_name="Perfect",
        clusters=[
            PredictedCluster(0, ["test1", "test2", "test3"]),
            PredictedCluster(1, ["test4", "test5"]),
            PredictedCluster(2, ["test6", "test7", "test8"])
        ],
        test_to_cluster={
            "test1": 0, "test2": 0, "test3": 0,
            "test4": 1, "test5": 1,
            "test6": 2, "test7": 2, "test8": 2
        }
    )
    
    eval_result = evaluate_clustering(perfect_result, ground_truth, all_tests)
    print(f"\nPerfect Clustering:")
    print(f"  ARI: {eval_result.ari:.4f} (expected: 1.0)")
    print(f"  NMI: {eval_result.nmi:.4f} (expected: 1.0)")
    print(f"  Purity: {eval_result.purity:.4f} (expected: 1.0)")
    print(f"  V-Measure: {eval_result.v_measure:.4f} (expected: 1.0)")
    
    # Random-ish prediction
    random_result = ClusteringResult(
        method_name="Random",
        clusters=[
            PredictedCluster(0, ["test1", "test4", "test7"]),
            PredictedCluster(1, ["test2", "test5", "test8"]),
            PredictedCluster(2, ["test3", "test6"])
        ],
        test_to_cluster={
            "test1": 0, "test4": 0, "test7": 0,
            "test2": 1, "test5": 1, "test8": 1,
            "test3": 2, "test6": 2
        }
    )
    
    eval_result = evaluate_clustering(random_result, ground_truth, all_tests)
    print(f"\nRandom-ish Clustering:")
    print(f"  ARI: {eval_result.ari:.4f}")
    print(f"  NMI: {eval_result.nmi:.4f}")
    print(f"  Purity: {eval_result.purity:.4f}")
    print(f"  V-Measure: {eval_result.v_measure:.4f}")
