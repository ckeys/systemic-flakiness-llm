"""
RQ2 Experiment: Automated Identification of Systemic Flakiness Clusters

This script runs the complete RQ2 experiment:
1. Load data from Systemic Flakiness dataset
2. Extract features using Three-Tier Exception Categorization
3. Run all clustering methods (Our Method + Baselines)
4. Optionally run LLM verification
5. Evaluate against ground truth
6. Generate results report

Usage:
    python run_rq2_experiment.py [--skip-llm] [--skip-verification] [--project PROJECT]
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MANUAL_ANALYSIS_DIR,
    SOURCES_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    PROJECTS_WITH_CLUSTERS,
    LOG_LEVEL,
    LOG_FORMAT
)
from data_loader import (
    load_all_clusters,
    load_cluster,
    discover_clusters,
    Cluster,
    TestCase
)
from llm_client import create_llm_client, LLMClient, get_cost_tracker, reset_cost_tracker
from rq2_feature_extractor import (
    FeatureExtractor,
    TestFeatures,
    get_categorization_statistics
)
from rq2_clustering import (
    ClusteringResult,
    PredictedCluster,
    signature_based_clustering,
    random_clustering,
    test_class_clustering,
    exception_type_clustering,
    embedding_clustering,
    pure_llm_clustering,
    run_all_baselines
)
from rq2_verification import (
    ClusterVerifier,
    verified_to_clustering_result
)
from rq2_evaluation import (
    EvaluationResult,
    evaluate_clustering,
    evaluate_all_methods,
    format_evaluation_table,
    format_detailed_evaluation
)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging for the experiment."""
    logger = logging.getLogger("rq2_experiment")
    logger.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_experiment_data(
    clusters: List[Cluster]
) -> Tuple[List[TestCase], List[List[str]], Dict[str, str], Dict[str, str]]:
    """
    Prepare data for the experiment.
    
    Args:
        clusters: List of Cluster objects from the dataset
        
    Returns:
        Tuple of:
        - all_tests: List of all TestCase objects
        - ground_truth_clusters: List of lists (test names per cluster)
        - stack_traces: Dict mapping test name to stack trace
        - source_codes: Dict mapping test name to source code
    """
    all_tests = []
    ground_truth_clusters = []
    stack_traces = {}
    source_codes = {}
    
    for cluster in clusters:
        cluster_test_names = []
        
        for test in cluster.tests:
            all_tests.append(test)
            cluster_test_names.append(test.name)
            
            # Store stack trace
            if test.stack_traces:
                stack_traces[test.name] = test.stack_traces[0]
            
            # Store source code
            if test.source_code:
                source_codes[test.name] = test.source_code
        
        if cluster_test_names:
            ground_truth_clusters.append(cluster_test_names)
    
    return all_tests, ground_truth_clusters, stack_traces, source_codes


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(
    skip_llm: bool = False,
    skip_verification: bool = False,
    skip_embedding: bool = False,
    project_filter: Optional[str] = None,
    output_dir: Optional[Path] = None,
    num_runs: int = 1
) -> Dict[str, Any]:
    """
    Run the complete RQ2 experiment.
    
    Args:
        skip_llm: Skip LLM-based methods (B5 and verification)
        skip_verification: Skip LLM verification phase
        skip_embedding: Skip embedding-based clustering (requires sentence-transformers)
        project_filter: Only run on specific project
        output_dir: Directory for output files
        num_runs: Number of runs for LLM-based methods (to reduce variance)
        
    Returns:
        Dict with experiment results
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = LOGS_DIR / f"rq2_experiment_{timestamp}.log"
    logger = setup_logging(log_file)
    
    # Reset cost tracker for this experiment
    reset_cost_tracker()
    cost_tracker = get_cost_tracker()
    
    logger.info("=" * 60)
    logger.info("RQ2 Experiment: Automated Identification of Systemic Flakiness Clusters")
    logger.info("=" * 60)
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    logger.info("\n[Step 1] Loading cluster data...")
    
    # For RQ2, we load ALL clusters (45 clusters, 606 tests) to match EMSE paper
    # We don't need Q3 ground truth for clustering evaluation
    clusters = load_all_clusters(require_ground_truth=False)
    
    if project_filter:
        clusters = [c for c in clusters if c.project == project_filter]
        logger.info(f"Filtered to project: {project_filter}")
    
    logger.info(f"Loaded {len(clusters)} clusters")
    
    if not clusters:
        logger.error("No clusters found!")
        return {"error": "No clusters found"}
    
    # Prepare data
    all_tests, ground_truth_clusters, stack_traces, source_codes = prepare_experiment_data(clusters)
    
    logger.info(f"Total tests: {len(all_tests)}")
    logger.info(f"Ground truth clusters: {len(ground_truth_clusters)}")
    logger.info(f"Tests with stack traces: {len(stack_traces)}")
    logger.info(f"Tests with source code: {len(source_codes)}")
    
    # ========================================================================
    # Step 2: Feature Extraction
    # ========================================================================
    logger.info("\n[Step 2] Extracting features (Three-Tier Exception Categorization)...")
    
    # Create LLM client for Tier 3 (optional)
    llm_client = None
    if not skip_llm:
        try:
            llm_client = create_llm_client("openai")
            logger.info("LLM client created for Tier 3 categorization")
        except Exception as e:
            logger.warning(f"Could not create LLM client: {e}")
            logger.warning("Tier 3 will fall back to 'Other' category")
    
    # Extract features
    extractor = FeatureExtractor(llm_client)
    features_list = []
    
    for test in all_tests:
        features = extractor.extract_features(test, test.project)
        features_list.append(features)
    
    # Get categorization statistics
    cat_stats = get_categorization_statistics(features_list)
    logger.info(f"Categorization statistics:")
    logger.info(f"  Tier 1 (exact match): {cat_stats.get('tier1_count', 0)} ({cat_stats.get('tier1_pct', 0):.1f}%)")
    logger.info(f"  Tier 2 (keyword): {cat_stats.get('tier2_count', 0)} ({cat_stats.get('tier2_pct', 0):.1f}%)")
    logger.info(f"  Tier 3 (LLM): {cat_stats.get('tier3_count', 0)} ({cat_stats.get('tier3_pct', 0):.1f}%)")
    logger.info(f"  Category distribution: {cat_stats.get('category_distribution', {})}")
    
    # Build feature map
    features_map = {f.test_name: f for f in features_list}
    
    # ========================================================================
    # Step 3: Run Clustering Methods
    # ========================================================================
    logger.info("\n[Step 3] Running clustering methods...")
    
    test_names = [f.test_name for f in features_list]
    num_gt_clusters = len(ground_truth_clusters)
    
    clustering_results: Dict[str, ClusteringResult] = {}
    
    # B1: Random Clustering
    logger.info("  Running B1: Random Clustering...")
    clustering_results["B1-Random"] = random_clustering(test_names, num_gt_clusters, seed=42)
    
    # B2: Test Class-based Clustering
    logger.info("  Running B2: Test Class-based Clustering...")
    clustering_results["B2-TestClass"] = test_class_clustering(test_names)
    
    # B3: Exception Type-only Clustering
    logger.info("  Running B3: Exception Type-only Clustering...")
    clustering_results["B3-ExceptionType"] = exception_type_clustering(features_list)
    
    # B4: Embedding-based Clustering
    if not skip_embedding:
        logger.info("  Running B4: Embedding-based Clustering...")
        try:
            clustering_results["B4-Embedding"] = embedding_clustering(
                features_list, stack_traces, distance_threshold=0.3  # Lower threshold for better separation
            )
        except Exception as e:
            logger.warning(f"  B4 failed: {e}")
            logger.warning("  Install sentence-transformers: pip install sentence-transformers")
    else:
        logger.info("  Skipping B4: Embedding-based Clustering")
    
    # B5: Pure LLM Clustering
    if not skip_llm and llm_client:
        logger.info("  Running B5: Pure LLM Clustering...")
        try:
            clustering_results["B5-PureLLM"] = pure_llm_clustering(
                features_list, stack_traces, llm_client, max_tests_per_batch=20
            )
        except Exception as e:
            logger.warning(f"  B5 failed: {e}")
    else:
        logger.info("  Skipping B5: Pure LLM Clustering")
    
    # Our Method: Signature-based Clustering (Phase 2)
    logger.info("  Running Our Method: Signature-based Pre-Clustering...")
    clustering_results["Hybrid-Signature"] = signature_based_clustering(features_list)
    
    # Log clustering statistics
    for method_name, result in clustering_results.items():
        logger.info(f"  {method_name}: {result.num_clusters} clusters")
    
    # ========================================================================
    # Step 4: LLM Verification (Optional)
    # ========================================================================
    if not skip_verification and not skip_llm and llm_client:
        logger.info("\n[Step 4] Running LLM Verification (Phase 3)...")
        
        verifier = ClusterVerifier(llm_client, max_tests_per_verification=10)
        
        # Verify signature-based clusters
        verified_clusters, verify_stats = verifier.verify_all_clusters(
            clustering_results["Hybrid-Signature"],
            features_map,
            stack_traces,
            source_codes,
            min_cluster_size=2
        )
        
        # Convert to ClusteringResult
        clustering_results["Hybrid-Verified"] = verified_to_clustering_result(verified_clusters)
        
        logger.info(f"  Verification statistics:")
        logger.info(f"    Total clusters: {verify_stats['total_clusters']}")
        logger.info(f"    Verified Yes: {verify_stats['verified_yes']}")
        logger.info(f"    Verified No: {verify_stats['verified_no']}")
        logger.info(f"    Verified Partial: {verify_stats['verified_partial']}")
        logger.info(f"    Skipped (small): {verify_stats['skipped_small']}")
        logger.info(f"    Splits performed: {verify_stats['splits_performed']}")
    else:
        logger.info("\n[Step 4] Skipping LLM Verification")
    
    # ========================================================================
    # Step 5: Evaluation
    # ========================================================================
    logger.info("\n[Step 5] Evaluating clustering methods...")
    
    evaluations = evaluate_all_methods(
        clustering_results, 
        ground_truth_clusters, 
        test_names
    )
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(format_evaluation_table(evaluations))
    
    # ========================================================================
    # Step 6: Save Results
    # ========================================================================
    logger.info("\n[Step 6] Saving results...")
    
    # Get cost summary
    cost_summary = cost_tracker.get_summary()
    
    # Log cost summary
    if cost_summary["total_calls"] > 0:
        logger.info("\n" + cost_tracker.format_summary())
    
    # Prepare results dict
    results = {
        "timestamp": timestamp,
        "config": {
            "skip_llm": skip_llm,
            "skip_verification": skip_verification,
            "skip_embedding": skip_embedding,
            "project_filter": project_filter
        },
        "data_stats": {
            "num_clusters": len(clusters),
            "num_tests": len(all_tests),
            "num_ground_truth_clusters": len(ground_truth_clusters),
            "tests_with_stack_traces": len(stack_traces),
            "tests_with_source_code": len(source_codes)
        },
        "categorization_stats": cat_stats,
        "clustering_stats": {
            name: {
                "num_clusters": result.num_clusters,
                "num_tests": result.num_tests,
                "cluster_sizes": result.get_cluster_sizes()
            }
            for name, result in clustering_results.items()
        },
        "evaluation_results": {
            name: {
                "ari": eval_result.ari,
                "nmi": eval_result.nmi,
                "purity": eval_result.purity,
                "v_measure": eval_result.v_measure,
                "num_predicted_clusters": eval_result.num_predicted_clusters,
                "num_ground_truth_clusters": eval_result.num_ground_truth_clusters
            }
            for name, eval_result in evaluations.items()
        },
        "llm_cost": cost_summary
    }
    
    # Save JSON results
    results_file = output_dir / f"rq2_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Save summary report
    summary_file = output_dir / f"rq2_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RQ2 Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Clusters: {len(clusters)}\n")
        f.write(f"Total Tests: {len(all_tests)}\n\n")
        
        f.write("Categorization Statistics:\n")
        f.write(f"  Tier 1: {cat_stats.get('tier1_pct', 0):.1f}%\n")
        f.write(f"  Tier 2: {cat_stats.get('tier2_pct', 0):.1f}%\n")
        f.write(f"  Tier 3: {cat_stats.get('tier3_pct', 0):.1f}%\n")
        f.write(f"  Tier 4: {cat_stats.get('tier4_pct', 0):.1f}%\n\n")
        
        f.write("Evaluation Results:\n")
        f.write(format_evaluation_table(evaluations))
        f.write("\n\n")
        
        for name, eval_result in evaluations.items():
            f.write(format_detailed_evaluation(eval_result))
            f.write("\n")
        
        # Add LLM cost summary
        if cost_summary["total_calls"] > 0:
            f.write("\n" + cost_tracker.format_summary() + "\n")
    
    logger.info(f"Summary saved to: {summary_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Experiment completed!")
    logger.info("=" * 60)
    
    return results


# ============================================================================
# MULTI-RUN EXPERIMENT (FOR LLM VARIANCE REDUCTION)
# ============================================================================

def run_multi_run_experiment(
    num_runs: int = 5,
    skip_embedding: bool = False,
    project_filter: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run the experiment multiple times for LLM-based methods to reduce variance.
    
    LLM outputs are non-deterministic, so we need multiple runs to:
    1. Assess stability/variance of results
    2. Report mean and standard deviation
    3. Ensure reproducibility of conclusions
    
    Args:
        num_runs: Number of runs for LLM-based methods
        skip_embedding: Skip embedding-based clustering
        project_filter: Only run on specific project
        output_dir: Directory for output files
        
    Returns:
        Dict with aggregated results including mean, std for each metric
    """
    import numpy as np
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = LOGS_DIR / f"rq2_multirun_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info(f"RQ2 Multi-Run Experiment ({num_runs} runs)")
    logger.info("=" * 60)
    
    # Reset cost tracker
    reset_cost_tracker()
    cost_tracker = get_cost_tracker()
    
    # ========================================================================
    # Step 1: Load Data (once)
    # ========================================================================
    logger.info("\n[Step 1] Loading cluster data...")
    
    # For RQ2, we load ALL clusters (45 clusters, 606 tests) to match EMSE paper
    clusters = load_all_clusters(require_ground_truth=False)
    if project_filter:
        clusters = [c for c in clusters if c.project == project_filter]
    
    all_tests, ground_truth_clusters, stack_traces, source_codes = prepare_experiment_data(clusters)
    logger.info(f"Loaded {len(clusters)} clusters, {len(all_tests)} tests")
    
    # ========================================================================
    # Step 2: Feature Extraction (once - deterministic for Tier 1/2)
    # ========================================================================
    logger.info("\n[Step 2] Extracting features...")
    
    # Create LLM client
    try:
        llm_client = create_llm_client("openai")
    except Exception as e:
        logger.error(f"Could not create LLM client: {e}")
        return {"error": str(e)}
    
    extractor = FeatureExtractor(llm_client)
    features_list = [extractor.extract_features(test, test.project) for test in all_tests]
    features_map = {f.test_name: f for f in features_list}
    test_names = [f.test_name for f in features_list]
    
    cat_stats = get_categorization_statistics(features_list)
    logger.info(f"Categorization: Tier1={cat_stats.get('tier1_pct', 0):.1f}%, "
                f"Tier2={cat_stats.get('tier2_pct', 0):.1f}%, "
                f"Tier3={cat_stats.get('tier3_pct', 0):.1f}%, "
                f"Tier4={cat_stats.get('tier4_pct', 0):.1f}%")
    
    # ========================================================================
    # Step 3: Run Deterministic Methods (once)
    # ========================================================================
    logger.info("\n[Step 3] Running deterministic methods (1 run)...")
    
    deterministic_results = {}
    
    # B1: Random (with fixed seed)
    deterministic_results["B1-Random"] = random_clustering(
        test_names, len(ground_truth_clusters), seed=42
    )
    
    # B2: Test Class
    deterministic_results["B2-TestClass"] = test_class_clustering(test_names)
    
    # B3: Exception Type
    deterministic_results["B3-ExceptionType"] = exception_type_clustering(features_list)
    
    # B4: Embedding (deterministic with same model)
    if not skip_embedding:
        try:
            deterministic_results["B4-Embedding"] = embedding_clustering(
                features_list, stack_traces, distance_threshold=0.3  # Lower threshold for better separation
            )
        except Exception as e:
            logger.warning(f"B4 failed: {e}")
    
    # Our Method (Phase 2 - deterministic)
    deterministic_results["Hybrid-Signature"] = signature_based_clustering(features_list)
    
    # Evaluate deterministic methods
    deterministic_evaluations = evaluate_all_methods(
        deterministic_results, ground_truth_clusters, test_names
    )
    
    # ========================================================================
    # Step 4: Run LLM-based Methods (multiple runs)
    # ========================================================================
    logger.info(f"\n[Step 4] Running LLM-based methods ({num_runs} runs)...")
    
    # Store results from each run
    llm_run_results = {
        "B5-PureLLM": [],
        "Hybrid-Verified": []
    }
    
    for run_idx in range(num_runs):
        logger.info(f"\n  --- Run {run_idx + 1}/{num_runs} ---")
        
        # B5: Pure LLM Clustering
        logger.info(f"  Running B5: Pure LLM Clustering...")
        try:
            b5_result = pure_llm_clustering(
                features_list, stack_traces, llm_client, max_tests_per_batch=20
            )
            b5_eval = evaluate_clustering(b5_result, ground_truth_clusters, test_names)
            llm_run_results["B5-PureLLM"].append(b5_eval)
            logger.info(f"    B5 ARI: {b5_eval.ari:.4f}, NMI: {b5_eval.nmi:.4f}")
        except Exception as e:
            logger.warning(f"    B5 failed: {e}")
        
        # Hybrid-Verified (LLM verification)
        logger.info(f"  Running Hybrid-Verified...")
        try:
            verifier = ClusterVerifier(llm_client, max_tests_per_verification=10)
            verified_clusters, _ = verifier.verify_all_clusters(
                deterministic_results["Hybrid-Signature"],
                features_map, stack_traces, source_codes, min_cluster_size=2
            )
            verified_result = verified_to_clustering_result(verified_clusters)
            verified_eval = evaluate_clustering(verified_result, ground_truth_clusters, test_names)
            llm_run_results["Hybrid-Verified"].append(verified_eval)
            logger.info(f"    Verified ARI: {verified_eval.ari:.4f}, NMI: {verified_eval.nmi:.4f}")
        except Exception as e:
            logger.warning(f"    Verification failed: {e}")
    
    # ========================================================================
    # Step 5: Aggregate Results
    # ========================================================================
    logger.info("\n[Step 5] Aggregating results...")
    
    def aggregate_metrics(eval_list: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate metrics from multiple runs."""
        if not eval_list:
            return {}
        
        # Use only the 4 essential metrics (based on SE literature survey)
        metrics = ["ari", "nmi", "purity", "v_measure"]
        result = {}
        
        for metric in metrics:
            values = [getattr(e, metric) for e in eval_list]
            result[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values
            }
        
        return result
    
    # Aggregate LLM results
    aggregated_llm_results = {}
    for method_name, eval_list in llm_run_results.items():
        if eval_list:
            aggregated_llm_results[method_name] = aggregate_metrics(eval_list)
    
    # ========================================================================
    # Step 6: Print Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-RUN EXPERIMENT RESULTS")
    logger.info("=" * 80)
    
    # Deterministic methods
    logger.info("\nDeterministic Methods (1 run):")
    logger.info("-" * 80)
    logger.info(f"{'Method':<25} {'ARI':>10} {'NMI':>10} {'Purity':>10} {'V-Measure':>12}")
    logger.info("-" * 80)
    
    for method_name, eval_result in deterministic_evaluations.items():
        logger.info(
            f"{method_name:<25} {eval_result.ari:>10.4f} {eval_result.nmi:>10.4f} "
            f"{eval_result.purity:>10.4f} {eval_result.v_measure:>12.4f}"
        )
    
    # LLM methods
    logger.info(f"\nLLM-based Methods ({num_runs} runs, mean ± std):")
    logger.info("-" * 80)
    logger.info(f"{'Method':<25} {'ARI':>15} {'NMI':>15} {'Purity':>15}")
    logger.info("-" * 80)
    
    for method_name, metrics in aggregated_llm_results.items():
        ari = metrics.get("ari", {})
        nmi = metrics.get("nmi", {})
        purity = metrics.get("purity", {})
        logger.info(
            f"{method_name:<25} "
            f"{ari.get('mean', 0):.4f}±{ari.get('std', 0):.4f} "
            f"{nmi.get('mean', 0):.4f}±{nmi.get('std', 0):.4f} "
            f"{purity.get('mean', 0):.4f}±{purity.get('std', 0):.4f}"
        )
    
    # Cost summary
    cost_summary = cost_tracker.get_summary()
    if cost_summary["total_calls"] > 0:
        logger.info("\n" + cost_tracker.format_summary())
    
    # ========================================================================
    # Step 7: Save Results
    # ========================================================================
    logger.info("\n[Step 7] Saving results...")
    
    results = {
        "timestamp": timestamp,
        "config": {
            "num_runs": num_runs,
            "skip_embedding": skip_embedding,
            "project_filter": project_filter
        },
        "data_stats": {
            "num_clusters": len(clusters),
            "num_tests": len(all_tests),
            "num_ground_truth_clusters": len(ground_truth_clusters)
        },
        "categorization_stats": cat_stats,
        "deterministic_results": {
            name: {
                "ari": e.ari, "nmi": e.nmi, "purity": e.purity,
                "v_measure": e.v_measure,
                "num_clusters": e.num_predicted_clusters
            }
            for name, e in deterministic_evaluations.items()
        },
        "llm_results": aggregated_llm_results,
        "llm_cost": cost_summary
    }
    
    results_file = output_dir / f"rq2_multirun_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Multi-run experiment completed!")
    logger.info("=" * 60)
    
    return results


# ============================================================================
# ABLATION STUDY
# ============================================================================

class AblationFeatureExtractor:
    """
    Feature extractor with configurable tiers for ablation study.
    """
    
    def __init__(
        self, 
        llm_client: Optional[LLMClient] = None,
        use_tier1: bool = True,
        use_tier2: bool = True,
        use_tier3: bool = True,
        use_tier4: bool = True
    ):
        """
        Initialize with configurable tiers.
        
        Args:
            llm_client: LLM client for Tier 3
            use_tier1: Enable Tier 1 (exact match)
            use_tier2: Enable Tier 2 (keyword match)
            use_tier3: Enable Tier 3 (LLM classification)
            use_tier4: Enable Tier 4 (code-based inference)
        """
        from rq2_feature_extractor import ThreeTierExceptionCategorizer
        
        self.use_tier1 = use_tier1
        self.use_tier2 = use_tier2
        self.use_tier3 = use_tier3
        self.use_tier4 = use_tier4
        
        # Only pass LLM client if Tier 3 is enabled
        self.categorizer = ThreeTierExceptionCategorizer(
            llm_client if use_tier3 else None
        )
    
    def extract_features(self, test_case: Any, project: str) -> TestFeatures:
        """Extract features with configured tiers."""
        features = TestFeatures(
            test_name=test_case.name,
            project=project,
            test_class=test_case.class_name,
            test_method=test_case.method_name,
            has_source_code=bool(test_case.source_code),
            has_stack_trace=bool(test_case.stack_traces),
        )
        
        # Extract exception info from stack trace
        if test_case.stack_traces:
            stack_trace = test_case.stack_traces[0]
            exc_type, exc_msg = self._parse_exception_from_trace(stack_trace)
            features.exception_type = exc_type
            features.exception_simple_name = exc_type.split(".")[-1] if "." in exc_type else exc_type
            features.exception_message = exc_msg
            
            # Categorize with configured tiers
            category, tier = self._categorize_with_tiers(exc_type, exc_msg, stack_trace)
            features.exception_category = category
            features.categorization_tier = tier
        
        elif test_case.source_code and self.use_tier4:
            # Tier 4: Code-based inference
            category = self._infer_category_from_code(test_case.source_code)
            features.exception_category = category
            features.categorization_tier = 4
        else:
            features.exception_category = "Unknown"
            features.categorization_tier = 0
        
        # Build signature
        features.signature = self._build_signature(features)
        
        return features
    
    def _categorize_with_tiers(
        self, 
        exc_type: str, 
        exc_msg: str, 
        stack_trace: str
    ) -> Tuple[str, int]:
        """Categorize using only enabled tiers."""
        simple_name = exc_type.split(".")[-1] if "." in exc_type else exc_type
        
        # Tier 1
        if self.use_tier1:
            category = self.categorizer._tier1_match(simple_name)
            if category:
                return category, 1
        
        # Tier 2
        if self.use_tier2:
            category = self.categorizer._tier2_match(exc_type, exc_msg)
            if category:
                return category, 2
        
        # Tier 3
        if self.use_tier3:
            category = self.categorizer._tier3_llm_classify(exc_type, exc_msg, stack_trace)
            return category, 3
        
        return "Unknown", 0
    
    def _parse_exception_from_trace(self, stack_trace: str) -> Tuple[str, str]:
        """Parse exception from stack trace."""
        if not stack_trace:
            return "", ""
        
        lines = stack_trace.strip().split("\n")
        if not lines:
            return "", ""
        
        first_line = lines[0].strip()
        if ":" in first_line:
            parts = first_line.split(":", 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        return first_line, ""
    
    def _infer_category_from_code(self, source_code: str) -> str:
        """Infer category from source code."""
        code_lower = source_code.lower()
        
        patterns = {
            "Networking": ["socket", "http", "url", "connection", "client", "server", "network"],
            "Filesystem": ["file", "path", "directory", "inputstream", "outputstream"],
            "Concurrency": ["thread", "executor", "concurrent", "async", "lock", "synchronized"],
            "Timeout": ["timeout", "wait", "sleep", "delay", "deadline"],
            "Assertion": ["assert", "expect", "verify", "should"],
        }
        
        for category, keywords in patterns.items():
            if any(kw in code_lower for kw in keywords):
                return category
        return "Unknown"
    
    def _build_signature(self, features: TestFeatures) -> str:
        """Build clustering signature."""
        project = features.project or "Unknown"
        category = features.exception_category or "Unknown"
        strong_categories = ["Networking", "Filesystem", "Timeout", "Concurrency"]
        
        if category in strong_categories:
            return f"{project}:{category}"
        return f"{project}:Other"


class SignatureStrategy:
    """
    Different signature strategies for ablation study.
    
    Based on Crash Bucketing literature (MSR 2016, FASE 2017, ICSE 2022),
    we explore various signature designs to understand what factors
    contribute most to clustering quality.
    
    Design Rationale:
    - Our observation: Systemic Flakiness is a project-level problem
    - Alluxio: 113 tests → 1 cluster (all share same DNS resolution issue)
    - Spring Boot: 147 tests → 6 clusters (different failure modes)
    - This suggests Project is the PRIMARY grouping factor
    """
    
    # ========================================================================
    # Group A: Project-centric strategies (our hypothesis)
    # ========================================================================
    
    @staticmethod
    def project_only(features: TestFeatures) -> str:
        """
        A1: Signature = Project only
        
        Hypothesis: All tests in the same project share the same root cause.
        Expected: High recall but low precision (over-grouping)
        """
        return features.project or "Unknown"
    
    @staticmethod
    def project_category(features: TestFeatures) -> str:
        """
        A2: Signature = Project:Category (OUR DEFAULT METHOD)
        
        Hypothesis: Tests in the same project with the same failure category
        share the same root cause. Strong categories (Networking, Filesystem,
        Timeout, Concurrency) indicate distinct failure modes.
        
        This is our proposed method based on data observation.
        """
        project = features.project or "Unknown"
        category = features.exception_category or "Unknown"
        strong_categories = ["Networking", "Filesystem", "Timeout", "Concurrency"]
        if category in strong_categories:
            return f"{project}:{category}"
        return f"{project}:Other"
    
    @staticmethod
    def project_category_type(features: TestFeatures) -> str:
        """
        A3: Signature = Project:Category:ExceptionType (fine-grained)
        
        Hypothesis: More specific signature leads to better precision.
        Expected: Higher precision but may over-segment (too many clusters)
        """
        project = features.project or "Unknown"
        category = features.exception_category or "Unknown"
        exc_type = features.exception_simple_name or "Unknown"
        return f"{project}:{category}:{exc_type}"
    
    # ========================================================================
    # Group B: Exception-centric strategies (traditional approach)
    # ========================================================================
    
    @staticmethod
    def category_only(features: TestFeatures) -> str:
        """
        B1: Signature = Category only
        
        Traditional approach: Group by exception category regardless of project.
        Expected: May incorrectly group unrelated tests from different projects.
        """
        return features.exception_category or "Unknown"
    
    @staticmethod
    def exception_type_only(features: TestFeatures) -> str:
        """
        B2: Signature = ExceptionType only
        
        Traditional crash bucketing approach (similar to WER top-frame).
        Expected: Similar exceptions from different projects grouped together.
        """
        return features.exception_simple_name or "Unknown"
    
    @staticmethod
    def exception_type_message(features: TestFeatures) -> str:
        """
        B3: Signature = ExceptionType:MessageHash
        
        More specific: Same exception type AND similar message.
        Uses first 50 chars of message to avoid noise.
        """
        exc_type = features.exception_simple_name or "Unknown"
        msg_hash = features.exception_message[:50] if features.exception_message else ""
        # Normalize message
        msg_hash = msg_hash.lower().strip()
        return f"{exc_type}:{msg_hash}"
    
    # ========================================================================
    # Group C: Test structure-centric strategies
    # ========================================================================
    
    @staticmethod
    def test_class_only(features: TestFeatures) -> str:
        """
        C1: Signature = TestClass only
        
        Hypothesis: Tests in the same class share the same root cause.
        Expected: Over-segmentation (too many clusters)
        """
        return features.test_class or "Unknown"
    
    @staticmethod
    def project_test_class(features: TestFeatures) -> str:
        """
        C2: Signature = Project:TestClass
        
        Combines project and class information.
        Expected: Very fine-grained, likely over-segmented.
        """
        project = features.project or "Unknown"
        test_class = features.test_class or "Unknown"
        # Use only the simple class name
        simple_class = test_class.split(".")[-1] if "." in test_class else test_class
        return f"{project}:{simple_class}"
    
    # ========================================================================
    # Group D: Hybrid strategies
    # ========================================================================
    
    @staticmethod
    def exception_type_entry_point(features: TestFeatures) -> str:
        """
        D1: Signature = ExceptionType:EntryPoint
        
        Crash bucketing style: Exception type + first project-specific frame.
        Based on WER and Socorro practices.
        """
        exc_type = features.exception_simple_name or "Unknown"
        entry = features.entry_point or "Unknown"
        # Use only class name from entry point
        entry_class = entry.split(".")[-1] if "." in entry else entry
        return f"{exc_type}:{entry_class}"
    
    @staticmethod
    def project_exception_entry(features: TestFeatures) -> str:
        """
        D2: Signature = Project:ExceptionType:EntryPoint
        
        Most comprehensive: Project + Exception + Entry point.
        Expected: Very specific, may over-segment.
        """
        project = features.project or "Unknown"
        exc_type = features.exception_simple_name or "Unknown"
        entry = features.entry_point or "Unknown"
        entry_class = entry.split(".")[-1] if "." in entry else entry
        return f"{project}:{exc_type}:{entry_class}"


def signature_clustering_with_strategy(
    features_list: List[TestFeatures],
    strategy_func
) -> ClusteringResult:
    """
    Signature-based clustering with custom strategy function.
    
    Args:
        features_list: List of TestFeatures
        strategy_func: Function that takes TestFeatures and returns signature string
        
    Returns:
        ClusteringResult
    """
    from collections import defaultdict
    
    signature_groups: Dict[str, List[TestFeatures]] = defaultdict(list)
    
    for features in features_list:
        sig = strategy_func(features)
        signature_groups[sig].append(features)
    
    clusters = []
    test_to_cluster = {}
    cluster_id = 0
    
    for signature, group in signature_groups.items():
        cluster = PredictedCluster(
            cluster_id=cluster_id,
            tests=[f.test_name for f in group],
            signature=signature,
            predicted_category=group[0].exception_category if group else "",
            method="signature"
        )
        clusters.append(cluster)
        for f in group:
            test_to_cluster[f.test_name] = cluster_id
        cluster_id += 1
    
    return ClusteringResult(
        method_name="Signature-Custom",
        clusters=clusters,
        test_to_cluster=test_to_cluster
    )


def run_ablation_study(
    output_dir: Optional[Path] = None,
    skip_llm: bool = False
) -> Dict[str, Any]:
    """
    Run complete ablation study to evaluate contribution of each component.
    
    Ablation 1: Feature Extraction Tiers
    - A1: Tier 1 only
    - A2: Tier 1 + 2
    - A3: Tier 1 + 2 + 3
    - A4: Tier 1 + 2 + 3 + 4 (full)
    
    Ablation 2: Signature Design
    - S1: Project only
    - S2: Category only
    - S3: Project + Category (default)
    - S4: Project + Category + Exception Type
    
    Args:
        output_dir: Output directory
        skip_llm: Skip LLM-based configurations
        
    Returns:
        Dict with ablation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = LOGS_DIR / f"rq2_ablation_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("RQ2 Ablation Study")
    logger.info("=" * 60)
    
    # Reset cost tracker
    reset_cost_tracker()
    cost_tracker = get_cost_tracker()
    
    # Load data
    logger.info("\n[Step 1] Loading data...")
    clusters = load_all_clusters(require_ground_truth=False)
    all_tests, ground_truth_clusters, stack_traces, source_codes = prepare_experiment_data(clusters)
    test_names = [t.name for t in all_tests]
    
    logger.info(f"Loaded {len(clusters)} clusters, {len(all_tests)} tests")
    
    # Create LLM client if needed
    llm_client = None
    if not skip_llm:
        try:
            llm_client = create_llm_client("openai")
            logger.info("LLM client created for Tier 3")
        except Exception as e:
            logger.warning(f"Could not create LLM client: {e}")
    
    results = {
        "timestamp": timestamp,
        "data_stats": {
            "num_clusters": len(clusters),
            "num_tests": len(all_tests),
            "num_ground_truth_clusters": len(ground_truth_clusters)
        },
        "ablation_tier": {},
        "ablation_signature": {}
    }
    
    # ========================================================================
    # Ablation 1: Feature Extraction Tiers
    # ========================================================================
    logger.info("\n[Step 2] Ablation 1: Feature Extraction Tiers")
    logger.info("-" * 60)
    
    tier_configs = [
        ("A1_Tier1", {"use_tier1": True, "use_tier2": False, "use_tier3": False, "use_tier4": False}),
        ("A2_Tier12", {"use_tier1": True, "use_tier2": True, "use_tier3": False, "use_tier4": False}),
        ("A3_Tier123", {"use_tier1": True, "use_tier2": True, "use_tier3": True, "use_tier4": False}),
        ("A4_Tier1234", {"use_tier1": True, "use_tier2": True, "use_tier3": True, "use_tier4": True}),
    ]
    
    for config_name, tier_config in tier_configs:
        # Skip Tier 3 configs if no LLM
        if tier_config["use_tier3"] and not llm_client:
            logger.info(f"  Skipping {config_name} (no LLM client)")
            continue
        
        logger.info(f"  Running {config_name}...")
        
        extractor = AblationFeatureExtractor(llm_client, **tier_config)
        features_list = [extractor.extract_features(t, t.project) for t in all_tests]
        
        clustering = signature_based_clustering(features_list)
        evaluation = evaluate_clustering(clustering, ground_truth_clusters, test_names)
        
        results["ablation_tier"][config_name] = {
            "config": tier_config,
            "ari": evaluation.ari,
            "nmi": evaluation.nmi,
            "purity": evaluation.purity,
            "v_measure": evaluation.v_measure,
            "num_clusters": evaluation.num_predicted_clusters
        }
        
        logger.info(f"    ARI: {evaluation.ari:.4f}, NMI: {evaluation.nmi:.4f}, "
                   f"Purity: {evaluation.purity:.4f}, V-Measure: {evaluation.v_measure:.4f}")
    
    # ========================================================================
    # Ablation 2: Signature Design (Comprehensive)
    # ========================================================================
    logger.info("\n[Step 3] Ablation 2: Signature Design")
    logger.info("=" * 60)
    logger.info("Testing different signature strategies based on Crash Bucketing literature")
    logger.info("-" * 60)
    
    # Use full feature extraction for signature ablation
    full_extractor = FeatureExtractor(llm_client if not skip_llm else None)
    full_features = [full_extractor.extract_features(t, t.project) for t in all_tests]
    
    # Comprehensive signature strategies organized by hypothesis
    signature_configs = [
        # Group A: Project-centric (our hypothesis)
        ("A1_ProjectOnly", SignatureStrategy.project_only, 
         "Project only - tests in same project share root cause"),
        ("A2_ProjectCategory", SignatureStrategy.project_category, 
         "Project:Category - OUR PROPOSED METHOD"),
        ("A3_ProjectCategoryType", SignatureStrategy.project_category_type, 
         "Project:Category:ExceptionType - fine-grained"),
        
        # Group B: Exception-centric (traditional approach)
        ("B1_CategoryOnly", SignatureStrategy.category_only, 
         "Category only - traditional approach"),
        ("B2_ExceptionTypeOnly", SignatureStrategy.exception_type_only, 
         "ExceptionType only - WER-style"),
        ("B3_ExceptionTypeMessage", SignatureStrategy.exception_type_message, 
         "ExceptionType:Message - more specific"),
        
        # Group C: Test structure-centric
        ("C1_TestClassOnly", SignatureStrategy.test_class_only, 
         "TestClass only - structural grouping"),
        ("C2_ProjectTestClass", SignatureStrategy.project_test_class, 
         "Project:TestClass - combined"),
        
        # Group D: Hybrid (crash bucketing style)
        ("D1_ExceptionEntryPoint", SignatureStrategy.exception_type_entry_point, 
         "ExceptionType:EntryPoint - crash bucketing style"),
        ("D2_ProjectExceptionEntry", SignatureStrategy.project_exception_entry, 
         "Project:Exception:Entry - comprehensive"),
    ]
    
    for config_name, strategy_func, description in signature_configs:
        logger.info(f"\n  [{config_name}] {description}")
        
        clustering = signature_clustering_with_strategy(full_features, strategy_func)
        evaluation = evaluate_clustering(clustering, ground_truth_clusters, test_names)
        
        results["ablation_signature"][config_name] = {
            "description": description,
            "ari": evaluation.ari,
            "nmi": evaluation.nmi,
            "purity": evaluation.purity,
            "v_measure": evaluation.v_measure,
            "num_clusters": evaluation.num_predicted_clusters,
            "num_gt_clusters": evaluation.num_ground_truth_clusters
        }
        
        logger.info(f"    ARI: {evaluation.ari:.4f}, NMI: {evaluation.nmi:.4f}, "
                   f"Purity: {evaluation.purity:.4f}, V-Measure: {evaluation.v_measure:.4f}, "
                   f"Clusters: {evaluation.num_predicted_clusters} (GT: {evaluation.num_ground_truth_clusters})")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    logger.info("\n[Step 4] Saving results...")
    
    cost_summary = cost_tracker.get_summary()
    results["llm_cost"] = cost_summary
    
    results_file = output_dir / f"rq2_ablation_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 80)
    
    logger.info("\nAblation 1: Feature Extraction Tiers")
    logger.info("-" * 80)
    logger.info(f"{'Config':<20} {'ARI':>10} {'NMI':>10} {'Purity':>10} {'V-Measure':>12} {'#Clusters':>10}")
    logger.info("-" * 80)
    for config_name, metrics in results["ablation_tier"].items():
        logger.info(f"{config_name:<20} {metrics['ari']:>10.4f} {metrics['nmi']:>10.4f} "
                   f"{metrics['purity']:>10.4f} {metrics['v_measure']:>12.4f} {metrics['num_clusters']:>10}")
    
    logger.info("\nAblation 2: Signature Design")
    logger.info("-" * 100)
    logger.info(f"{'Config':<25} {'ARI':>8} {'NMI':>8} {'Purity':>8} {'V-Meas':>8} {'#Pred':>7} {'#GT':>5}")
    logger.info("-" * 100)
    
    # Group results by category
    groups = {
        "A": "Project-centric (our hypothesis)",
        "B": "Exception-centric (traditional)",
        "C": "Test structure-centric",
        "D": "Hybrid (crash bucketing style)"
    }
    
    current_group = ""
    for config_name, metrics in sorted(results["ablation_signature"].items()):
        group = config_name[0]
        if group != current_group:
            current_group = group
            logger.info(f"\n  --- {groups.get(group, 'Other')} ---")
        
        logger.info(f"  {config_name:<23} {metrics['ari']:>8.4f} {metrics['nmi']:>8.4f} "
                   f"{metrics['purity']:>8.4f} {metrics['v_measure']:>8.4f} "
                   f"{metrics['num_clusters']:>7} {metrics.get('num_gt_clusters', 'N/A'):>5}")
    
    if cost_summary["total_calls"] > 0:
        logger.info("\n" + cost_tracker.format_summary())
    
    logger.info("\n" + "=" * 60)
    logger.info("Ablation study completed!")
    logger.info("=" * 60)
    
    return results


# ============================================================================
# TIER 4 ACCURACY EVALUATION (W3)
# ============================================================================

def evaluate_tier4_accuracy(
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate Tier 4 (code-based inference) accuracy.
    
    For tests that have BOTH stack trace AND source code:
    - Compare Tier 1/2 category (from stack trace) with Tier 4 category (from code)
    - Calculate agreement rate
    
    This addresses reviewer concern W3 about Tier 4 reliability.
    
    Returns:
        Dict with accuracy metrics and confusion matrix
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or RESULTS_DIR
    
    logger = setup_logging(LOGS_DIR / f"rq2_tier4_eval_{timestamp}.log")
    logger.info("=" * 60)
    logger.info("Tier 4 Accuracy Evaluation")
    logger.info("=" * 60)
    
    # Load data
    clusters = load_all_clusters(require_ground_truth=False)
    all_tests, _, stack_traces, source_codes = prepare_experiment_data(clusters)
    
    # Find tests with BOTH stack trace and source code
    tests_with_both = [
        t for t in all_tests 
        if t.name in stack_traces and t.name in source_codes
    ]
    
    logger.info(f"Total tests: {len(all_tests)}")
    logger.info(f"Tests with both stack trace and source code: {len(tests_with_both)}")
    
    if not tests_with_both:
        logger.warning("No tests with both stack trace and source code found!")
        return {"error": "No suitable tests"}
    
    # Extract categories using both methods
    from rq2_feature_extractor import ThreeTierExceptionCategorizer
    
    categorizer = ThreeTierExceptionCategorizer()  # No LLM for Tier 1/2 only
    
    agreements = []
    category_pairs = []
    
    for test in tests_with_both:
        # Get stack trace category (Tier 1/2)
        stack_trace = stack_traces[test.name]
        lines = stack_trace.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            if ":" in first_line:
                exc_type = first_line.split(":", 1)[0].strip()
                exc_msg = first_line.split(":", 1)[1].strip() if ":" in first_line else ""
            else:
                exc_type = first_line
                exc_msg = ""
            
            trace_category, trace_tier = categorizer.categorize(exc_type, exc_msg, stack_trace)
        else:
            trace_category = "Unknown"
            trace_tier = 0
        
        # Get code category (Tier 4)
        source_code = source_codes[test.name]
        code_category = _infer_category_from_code_standalone(source_code)
        
        # Record
        match = trace_category == code_category
        agreements.append({
            "test": test.name,
            "trace_category": trace_category,
            "code_category": code_category,
            "trace_tier": trace_tier,
            "match": match
        })
        category_pairs.append((trace_category, code_category))
    
    # Calculate metrics
    match_count = sum(1 for a in agreements if a["match"])
    match_rate = match_count / len(agreements) if agreements else 0
    
    # Build confusion matrix
    categories = ["Networking", "Filesystem", "Timeout", "Concurrency", 
                  "Configuration", "Resource", "Assertion", "Other", "Unknown"]
    confusion = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    for trace_cat, code_cat in category_pairs:
        if trace_cat in confusion and code_cat in confusion[trace_cat]:
            confusion[trace_cat][code_cat] += 1
    
    results = {
        "timestamp": timestamp,
        "num_tests_evaluated": len(agreements),
        "match_rate": match_rate,
        "match_count": match_count,
        "confusion_matrix": confusion,
        "details": agreements
    }
    
    # Log results
    logger.info(f"\nTier 4 Accuracy Results:")
    logger.info(f"  Tests evaluated: {len(agreements)}")
    logger.info(f"  Agreement rate: {match_rate:.2%}")
    logger.info(f"  Matches: {match_count}/{len(agreements)}")
    
    logger.info("\nConfusion Matrix (rows=trace, cols=code):")
    header = "          " + " ".join(f"{c[:8]:>10}" for c in categories)
    logger.info(header)
    for c1 in categories:
        row = f"{c1[:8]:>10}" + " ".join(f"{confusion[c1][c2]:>10}" for c2 in categories)
        logger.info(row)
    
    # Save results
    results_file = output_dir / f"rq2_tier4_eval_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    return results


def _infer_category_from_code_standalone(source_code: str) -> str:
    """Standalone Tier 4 inference function."""
    code_lower = source_code.lower()
    
    patterns = {
        "Networking": ["socket", "http", "url", "connection", "client", "server", 
                       "request", "response", "network", "dns", "host", "port"],
        "Filesystem": ["file", "path", "directory", "inputstream", "outputstream",
                       "reader", "writer", "filesystem", "tempfile"],
        "Concurrency": ["thread", "executor", "concurrent", "async", "await",
                        "lock", "synchronized", "latch", "barrier", "semaphore"],
        "Timeout": ["timeout", "wait", "sleep", "delay", "deadline"],
        "Assertion": ["assert", "expect", "verify", "should"],
    }
    
    for category, keywords in patterns.items():
        if any(kw in code_lower for kw in keywords):
            return category
    return "Unknown"


# ============================================================================
# INTRA-PROJECT CLUSTER ANALYSIS (W2)
# ============================================================================

def analyze_intra_project_clustering(
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze clustering performance within individual projects.
    
    This addresses reviewer concern W2 about Project-Aware strategy generalization.
    For projects with multiple clusters, evaluate if our method can distinguish them.
    
    Returns:
        Dict with per-project analysis
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or RESULTS_DIR
    
    logger = setup_logging(LOGS_DIR / f"rq2_intra_project_{timestamp}.log")
    logger.info("=" * 60)
    logger.info("Intra-Project Cluster Analysis")
    logger.info("=" * 60)
    
    # Load data
    clusters = load_all_clusters(require_ground_truth=False)
    all_tests, ground_truth_clusters, stack_traces, source_codes = prepare_experiment_data(clusters)
    
    # Group clusters by project
    from collections import defaultdict
    project_clusters: Dict[str, List[List[str]]] = defaultdict(list)
    project_tests: Dict[str, List[str]] = defaultdict(list)
    
    for cluster in clusters:
        test_names = [t.name for t in cluster.tests]
        if test_names:
            project_clusters[cluster.project].append(test_names)
            project_tests[cluster.project].extend(test_names)
    
    logger.info(f"\nProject Statistics:")
    logger.info(f"{'Project':<40} {'#Clusters':>10} {'#Tests':>10}")
    logger.info("-" * 60)
    for project in sorted(project_clusters.keys()):
        logger.info(f"{project[:40]:<40} {len(project_clusters[project]):>10} {len(project_tests[project]):>10}")
    
    # Extract features
    extractor = FeatureExtractor()
    features_list = [extractor.extract_features(t, t.project) for t in all_tests]
    features_map = {f.test_name: f for f in features_list}
    
    # Run clustering
    clustering_result = signature_based_clustering(features_list)
    
    # Analyze per-project
    results = {
        "timestamp": timestamp,
        "per_project": {}
    }
    
    logger.info("\nPer-Project Clustering Results:")
    logger.info("-" * 80)
    logger.info(f"{'Project':<40} {'GT Clusters':>12} {'Pred Clusters':>14} {'ARI':>10}")
    logger.info("-" * 80)
    
    for project in sorted(project_clusters.keys()):
        if len(project_clusters[project]) < 2:
            continue  # Skip projects with only 1 cluster
        
        # Filter to this project
        project_test_set = set(project_tests[project])
        
        # Get ground truth labels for this project
        gt_labels = {}
        for cluster_id, cluster_tests in enumerate(project_clusters[project]):
            for test in cluster_tests:
                gt_labels[test] = cluster_id
        
        # Get predicted labels for this project
        pred_labels = {}
        for test in project_test_set:
            if test in clustering_result.test_to_cluster:
                pred_labels[test] = clustering_result.test_to_cluster[test]
        
        # Compute ARI for this project
        common_tests = sorted(set(gt_labels.keys()) & set(pred_labels.keys()))
        if len(common_tests) >= 2:
            gt_list = [gt_labels[t] for t in common_tests]
            pred_list = [pred_labels[t] for t in common_tests]
            
            from rq2_evaluation import compute_ari
            project_ari = compute_ari(gt_list, pred_list)
            
            # Count unique predicted clusters for this project
            pred_cluster_count = len(set(pred_list))
            
            results["per_project"][project] = {
                "num_gt_clusters": len(project_clusters[project]),
                "num_pred_clusters": pred_cluster_count,
                "num_tests": len(common_tests),
                "ari": project_ari
            }
            
            logger.info(f"{project[:40]:<40} {len(project_clusters[project]):>12} "
                       f"{pred_cluster_count:>14} {project_ari:>10.4f}")
    
    # Save results
    results_file = output_dir / f"rq2_intra_project_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    return results


# ============================================================================
# GROUND TRUTH VALIDATION (W1)
# ============================================================================

def validate_ground_truth(
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Validate Ground Truth clusters for semantic meaningfulness.
    
    This addresses reviewer concern W1 about Ground Truth validity.
    
    IMPORTANT: The Ground Truth has TWO layers of validation:
    
    Layer 1: Statistical clustering (Jaccard Distance)
        - Based on 10,000 test runs
        - Co-failing patterns identified statistically
        - Produces 45 clusters
    
    Layer 2: Human annotation (Q3 in cluster_form.txt)
        - EMSE paper authors manually analyzed each cluster
        - Q3 answers: "What is the root cause of these tests?"
        - Confirms semantic validity of clusters
        - 28 clusters have explicit Q3 annotations
    
    This function reports:
    - How many clusters have Q3 annotations
    - How many Q3 annotations indicate shared root cause
    
    Returns:
        Dict with validation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or RESULTS_DIR
    
    logger = setup_logging(LOGS_DIR / f"rq2_gt_validation_{timestamp}.log")
    logger.info("=" * 60)
    logger.info("Ground Truth Validation")
    logger.info("=" * 60)
    
    # Load clusters WITH ground truth
    clusters = load_all_clusters(require_ground_truth=False)
    
    results = {
        "timestamp": timestamp,
        "total_clusters": len(clusters),
        "clusters_with_q3": 0,
        "clusters_with_shared_cause": 0,
        "cluster_details": []
    }
    
    # Shared cause indicators
    shared_cause_keywords = [
        "dns", "network", "connection", "timeout", "resource",
        "configuration", "dependency", "external", "service",
        "infrastructure", "environment", "port", "socket"
    ]
    
    for cluster in clusters:
        q3 = cluster.q3_answer or ""
        has_q3 = bool(q3 and q3.strip() and q3.strip() != "...")
        
        if has_q3:
            results["clusters_with_q3"] += 1
            
            # Check if Q3 indicates shared cause
            q3_lower = q3.lower()
            has_shared_cause = any(kw in q3_lower for kw in shared_cause_keywords)
            
            if has_shared_cause:
                results["clusters_with_shared_cause"] += 1
        else:
            has_shared_cause = False
        
        results["cluster_details"].append({
            "project": cluster.project,
            "cluster_id": cluster.cluster_id,
            "num_tests": len(cluster.tests),
            "has_q3": has_q3,
            "has_shared_cause": has_shared_cause,
            "q3_excerpt": q3[:200] if q3 else ""
        })
    
    # Calculate rates
    results["q3_coverage"] = results["clusters_with_q3"] / results["total_clusters"] if results["total_clusters"] > 0 else 0
    results["shared_cause_rate"] = results["clusters_with_shared_cause"] / results["clusters_with_q3"] if results["clusters_with_q3"] > 0 else 0
    
    # Log results
    logger.info(f"\nGround Truth Validation Results:")
    logger.info(f"  Total clusters: {results['total_clusters']}")
    logger.info(f"  Clusters with Q3 annotation: {results['clusters_with_q3']} ({results['q3_coverage']:.1%})")
    logger.info(f"  Clusters with shared cause: {results['clusters_with_shared_cause']} ({results['shared_cause_rate']:.1%})")
    
    # Save results
    results_file = output_dir / f"rq2_gt_validation_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    return results


# ============================================================================
# MULTI-MODEL EXPERIMENT (Reviewer Concern #2)
# ============================================================================

def run_multi_model_experiment(
    providers: List[str] = None,
    skip_embedding: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run RQ2 experiment with multiple LLM models.
    
    This addresses reviewer concern about generalizability across models.
    
    Args:
        providers: List of LLM providers to test (default: openai, deepseek, groq)
        skip_embedding: Skip embedding-based clustering
        output_dir: Output directory
        
    Returns:
        Dict with per-model results
    """
    if providers is None:
        providers = ["openai", "deepseek", "groq"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = LOGS_DIR / f"rq2_multimodel_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("RQ2 Multi-Model Experiment")
    logger.info("=" * 60)
    logger.info(f"Models to test: {providers}")
    
    # Load data once
    clusters = load_all_clusters(require_ground_truth=False)
    all_tests, ground_truth_clusters, stack_traces, source_codes = prepare_experiment_data(clusters)
    test_names = [t.name for t in all_tests]
    
    logger.info(f"Loaded {len(clusters)} clusters, {len(all_tests)} tests")
    
    results = {
        "timestamp": timestamp,
        "providers": providers,
        "data_stats": {
            "num_clusters": len(clusters),
            "num_tests": len(all_tests)
        },
        "model_results": {}
    }
    
    for provider in providers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running with provider: {provider}")
        logger.info(f"{'='*60}")
        
        try:
            # Create LLM client
            llm_client = create_llm_client(provider)
            model_name = llm_client.get_model_name()
            logger.info(f"Model: {model_name}")
            
            # Reset cost tracker for this provider
            reset_cost_tracker()
            cost_tracker = get_cost_tracker()
            
            # Extract features with LLM (Tier 3)
            extractor = FeatureExtractor(llm_client)
            features_list = [extractor.extract_features(t, t.project) for t in all_tests]
            features_map = {f.test_name: f for f in features_list}
            
            # Run signature-based clustering
            clustering = signature_based_clustering(features_list)
            
            # Run LLM verification
            verifier = ClusterVerifier(llm_client, max_tests_per_verification=10)
            verified_clusters, verify_stats = verifier.verify_all_clusters(
                clustering, features_map, stack_traces, source_codes, min_cluster_size=2
            )
            verified_result = verified_to_clustering_result(verified_clusters)
            
            # Evaluate
            sig_eval = evaluate_clustering(clustering, ground_truth_clusters, test_names)
            ver_eval = evaluate_clustering(verified_result, ground_truth_clusters, test_names)
            
            # Get cost
            cost_summary = cost_tracker.get_summary()
            
            results["model_results"][provider] = {
                "model": model_name,
                "signature_clustering": {
                    "ari": sig_eval.ari,
                    "nmi": sig_eval.nmi,
                    "purity": sig_eval.purity,
                    "v_measure": sig_eval.v_measure,
                    "num_clusters": sig_eval.num_predicted_clusters
                },
                "verified_clustering": {
                    "ari": ver_eval.ari,
                    "nmi": ver_eval.nmi,
                    "purity": ver_eval.purity,
                    "v_measure": ver_eval.v_measure,
                    "num_clusters": ver_eval.num_predicted_clusters
                },
                "verification_stats": verify_stats,
                "cost": cost_summary
            }
            
            logger.info(f"\nResults for {provider} ({model_name}):")
            logger.info(f"  Signature: ARI={sig_eval.ari:.4f}, NMI={sig_eval.nmi:.4f}")
            logger.info(f"  Verified:  ARI={ver_eval.ari:.4f}, NMI={ver_eval.nmi:.4f}")
            logger.info(f"  Cost: ${cost_summary['total_cost_usd']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed for provider {provider}: {e}")
            results["model_results"][provider] = {"error": str(e)}
    
    # Save results
    results_file = output_dir / f"rq2_multimodel_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-MODEL RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Provider':<15} {'Model':<35} {'Sig ARI':>10} {'Ver ARI':>10} {'Cost':>10}")
    logger.info("-" * 80)
    
    for provider, data in results["model_results"].items():
        if "error" in data:
            logger.info(f"{provider:<15} {'ERROR':<35} {'-':>10} {'-':>10} {'-':>10}")
        else:
            sig_ari = data["signature_clustering"]["ari"]
            ver_ari = data["verified_clustering"]["ari"]
            cost = data["cost"]["total_cost_usd"]
            logger.info(f"{provider:<15} {data['model'][:35]:<35} {sig_ari:>10.4f} {ver_ari:>10.4f} ${cost:>9.4f}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Experiment: Automated Identification of Systemic Flakiness Clusters"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-based methods (B5 and verification)"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip LLM verification phase"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding-based clustering (B4)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Only run on specific project"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "deepseek", "together", "groq"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study"
    )
    parser.add_argument(
        "--multi-run",
        action="store_true",
        help="Run LLM-based methods multiple times to reduce variance"
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Run with multiple LLM models"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs for LLM-based methods (default: 5)"
    )
    parser.add_argument(
        "--tier4-eval",
        action="store_true",
        help="Evaluate Tier 4 accuracy"
    )
    parser.add_argument(
        "--intra-project",
        action="store_true",
        help="Run intra-project cluster analysis"
    )
    parser.add_argument(
        "--validate-gt",
        action="store_true",
        help="Validate ground truth clusters"
    )
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation_study(skip_llm=args.skip_llm)
    elif args.multi_run:
        run_multi_run_experiment(
            num_runs=args.num_runs,
            skip_embedding=args.skip_embedding,
            project_filter=args.project
        )
    elif args.multi_model:
        run_multi_model_experiment(skip_embedding=args.skip_embedding)
    elif args.tier4_eval:
        evaluate_tier4_accuracy()
    elif args.intra_project:
        analyze_intra_project_clustering()
    elif args.validate_gt:
        validate_ground_truth()
    else:
        run_experiment(
            skip_llm=args.skip_llm,
            skip_verification=args.skip_verification,
            skip_embedding=args.skip_embedding,
            project_filter=args.project
        )


if __name__ == "__main__":
    main()
