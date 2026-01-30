"""
Main Experiment Script for RQ1

This script runs the complete RQ1 experiment:
1. Load all clusters from the Systemic Flakiness dataset
2. Run Individual Analysis on each cluster
3. Run Collective Analysis on each cluster
4. Evaluate both diagnoses against ground truth
5. Compute aggregate statistics
6. Save results
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from tqdm import tqdm

from config import (
    RESULTS_DIR,
    LOGS_DIR,
    DEFAULT_LLM_PROVIDER,
    RUNS_PER_CLUSTER,
    MAX_TESTS_PER_CLUSTER,
    LOG_LEVEL,
    LOG_FORMAT
)
from data_loader import load_all_clusters, get_cluster_statistics, Cluster
from llm_client import create_llm_client, LLMClient
from analyzers import IndividualAnalyzer, CollectiveAnalyzer, DiagnosisResult
from evaluator import (
    Evaluator, EvaluationResult, compute_aggregate_metrics, 
    format_results_table, format_summary_report
)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Optional[Path] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=handlers
    )


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """
    Main experiment runner for RQ1.
    """
    
    def __init__(
        self,
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        runs_per_cluster: int = RUNS_PER_CLUSTER,
        max_tests: int = MAX_TESTS_PER_CLUSTER
    ):
        """
        Initialize the experiment runner.
        
        Args:
            llm_provider: LLM provider to use ("openai" or "anthropic")
            runs_per_cluster: Number of runs per cluster for handling non-determinism
            max_tests: Maximum tests to include per cluster
        """
        self.llm_provider = llm_provider
        self.runs_per_cluster = runs_per_cluster
        self.max_tests = max_tests
        
        # Initialize LLM client
        self.llm_client = create_llm_client(llm_provider)
        
        # Initialize analyzers
        self.individual_analyzer = IndividualAnalyzer(self.llm_client)
        self.collective_analyzer = CollectiveAnalyzer(self.llm_client)
        
        # Initialize evaluator
        self.evaluator = Evaluator(self.llm_client)
        
        # Results storage
        self.diagnosis_results: list[DiagnosisResult] = []
        self.evaluation_results: list[EvaluationResult] = []
        
        self.logger = logging.getLogger(__name__)
    
    def run_single_cluster(
        self,
        cluster: Cluster,
        run_individual: bool = True,
        run_collective: bool = True
    ) -> tuple[Optional[DiagnosisResult], Optional[DiagnosisResult]]:
        """
        Run analysis on a single cluster.
        
        Args:
            cluster: Cluster to analyze
            run_individual: Whether to run individual analysis
            run_collective: Whether to run collective analysis
            
        Returns:
            Tuple of (individual_result, collective_result)
        """
        individual_result = None
        collective_result = None
        
        if run_individual:
            try:
                individual_result = self.individual_analyzer.analyze_cluster(
                    cluster, max_tests=self.max_tests
                )
                self.diagnosis_results.append(individual_result)
            except Exception as e:
                self.logger.error(f"Individual analysis failed for {cluster.project}/cluster{cluster.cluster_id}: {e}")
        
        if run_collective:
            try:
                collective_result = self.collective_analyzer.analyze_cluster(
                    cluster, max_tests=self.max_tests
                )
                self.diagnosis_results.append(collective_result)
            except Exception as e:
                self.logger.error(f"Collective analysis failed for {cluster.project}/cluster{cluster.cluster_id}: {e}")
        
        return individual_result, collective_result
    
    def evaluate_diagnosis(
        self,
        diagnosis_result: DiagnosisResult,
        ground_truth: str
    ) -> EvaluationResult:
        """
        Evaluate a diagnosis result against ground truth.
        
        Args:
            diagnosis_result: Diagnosis to evaluate
            ground_truth: Human-annotated ground truth
            
        Returns:
            EvaluationResult
        """
        result = self.evaluator.evaluate(
            llm_diagnosis=diagnosis_result.diagnosis,
            ground_truth=ground_truth,
            cluster_project=diagnosis_result.cluster_project,
            cluster_id=diagnosis_result.cluster_id,
            method=diagnosis_result.method
        )
        self.evaluation_results.append(result)
        return result
    
    def run_experiment(
        self,
        clusters: Optional[list[Cluster]] = None,
        skip_existing: bool = False
    ) -> dict:
        """
        Run the complete experiment with multiple runs to handle LLM non-determinism.
        
        Args:
            clusters: List of clusters to analyze (loads all if None)
            skip_existing: Skip clusters that already have results
            
        Returns:
            Dictionary with experiment results (mean ± std across runs)
        """
        import numpy as np
        
        # Load clusters if not provided
        if clusters is None:
            self.logger.info("Loading clusters...")
            clusters = load_all_clusters()
        
        # Filter clusters with ground truth
        clusters_with_gt = [c for c in clusters if c.q3_answer]
        
        self.logger.info(f"Running experiment on {len(clusters_with_gt)} clusters (with ground truth)")
        self.logger.info(f"LLM Provider: {self.llm_provider}")
        self.logger.info(f"Model: {self.llm_client.get_model_name()}")
        self.logger.info(f"Max tests per cluster: {self.max_tests}")
        self.logger.info(f"Runs per cluster: {self.runs_per_cluster}")
        
        # Print dataset statistics
        stats = get_cluster_statistics(clusters)
        self.logger.info(f"Dataset statistics: {stats}")
        
        # Storage for multi-run results
        # Structure: {cluster_key: {method: [run1_result, run2_result, ...]}}
        all_run_results: dict[str, dict[str, list[EvaluationResult]]] = {}
        all_diagnoses: list[dict] = []
        
        # Run multiple times to handle LLM non-determinism
        for run_idx in range(self.runs_per_cluster):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"RUN {run_idx + 1}/{self.runs_per_cluster}")
            self.logger.info(f"{'='*60}")
            
            # Clear per-run storage
            self.diagnosis_results = []
            self.evaluation_results = []
            
            for cluster in tqdm(clusters_with_gt, desc=f"Run {run_idx+1}"):
                cluster_key = f"{cluster.project}/cluster{cluster.cluster_id}"
                
                if cluster_key not in all_run_results:
                    all_run_results[cluster_key] = {"individual": [], "collective": []}
                
                # Run both analyses
                individual_result, collective_result = self.run_single_cluster(cluster)
                
                # Evaluate and store results
                if individual_result:
                    eval_result = self.evaluate_diagnosis(individual_result, cluster.q3_answer)
                    all_run_results[cluster_key]["individual"].append(eval_result)
                    all_diagnoses.append({
                        "run": run_idx + 1,
                        "project": cluster.project,
                        "cluster_id": cluster.cluster_id,
                        "method": "individual",
                        "diagnosis": individual_result.diagnosis,
                        "ground_truth": cluster.q3_answer
                    })
                
                if collective_result:
                    eval_result = self.evaluate_diagnosis(collective_result, cluster.q3_answer)
                    all_run_results[cluster_key]["collective"].append(eval_result)
                    all_diagnoses.append({
                        "run": run_idx + 1,
                        "project": cluster.project,
                        "cluster_id": cluster.cluster_id,
                        "method": "collective",
                        "diagnosis": collective_result.diagnosis,
                        "ground_truth": cluster.q3_answer
                    })
        
        # Aggregate results across runs (compute mean ± std)
        aggregated_results = self._aggregate_multi_run_results(all_run_results)
        
        return {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "llm_provider": self.llm_provider,
                "model": self.llm_client.get_model_name(),
                "num_clusters": len(clusters_with_gt),
                "max_tests_per_cluster": self.max_tests,
                "runs_per_cluster": self.runs_per_cluster
            },
            "dataset_statistics": stats,
            "aggregate_metrics": aggregated_results["aggregate_metrics"],
            "per_cluster_results": aggregated_results["per_cluster"],
            "all_diagnoses": all_diagnoses
        }
    
    def _aggregate_multi_run_results(
        self, 
        all_run_results: dict[str, dict[str, list[EvaluationResult]]]
    ) -> dict:
        """
        Aggregate evaluation results across multiple runs.
        
        For each cluster and method, compute mean ± std of all metrics.
        
        Args:
            all_run_results: {cluster_key: {method: [results]}}
            
        Returns:
            Dictionary with aggregated metrics
        """
        import numpy as np
        
        per_cluster_results = []
        
        # Collect all metrics for overall aggregation
        individual_metrics = {
            "similarity_score": [], "cosine_similarity": [], "bert_score_f1": [],
            "category_match": [], "correct": [], "partial": []
        }
        collective_metrics = {
            "similarity_score": [], "cosine_similarity": [], "bert_score_f1": [],
            "category_match": [], "correct": [], "partial": []
        }
        
        for cluster_key, methods in all_run_results.items():
            cluster_result = {"cluster": cluster_key}
            
            for method in ["individual", "collective"]:
                results = methods.get(method, [])
                if not results:
                    continue
                
                # Extract metrics from all runs
                sim_scores = [r.similarity_score for r in results if r.similarity_score > 0]
                cos_scores = [r.cosine_similarity for r in results]
                bert_scores = [r.bert_score_f1 for r in results]
                cat_matches = [1 if r.category_match else 0 for r in results]
                correct = [1 if r.is_correct == "Yes" else 0 for r in results]
                partial = [1 if r.is_correct == "Partial" else 0 for r in results]
                
                # Compute mean ± std
                cluster_result[f"{method}_similarity_mean"] = np.mean(sim_scores) if sim_scores else 0
                cluster_result[f"{method}_similarity_std"] = np.std(sim_scores) if sim_scores else 0
                cluster_result[f"{method}_cosine_mean"] = np.mean(cos_scores) if cos_scores else 0
                cluster_result[f"{method}_cosine_std"] = np.std(cos_scores) if cos_scores else 0
                cluster_result[f"{method}_bert_f1_mean"] = np.mean(bert_scores) if bert_scores else 0
                cluster_result[f"{method}_bert_f1_std"] = np.std(bert_scores) if bert_scores else 0
                cluster_result[f"{method}_category_accuracy"] = np.mean(cat_matches) if cat_matches else 0
                cluster_result[f"{method}_correct_rate"] = np.mean(correct) if correct else 0
                cluster_result[f"{method}_partial_rate"] = np.mean(partial) if partial else 0
                
                # Collect for overall aggregation (use mean of each cluster)
                target = individual_metrics if method == "individual" else collective_metrics
                if sim_scores:
                    target["similarity_score"].append(np.mean(sim_scores))
                if cos_scores:
                    target["cosine_similarity"].append(np.mean(cos_scores))
                if bert_scores:
                    target["bert_score_f1"].append(np.mean(bert_scores))
                target["category_match"].append(np.mean(cat_matches))
                target["correct"].append(np.mean(correct))
                target["partial"].append(np.mean(partial))
            
            per_cluster_results.append(cluster_result)
        
        # Compute overall aggregate metrics
        def compute_overall(metrics: dict, prefix: str) -> dict:
            result = {}
            for key, values in metrics.items():
                if values:
                    result[f"{prefix}_{key}_mean"] = float(np.mean(values))
                    result[f"{prefix}_{key}_std"] = float(np.std(values))
            return result
        
        aggregate_metrics = {}
        aggregate_metrics.update(compute_overall(individual_metrics, "individual"))
        aggregate_metrics.update(compute_overall(collective_metrics, "collective"))
        
        # Compute improvement metrics
        if individual_metrics["similarity_score"] and collective_metrics["similarity_score"]:
            ind_mean = np.mean(individual_metrics["similarity_score"])
            col_mean = np.mean(collective_metrics["similarity_score"])
            aggregate_metrics["similarity_improvement"] = float(col_mean - ind_mean)
        
        if individual_metrics["cosine_similarity"] and collective_metrics["cosine_similarity"]:
            ind_mean = np.mean(individual_metrics["cosine_similarity"])
            col_mean = np.mean(collective_metrics["cosine_similarity"])
            aggregate_metrics["cosine_improvement"] = float(col_mean - ind_mean)
        
        if individual_metrics["bert_score_f1"] and collective_metrics["bert_score_f1"]:
            ind_mean = np.mean(individual_metrics["bert_score_f1"])
            col_mean = np.mean(collective_metrics["bert_score_f1"])
            aggregate_metrics["bert_score_improvement"] = float(col_mean - ind_mean)
        
        # Win/Loss analysis (based on mean similarity per cluster)
        wins = {"collective": 0, "individual": 0, "tie": 0}
        for result in per_cluster_results:
            ind_sim = result.get("individual_similarity_mean", 0)
            col_sim = result.get("collective_similarity_mean", 0)
            if col_sim > ind_sim:
                wins["collective"] += 1
            elif ind_sim > col_sim:
                wins["individual"] += 1
            else:
                wins["tie"] += 1
        
        aggregate_metrics["collective_wins"] = wins["collective"]
        aggregate_metrics["individual_wins"] = wins["individual"]
        aggregate_metrics["ties"] = wins["tie"]
        aggregate_metrics["num_clusters"] = len(per_cluster_results)
        aggregate_metrics["runs_per_cluster"] = self.runs_per_cluster
        
        return {
            "aggregate_metrics": aggregate_metrics,
            "per_cluster": per_cluster_results
        }
    
    def save_results(self, results: dict, output_dir: Path = RESULTS_DIR):
        """
        Save experiment results to files.
        
        Args:
            results: Experiment results dictionary
            output_dir: Directory to save results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        json_path = output_dir / f"rq1_results_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved full results to {json_path}")
        
        # Save comprehensive summary report
        report_path = output_dir / f"rq1_report_{timestamp}.txt"
        metrics = results.get("aggregate_metrics", {})
        
        with open(report_path, "w", encoding="utf-8") as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write("RQ1 EXPERIMENT: COLLECTIVE VS INDIVIDUAL ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            # Experiment Info
            f.write("EXPERIMENT INFO:\n")
            f.write("-" * 70 + "\n")
            for key, value in results["experiment_info"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Multi-run Summary
            f.write("MULTI-RUN EXPERIMENT SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Clusters: {metrics.get('num_clusters', 0)}\n")
            f.write(f"  Runs per cluster: {metrics.get('runs_per_cluster', 1)}\n")
            f.write("\n")
            
            # Dimension 1: Category Accuracy
            f.write("DIMENSION 1: CATEGORY ACCURACY\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Method':<15} {'Accuracy (mean ± std)':<25}\n")
            ind_cat = metrics.get('individual_category_match_mean', 0)
            ind_cat_std = metrics.get('individual_category_match_std', 0)
            col_cat = metrics.get('collective_category_match_mean', 0)
            col_cat_std = metrics.get('collective_category_match_std', 0)
            f.write(f"{'Individual':<15} {ind_cat:.3f} ± {ind_cat_std:.3f}\n")
            f.write(f"{'Collective':<15} {col_cat:.3f} ± {col_cat_std:.3f}\n")
            f.write("\n")
            
            # Dimension 2: Semantic Alignment
            f.write("DIMENSION 2: SEMANTIC ALIGNMENT (Automated)\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Method':<15} {'Cosine Sim':<20} {'BERTScore-F1':<20}\n")
            ind_cos = metrics.get('individual_cosine_similarity_mean', 0)
            ind_cos_std = metrics.get('individual_cosine_similarity_std', 0)
            col_cos = metrics.get('collective_cosine_similarity_mean', 0)
            col_cos_std = metrics.get('collective_cosine_similarity_std', 0)
            ind_bert = metrics.get('individual_bert_score_f1_mean', 0)
            ind_bert_std = metrics.get('individual_bert_score_f1_std', 0)
            col_bert = metrics.get('collective_bert_score_f1_mean', 0)
            col_bert_std = metrics.get('collective_bert_score_f1_std', 0)
            f.write(f"{'Individual':<15} {ind_cos:.3f} ± {ind_cos_std:.3f}      {ind_bert:.3f} ± {ind_bert_std:.3f}\n")
            f.write(f"{'Collective':<15} {col_cos:.3f} ± {col_cos_std:.3f}      {col_bert:.3f} ± {col_bert_std:.3f}\n")
            f.write(f"{'Improvement':<15} {metrics.get('cosine_improvement', 0):+.3f}               {metrics.get('bert_score_improvement', 0):+.3f}\n")
            f.write("\n")
            
            # Dimension 3: Diagnosis Quality (LLM-based)
            f.write("DIMENSION 3: DIAGNOSIS QUALITY (LLM-based 1-5 Scale)\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Method':<15} {'Similarity':<20} {'Correct Rate':<15} {'Partial Rate':<15}\n")
            ind_sim = metrics.get('individual_similarity_score_mean', 0)
            ind_sim_std = metrics.get('individual_similarity_score_std', 0)
            col_sim = metrics.get('collective_similarity_score_mean', 0)
            col_sim_std = metrics.get('collective_similarity_score_std', 0)
            ind_corr = metrics.get('individual_correct_mean', 0)
            col_corr = metrics.get('collective_correct_mean', 0)
            ind_part = metrics.get('individual_partial_mean', 0)
            col_part = metrics.get('collective_partial_mean', 0)
            f.write(f"{'Individual':<15} {ind_sim:.2f} ± {ind_sim_std:.2f}        {ind_corr:.1%}           {ind_part:.1%}\n")
            f.write(f"{'Collective':<15} {col_sim:.2f} ± {col_sim_std:.2f}        {col_corr:.1%}           {col_part:.1%}\n")
            f.write(f"{'Improvement':<15} {metrics.get('similarity_improvement', 0):+.2f}\n")
            f.write("\n")
            
            # Win/Loss Analysis
            f.write("WIN/LOSS ANALYSIS (Based on Mean Similarity per Cluster)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Collective Wins: {metrics.get('collective_wins', 0)}\n")
            f.write(f"  Individual Wins: {metrics.get('individual_wins', 0)}\n")
            f.write(f"  Ties: {metrics.get('ties', 0)}\n")
            f.write("\n")
            
            f.write("=" * 70 + "\n")
        
        self.logger.info(f"Saved summary report to {report_path}")
        
        # Save all diagnoses from all runs
        diagnoses_path = output_dir / f"rq1_diagnoses_{timestamp}.json"
        with open(diagnoses_path, "w", encoding="utf-8") as f:
            json.dump(results.get("all_diagnoses", []), f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved diagnoses to {diagnoses_path}")
        
        # Save per-cluster results
        per_cluster_path = output_dir / f"rq1_per_cluster_{timestamp}.json"
        with open(per_cluster_path, "w", encoding="utf-8") as f:
            json.dump(results.get("per_cluster_results", []), f, indent=2)
        self.logger.info(f"Saved per-cluster results to {per_cluster_path}")
        
        # Save LLM cost report
        cost_report = self.llm_client.get_cost_summary()
        cost_path = output_dir / f"rq1_cost_{timestamp}.json"
        with open(cost_path, "w", encoding="utf-8") as f:
            json.dump(cost_report, f, indent=2)
        self.logger.info(f"Saved cost report to {cost_path}")
        self.logger.info(f"Total LLM cost: ${cost_report.get('total_cost_usd', 0):.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run RQ1 Experiment: Automated Root Cause Diagnosis for Systemic Flakiness"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_LLM_PROVIDER,
        choices=["openai", "anthropic", "deepseek", "together", "groq"],
        help="LLM provider to use (openai, anthropic, deepseek, together, groq)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_CLUSTER,
        help="Number of runs per cluster for handling non-determinism"
    )
    
    parser.add_argument(
        "--max-tests",
        type=int,
        default=MAX_TESTS_PER_CLUSTER,
        help="Maximum number of tests to analyze per cluster"
    )
    
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Maximum number of clusters to analyze (for testing)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Only analyze clusters from this project"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show statistics without running LLM analysis"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = LOGS_DIR / f"rq1_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("RQ1 Experiment: Automated Root Cause Diagnosis")
    logger.info("=" * 60)
    
    # Load clusters
    logger.info("Loading clusters...")
    clusters = load_all_clusters()
    
    # Filter by project if specified
    if args.project:
        clusters = [c for c in clusters if c.project == args.project]
        logger.info(f"Filtered to {len(clusters)} clusters from {args.project}")
    
    # Limit number of clusters if specified
    if args.max_clusters:
        clusters = clusters[:args.max_clusters]
        logger.info(f"Limited to {len(clusters)} clusters")
    
    # Show statistics
    stats = get_cluster_statistics(clusters)
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    if args.dry_run:
        logger.info("\nDry run mode - not running LLM analysis")
        logger.info("\nSample clusters:")
        for cluster in clusters[:5]:
            logger.info(f"  {cluster.project}/cluster{cluster.cluster_id}: {cluster.size} tests")
            logger.info(f"    Ground Truth: {cluster.q3_answer}")
        return
    
    # Run experiment
    runner = ExperimentRunner(
        llm_provider=args.provider,
        runs_per_cluster=args.runs,
        max_tests=args.max_tests
    )
    
    try:
        results = runner.run_experiment(clusters)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        runner.save_results(results, output_dir)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        
        metrics = results["aggregate_metrics"]
        if metrics:
            logger.info(f"\nRuns per cluster: {metrics.get('runs_per_cluster', 1)}")
            logger.info(f"Total clusters: {metrics.get('num_clusters', 0)}")
            
            logger.info("\nKey Findings (mean ± std across runs):")
            
            # Similarity scores
            ind_sim = metrics.get('individual_similarity_score_mean', 0)
            ind_sim_std = metrics.get('individual_similarity_score_std', 0)
            col_sim = metrics.get('collective_similarity_score_mean', 0)
            col_sim_std = metrics.get('collective_similarity_score_std', 0)
            logger.info(f"  Individual Analysis Similarity: {ind_sim:.2f} ± {ind_sim_std:.2f}")
            logger.info(f"  Collective Analysis Similarity: {col_sim:.2f} ± {col_sim_std:.2f}")
            
            if "similarity_improvement" in metrics:
                logger.info(f"  Improvement (Collective - Individual): {metrics['similarity_improvement']:+.2f}")
            
            # Win/Loss
            if "collective_wins" in metrics:
                logger.info(f"\n  Collective Wins: {metrics['collective_wins']}")
                logger.info(f"  Individual Wins: {metrics['individual_wins']}")
                logger.info(f"  Ties: {metrics['ties']}")
    
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        if runner.evaluation_results:
            logger.info("Saving partial results...")
            partial_results = {
                "experiment_info": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "interrupted",
                    "llm_provider": args.provider
                },
                "aggregate_metrics": compute_aggregate_metrics(runner.evaluation_results),
                "detailed_results": [asdict(r) for r in runner.evaluation_results]
            }
            runner.save_results(partial_results, Path(args.output_dir))
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

