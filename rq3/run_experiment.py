#!/usr/bin/env python3
"""
Background experiment runner for RQ3 with precomputed clusters.
Saves all output to log files and results to JSON.
"""

import argparse
import json
import logging
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================================
# CODE TRUNCATION UTILITIES
# ============================================================================

def extract_test_method(full_code: str, test_name: str, max_fallback_len: int = 3000) -> str:
    """
    Extract specific test methods from the complete test class code.
    
    Args:
        full_code: Complete test class code
        test_name: Test name, format like "org.example.TestClass.testMethod"
        max_fallback_len: If method is not found, truncate to this length
    
    Returns:
        Extracted test method code, or truncated code if not found
    """
    if not full_code:
        return "[Code not available]"
    
    # Extract method name from test_name
    if '#' in test_name:
        method_name = test_name.split('#')[-1]
    elif '.' in test_name:
        method_name = test_name.split('.')[-1]
    else:
        method_name = test_name
    
    # Try matching test method with regex
    patterns = [
        # Match @Test annotation + method
        rf'(@Test[^\n]*\n\s*(?:public\s+)?(?:void\s+)?{re.escape(method_name)}\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{{)',
        # Match regular method definition
        rf'((?:public|private|protected)?\s*(?:static)?\s*\w+\s+{re.escape(method_name)}\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{{)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_code)
        if match:
            start = match.start()
            # Find the end position of the method (matching braces)
            brace_count = 0
            end = start
            in_string = False
            escape_next = False
            
            for i, char in enumerate(full_code[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
            
            if end > start:
                # Look forward for annotations
                annotation_start = start
                lines_before = full_code[:start].split('\n')
                for line in reversed(lines_before[-5:]):
                    stripped = line.strip()
                    if stripped.startswith('@') or stripped.startswith('//') or stripped == '':
                        pos = full_code.rfind(line, 0, start)
                        if pos >= 0:
                            annotation_start = pos
                    else:
                        break
                
                return full_code[annotation_start:end].strip()
    
    # If specific method not found, return truncated code
    if len(full_code) <= max_fallback_len:
        return full_code
    
    return full_code[:max_fallback_len] + "\n// ... (code truncated)"


def truncate_code_for_prompt(code: str, max_len: int = 4000) -> str:
    """
    Truncate code to fit prompt length limits.
    
    Args:
        code: Original code
        max_len: Maximum length
    
    Returns:
        Truncated code
    """
    if not code or len(code) <= max_len:
        return code or "[Code not available]"
    
    return code[:max_len] + "\n// ... (code truncated)"

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_client import create_llm_client, get_cost_tracker, reset_cost_tracker
from rq3.baselines import (
    run_baseline_b1_zero_shot,
    run_baseline_b2_individual,
)
from rq3.ml_clustering import train_model_from_rq2_data, cluster_samples_with_ml
from rq3.models import ExperimentResult, FlakyTestSample, RepairEvaluationResult

BASE_PATH = Path(__file__).parent.parent.parent
PRECOMPUTED_CLUSTERS_PATH = BASE_PATH / "datasets/idoft/rq3_precomputed_clusters.json"
SAMPLES_FILE = BASE_PATH / "datasets/idoft/rq3_experiment_samples_1000.json"


def load_all_samples() -> Tuple[List[FlakyTestSample], Dict[str, str]]:
    """Load all 1000 samples from the experiment file."""
    with open(SAMPLES_FILE) as f:
        data = json.load(f)
    
    samples = []
    ground_truths = {}
    
    for d in data["samples"]:
        sample = FlakyTestSample(
            sample_id=d["sample_id"],
            dataset_source="idoft",
            project_url=d["project_url"],
            project_name=d["project_name"],
            test_name=d["test_name"],
            sha_detected=d["sha_detected"],
            category=d["category"],
            pr_link=d.get("pr_link"),
            test_code_before=d.get("test_code_before"),
        )
        samples.append(sample)
        
        # Prefer full method from test_code_after as Ground Truth
        if d.get("test_code_after"):
            gt_method = extract_test_method(d["test_code_after"], d["test_name"])
            ground_truths[d["sample_id"]] = gt_method
        elif d.get("ground_truth_diff"):
            ground_truths[d["sample_id"]] = d["ground_truth_diff"]
        else:
            ground_truths[d["sample_id"]] = ""
    
    return samples, ground_truths


def setup_logging(output_path: Path):
    """Setup logging to both file and console."""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(output_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def load_or_compute_clusters(
    all_samples: List[FlakyTestSample], 
    logger
) -> Tuple[Dict[str, List[FlakyTestSample]], Dict[str, Any]]:
    """Load precomputed ML-based clusters or compute them on-the-fly."""
    
    # Build sample lookup by sample_id
    sample_by_id = {s.sample_id: s for s in all_samples}
    
    # Try to load precomputed ML-based clusters
    if PRECOMPUTED_CLUSTERS_PATH.exists():
        logger.info(f"Loading precomputed ML-based clusters from {PRECOMPUTED_CLUSTERS_PATH}")
        with open(PRECOMPUTED_CLUSTERS_PATH, 'r') as f:
            data = json.load(f)
        
        raw_clusters = data.get("clusters", {})
        metadata_from_file = data.get("metadata", {})
        logger.info(f"Loaded {metadata_from_file.get('total_clusters', 'N/A')} clusters from file")
        
        # Convert sample_ids to FlakyTestSample objects
        clusters = {}
        for cluster_id, sample_ids in raw_clusters.items():
            cluster_samples = []
            for sid in sample_ids:
                if sid in sample_by_id:
                    cluster_samples.append(sample_by_id[sid])
            if cluster_samples:
                clusters[cluster_id] = cluster_samples
        
        logger.info(f"Successfully loaded {len(clusters)} clusters with samples")
    else:
        logger.info("Precomputed clusters not found, computing on-the-fly...")
        logger.info("This uses project-based clustering (faster) instead of full ML clustering")
        
        # Use simple project-based clustering for speed
        from collections import defaultdict
        project_clusters = defaultdict(list)
        for sample in all_samples:
            project_clusters[sample.project_name].append(sample)
        
        clusters = {f"project_{i}_{name}": samples 
                   for i, (name, samples) in enumerate(project_clusters.items())}
        
        logger.info(f"Created {len(clusters)} project-based clusters")
    
    # Compute metadata
    total_samples = sum(len(c) for c in clusters.values())
    non_singleton = {k: v for k, v in clusters.items() if len(v) > 1}
    
    metadata = {
        "total_clusters": len(clusters),
        "total_samples": total_samples,
        "non_singleton_clusters": len(non_singleton),
        "samples_in_non_singleton": sum(len(c) for c in non_singleton.values()),
    }
    
    return clusters, metadata


def print_result(logger, name: str, result: ExperimentResult):
    """Print result metrics."""
    logger.info(f"  {name}: BLEU-4={result.avg_bleu_4:.4f}, CodeBLEU={result.avg_codebleu:.4f}, EditSim={result.avg_edit_similarity:.4f}")


def run_ours_with_precomputed(
    samples: List[FlakyTestSample],
    llm_client,
    ground_truths: Dict[str, str],
    precomputed_clusters: Dict[str, List[FlakyTestSample]],
    logger,
) -> ExperimentResult:
    """Run Ours method using precomputed clusters."""
    from rq3.baselines import (
        COLLECTIVE_DIAGNOSIS_PROMPT,
        generate_collective_repair,
        evaluate_repair,
        aggregate_results,
    )
    
    logger.info(f"Running Ours (CoFlaR) with precomputed clusters on {len(samples)} samples")
    
    # Build sample_id -> cluster mapping
    sample_ids = {s.sample_id for s in samples}
    relevant_clusters = {}
    
    for cluster_id, cluster_samples in precomputed_clusters.items():
        relevant_in_cluster = [s for s in cluster_samples if s.sample_id in sample_ids]
        if relevant_in_cluster:
            relevant_clusters[cluster_id] = relevant_in_cluster
    
    logger.info(f"Samples distributed across {len(relevant_clusters)} clusters")
    
    evaluations = []
    processed = 0
    
    for cluster_id, cluster_samples in relevant_clusters.items():
        if len(cluster_samples) == 0:
            continue
        
        # Build test details for collective diagnosis
        # Use code extraction to reduce prompt length
        test_details = []
        for i, s in enumerate(cluster_samples, 1):
            # Extract specific test method instead of the entire file
            extracted_code = extract_test_method(
                s.test_code_before or "", 
                s.test_name,
                max_fallback_len=3000
            )
            # Maintain consistent code processing with B2 (no second-layer truncation)
            # This preserves more code context, improving CodeBLEU and EditSim
            
            test_details.append(f"""### Test {i}: {s.test_name}
```java
{extracted_code}
```
Error: {s.stack_trace or s.error_message or '[No error info]'}
""")
        
        # Collective diagnosis for the cluster
        diagnosis_prompt = COLLECTIVE_DIAGNOSIS_PROMPT.format(
            num_tests=len(cluster_samples),
            test_details="\n".join(test_details)
        )
        llm_client.set_component("Ours_collective_diagnosis")
        diagnosis_response = llm_client.generate(diagnosis_prompt)
        
        # Get cluster test names for context
        cluster_test_names = [s.test_name for s in cluster_samples]
        
        # Generate repair for each sample in cluster
        for sample in cluster_samples:
            if sample.sample_id not in ground_truths:
                continue
            
            # Generate repair with the collective diagnosis
            repair_result = generate_collective_repair(
                sample, diagnosis_response, cluster_test_names, llm_client
            )
            
            # Evaluate
            gt = ground_truths.get(sample.sample_id, "")
            generated = repair_result.generated_repair or ""
            
            evaluation = evaluate_repair(
                generated=generated,
                ground_truth=gt,
                sample_id=sample.sample_id,
                method="Ours_CoFlaR",
            )
            evaluations.append(evaluation)
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"  Processed {processed}/{len(samples)} samples")
    
    if not evaluations:
        logger.warning("No evaluations produced!")
        return ExperimentResult(
            total_samples=0,
            avg_bleu_4=0.0,
            avg_codebleu=0.0,
            avg_edit_similarity=0.0,
            individual_results=[],
        )
    
    return aggregate_results(evaluations, "Ours_CoFlaR")


def run_experiment(
    n_samples: int,
    provider: str,
    min_cluster_size: int,
    seed: int,
    logger,
) -> Dict[str, ExperimentResult]:
    """Run the full experiment."""
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info("=" * 60)
    logger.info("RQ3 Experiment with Precomputed ML-based Clusters")
    logger.info("=" * 60)
    logger.info(f"Parameters: samples={n_samples}, provider={provider}, min_cluster_size={min_cluster_size}, seed={seed}")
    
    # Load all samples and ground truths first
    logger.info("Loading samples and ground truths...")
    all_samples, ground_truths = load_all_samples()
    logger.info(f"Loaded {len(all_samples)} samples with {len(ground_truths)} ground truths")
    
    # Load or compute clusters
    clusters, metadata = load_or_compute_clusters(all_samples, logger)
    logger.info(f"Total clusters: {metadata['total_clusters']} with {metadata['total_samples']} samples")
    logger.info(f"Non-singleton clusters: {metadata['non_singleton_clusters']} ({metadata['samples_in_non_singleton']} samples)")
    
    # Filter for non-singleton clusters
    non_singleton_clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
    
    # New sampling strategy: Ensure at least 2 samples are drawn from each cluster (if cluster has enough samples)
    # This avoids generating singletons after sampling
    logger.info("Using cluster-aware sampling to avoid post-sampling singletons...")
    
    # First, filter samples with ground truth
    clusters_with_gt = {}
    for cluster_id, cluster_samples in non_singleton_clusters.items():
        samples_with_gt = [s for s in cluster_samples if s.sample_id in ground_truths]
        if len(samples_with_gt) >= min_cluster_size:  # Keep only clusters with at least min_cluster_size samples having GT
            clusters_with_gt[cluster_id] = samples_with_gt
    
    logger.info(f"Clusters with >={min_cluster_size} samples having ground truth: {len(clusters_with_gt)}")
    total_eligible = sum(len(v) for v in clusters_with_gt.values())
    logger.info(f"Total eligible samples: {total_eligible}")
    
    # Sample from each cluster, ensuring each selected cluster has at least min_cluster_size samples
    experiment_samples = []
    selected_clusters = []
    
    if total_eligible <= n_samples:
        # If total samples are insufficient, use all samples
        logger.warning(f"Only {total_eligible} eligible samples, using all")
        for cluster_samples in clusters_with_gt.values():
            experiment_samples.extend(cluster_samples)
    else:
        # Randomly shuffle clusters
        cluster_list = list(clusters_with_gt.items())
        random.shuffle(cluster_list)
        
        remaining = n_samples
        for cluster_id, cluster_samples in cluster_list:
            if remaining <= 0:
                break
            
            # Determine how many samples to draw from this cluster
            # Draw at least min_cluster_size (if cluster has enough samples)
            min_to_sample = min(min_cluster_size, len(cluster_samples))
            max_to_sample = min(len(cluster_samples), remaining)
            
            if max_to_sample < min_to_sample:
                continue  # Remaining quota insufficient to draw the minimum number
            
            # Randomly decide the number of samples to draw (between min and max)
            n_to_sample = random.randint(min_to_sample, max_to_sample)
            
            sampled = random.sample(cluster_samples, n_to_sample)
            experiment_samples.extend(sampled)
            selected_clusters.append((cluster_id, n_to_sample))
            remaining -= n_to_sample
        
        logger.info(f"Sampled from {len(selected_clusters)} clusters")
    
    logger.info(f"Selected {len(experiment_samples)} samples for experiment")
    
    # Verification: Check cluster distribution after sampling
    sampled_ids = {s.sample_id for s in experiment_samples}
    post_sample_clusters = {}
    for cluster_id, cluster_samples in clusters.items():
        in_sample = [s for s in cluster_samples if s.sample_id in sampled_ids]
        if in_sample:
            post_sample_clusters[cluster_id] = in_sample
    
    singleton_count = sum(1 for v in post_sample_clusters.values() if len(v) == 1)
    logger.info(f"Post-sampling cluster distribution: {len(post_sample_clusters)} clusters, {singleton_count} singletons")
    
    # Get ground truths for selected samples
    exp_ground_truths = {s.sample_id: ground_truths[s.sample_id] for s in experiment_samples}
    
    # Create LLM client
    logger.info(f"Creating LLM client with provider: {provider}")
    reset_cost_tracker()
    llm_client = create_llm_client(provider)
    
    results = {}
    
    # Run B1: Zero-shot
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running B1: Zero-shot Repair")
    logger.info("=" * 60)
    start_time = time.time()
    results["B1_zero_shot"] = run_baseline_b1_zero_shot(experiment_samples, llm_client, exp_ground_truths)
    logger.info(f"B1 completed in {time.time() - start_time:.1f}s")
    print_result(logger, "B1", results["B1_zero_shot"])
    
    # Run B2: Individual Diagnosis
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running B2: Individual Diagnosis + Repair")
    logger.info("=" * 60)
    start_time = time.time()
    results["B2_individual"] = run_baseline_b2_individual(experiment_samples, llm_client, exp_ground_truths)
    logger.info(f"B2 completed in {time.time() - start_time:.1f}s")
    print_result(logger, "B2", results["B2_individual"])
    
    # Run Ours: ML-based Clustering
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running Ours (CoFlaR): ML-based Clustering + Collective Diagnosis")
    logger.info("=" * 60)
    start_time = time.time()
    results["Ours_CoFlaR"] = run_ours_with_precomputed(
        experiment_samples, llm_client, exp_ground_truths, clusters, logger
    )
    logger.info(f"Ours completed in {time.time() - start_time:.1f}s")
    print_result(logger, "Ours", results["Ours_CoFlaR"])
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Method':<25} {'BLEU-4':>10} {'CodeBLEU':>10} {'EditSim':>10}")
    logger.info("-" * 60)
    for method, result in results.items():
        logger.info(f"{method:<25} {result.avg_bleu_4:>10.4f} {result.avg_codebleu:>10.4f} {result.avg_edit_similarity:>10.4f}")
    
    # Cost tracking
    cost_tracker = get_cost_tracker()
    logger.info("")
    logger.info(f"Total API calls: {cost_tracker.total_calls}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens}")
    logger.info(f"Estimated cost: ${cost_tracker.total_cost:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RQ3 Background Experiment Runner")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Min cluster size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = BASE_PATH / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"rq3_experiment_{timestamp}.log"
    results_path = output_dir / f"rq3_experiment_{timestamp}.json"
    
    # Setup logging
    logger = setup_logging(log_path)
    
    logger.info(f"Experiment started at {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Results file: {results_path}")
    
    try:
        # Run experiment
        results = run_experiment(
            n_samples=args.samples,
            provider=args.provider,
            min_cluster_size=args.min_cluster_size,
            seed=args.seed,
            logger=logger,
        )
        
        # Save results to JSON
        results_dict = {
            "timestamp": timestamp,
            "parameters": {
                "samples": args.samples,
                "provider": args.provider,
                "min_cluster_size": args.min_cluster_size,
                "seed": args.seed,
            },
            "results": {
                method: {
                    "total_samples": r.total_samples,
                    "avg_bleu_4": r.avg_bleu_4,
                    "avg_codebleu": r.avg_codebleu,
                    "avg_edit_similarity": r.avg_edit_similarity,
                }
                for method, r in results.items()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Experiment completed at {datetime.now().isoformat()}")
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
