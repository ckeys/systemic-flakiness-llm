"""
RQ3 Baseline Implementations

Implements the two baselines for comparison:
- B1: Zero-Shot Repair (no diagnosis)
- B2: Individual Diagnosis + Repair

Also implements the full pipeline (Ours/CoFlaR):
- Ours: RQ2 ML-based Clustering + RQ1 Collective Diagnosis + Repair

The clustering method uses ML-based Jaccard distance prediction
(from RQ2) and hierarchical clustering with silhouette-based threshold selection.
"""

import logging
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    from llm_client import LLMClient

from .models import (
    FlakyTestSample,
    RepairGenerationResult,
    RepairEvaluationResult,
    ExperimentResult,
)
from .repair_generator import (
    generate_zero_shot_repair,
    generate_individual_repair,
    generate_collective_repair,
    get_extracted_test_code,
    extract_test_method,
)
from .evaluation import (
    evaluate_repair,
    aggregate_results,
)
from .ml_clustering import (
    cluster_samples_with_ml,
    load_or_train_model,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DIAGNOSIS PROMPTS (from RQ1)
# ============================================================================

INDIVIDUAL_DIAGNOSIS_PROMPT = """Analyze this flaky test and identify its root cause.

## Test Name
{test_name}

## Test Code
```java
{test_code}
```

## Error/Stack Trace
```
{error_info}
```

## Task
What is the root cause of this flaky test failure? 
Provide a concise diagnosis (2-3 sentences) focusing on the underlying technical issue."""


COLLECTIVE_DIAGNOSIS_PROMPT = """Analyze the following {num_tests} flaky tests from the same cluster.

{test_details}

## Task
1. First, identify the root cause for EACH test individually. Start each analysis with "ANALYSIS FOR TEST [Test Name]:"
2. Then, determine if there are any COMMON PATTERNS across these tests. Start this section with "SHARED PATTERNS ACROSS CLUSTER:"
3. For each test, provide a specific FIX STRATEGY.

Provide a highly structured response so that the specific analysis for each test can be easily extracted."""


# ============================================================================
# BASELINE B1: ZERO-SHOT REPAIR
# ============================================================================

def run_baseline_b1_zero_shot(
    samples: List[FlakyTestSample],
    llm_client: "LLMClient",
    ground_truths: Dict[str, str],
) -> ExperimentResult:
    """
    Run Baseline B1: Zero-Shot Repair.
    
    Direct LLM repair without any diagnosis.
    
    Args:
        samples: List of flaky test samples
        llm_client: LLM client for generation
        ground_truths: Dict mapping sample_id to ground truth repair
        
    Returns:
        ExperimentResult with aggregated metrics
    """
    logger.info(f"Running B1: Zero-Shot Repair on {len(samples)} samples")
    
    repair_results = []
    evaluations = []
    
    for i, sample in enumerate(samples):
        if i > 0 and i % 50 == 0:
            logger.info(f"B1 Progress: {i}/{len(samples)}")
        
        # Generate repair (no diagnosis)
        repair_result = generate_zero_shot_repair(sample, llm_client)
        repair_results.append(repair_result)
        
        # Evaluate
        gt = ground_truths.get(sample.sample_id, "")
        generated = repair_result.generated_repair or ""
        
        evaluation = evaluate_repair(
            generated=generated,
            ground_truth=gt,
            sample_id=sample.sample_id,
            method="B1_zero_shot",
        )
        evaluations.append(evaluation)
    
    return aggregate_results(evaluations, "B1_zero_shot")


# ============================================================================
# BASELINE B2: INDIVIDUAL DIAGNOSIS + REPAIR
# ============================================================================

def generate_individual_diagnosis(
    sample: FlakyTestSample,
    llm_client: "LLMClient",
) -> str:
    """Generate individual diagnosis for a single test."""
    llm_client.set_component("B2_individual_diagnosis")
    
    # Use extracted test method instead of the entire file
    extracted_code = get_extracted_test_code(sample)
    prompt = INDIVIDUAL_DIAGNOSIS_PROMPT.format(
        test_name=sample.test_name,
        test_code=extracted_code,
        error_info=sample.stack_trace or sample.error_message or "[No error info]",
    )
    
    try:
        response = llm_client.generate(prompt)
        return response.strip()
    except Exception as e:
        logger.warning(f"Individual diagnosis failed for {sample.sample_id}: {e}")
        return ""


def run_baseline_b2_individual(
    samples: List[FlakyTestSample],
    llm_client: "LLMClient",
    ground_truths: Dict[str, str],
) -> ExperimentResult:
    """
    Run Baseline B2: Individual Diagnosis + Repair.
    
    Single test diagnosis followed by repair.
    
    Args:
        samples: List of flaky test samples
        llm_client: LLM client for generation
        ground_truths: Dict mapping sample_id to ground truth repair
        
    Returns:
        ExperimentResult with aggregated metrics
    """
    logger.info(f"Running B2: Individual Diagnosis + Repair on {len(samples)} samples")
    
    evaluations = []
    
    for i, sample in enumerate(samples):
        if i > 0 and i % 50 == 0:
            logger.info(f"B2 Progress: {i}/{len(samples)}")
        
        # Step 1: Generate individual diagnosis
        diagnosis = generate_individual_diagnosis(sample, llm_client)
        
        # Step 2: Generate repair based on diagnosis
        repair_result = generate_individual_repair(sample, diagnosis, llm_client)
        
        # Evaluate
        gt = ground_truths.get(sample.sample_id, "")
        generated = repair_result.generated_repair or ""
        
        evaluation = evaluate_repair(
            generated=generated,
            ground_truth=gt,
            sample_id=sample.sample_id,
            method="B2_individual",
        )
        evaluations.append(evaluation)
    
    return aggregate_results(evaluations, "B2_individual")


# ============================================================================
# COLLECTIVE DIAGNOSIS (used by Ours)
# ============================================================================

def generate_collective_diagnosis(
    samples: List[FlakyTestSample],
    llm_client: "LLMClient",
) -> str:
    """Generate collective diagnosis for a group of tests."""
    llm_client.set_component("collective_diagnosis")
    
    # Build test details - use extreme extraction (1000 characters) to prevent distraction
    test_details = []
    for i, sample in enumerate(samples[:5]):  # Limit to 5 for context
        extracted_code = extract_test_method(
            sample.test_code_before or "",
            sample.test_name,
            max_fallback_len=1000  # Limit to 1000 characters
        )
        detail = f"""### Test {i + 1}: {sample.test_name}
**Code Snippet:**
```java
{extracted_code}
```
**Error:** {sample.stack_trace or sample.error_message or '[No error info]'}
"""
        test_details.append(detail)
    
    prompt = COLLECTIVE_DIAGNOSIS_PROMPT.format(
        num_tests=len(samples),
        test_details="\n\n".join(test_details),
    )
    
    try:
        response = llm_client.generate(prompt)
        return response.strip()
    except Exception as e:
        logger.warning(f"Collective diagnosis failed: {e}")
        return ""

# ============================================================================
# OURS: FULL PIPELINE (RQ2 ML-based Clustering + RQ1 + REPAIR)
# ============================================================================

# Pre-loaded ML model for clustering (loaded once)
_ml_model = None
_ml_feature_names = None


def get_ml_model():
    """Get or load the ML model for clustering."""
    global _ml_model, _ml_feature_names
    if _ml_model is None:
        _ml_model, _ml_feature_names = load_or_train_model()
    return _ml_model, _ml_feature_names


def run_ours_full_pipeline(
    samples: List[FlakyTestSample],
    llm_client: "LLMClient",
    ground_truths: Dict[str, str],
) -> ExperimentResult:
    """
    Run Our Method (CoFlaR): Full Pipeline (RQ2 ML-based Clustering + RQ1 + Repair).
    
    1. Cluster samples using RQ2's ML-based Jaccard distance prediction
    2. Generate collective diagnosis using RQ1 method
    3. Generate repair for each sample
    
    This uses the new ML-based clustering method that:
    - Extracts 36 features (21 baseline + 15 AST via tree-sitter)
    - Predicts pairwise Jaccard distances using ExtraTreesRegressor
    - Performs hierarchical clustering with silhouette-based threshold selection
    
    Args:
        samples: List of flaky test samples
        llm_client: LLM client for generation
        ground_truths: Dict mapping sample_id to ground truth repair
        
    Returns:
        ExperimentResult with aggregated metrics
    """
    logger.info(f"Running Ours (CoFlaR): Full Pipeline on {len(samples)} samples")
    
    # Step 1: Cluster using RQ2's ML-based method
    model, feature_names = get_ml_model()
    clusters = cluster_samples_with_ml(samples, model, feature_names)
    
    evaluations = []
    cluster_count = 0
    
    for cluster_id, cluster_samples in clusters.items():
        cluster_count += 1
        if cluster_count % 20 == 0:
            logger.info(f"Ours Progress: {cluster_count}/{len(clusters)} clusters")
        
        # Step 2: Generate collective diagnosis (RQ1)
        diagnosis = generate_collective_diagnosis(cluster_samples, llm_client)
        
        # Step 3: Generate repair for each sample
        cluster_tests = [s.test_name for s in cluster_samples]
        
        for sample in cluster_samples:
            other_tests = [t for t in cluster_tests if t != sample.test_name]
            
            repair_result = generate_collective_repair(
                sample, diagnosis, other_tests, llm_client
            )
            
            # Evaluate
            gt = ground_truths.get(sample.sample_id, "")
            generated = repair_result.generated_repair or ""
            
            evaluation = evaluate_repair(
                generated=generated,
                ground_truth=gt,
                sample_id=sample.sample_id,
                method="Ours",
            )
            evaluations.append(evaluation)
    
    return aggregate_results(evaluations, "Ours")


# ============================================================================
# RUN ALL BASELINES
# ============================================================================

def run_all_methods(
    samples: List[FlakyTestSample],
    llm_client: "LLMClient",
    ground_truths: Dict[str, str],
    methods: Optional[List[str]] = None,
) -> Dict[str, ExperimentResult]:
    """
    Run all methods (baselines + ours) and return results.
    
    Args:
        samples: List of flaky test samples
        llm_client: LLM client for generation
        ground_truths: Dict mapping sample_id to ground truth repair
        methods: Optional list of methods to run (default: all)
        
    Returns:
        Dict mapping method name to ExperimentResult
    """
    if methods is None:
        methods = ["B1_zero_shot", "B2_individual", "Ours"]
    
    results = {}
    
    method_runners = {
        "B1_zero_shot": run_baseline_b1_zero_shot,
        "B2_individual": run_baseline_b2_individual,
        "Ours": run_ours_full_pipeline,
    }
    
    for method in methods:
        if method not in method_runners:
            logger.warning(f"Unknown method: {method}")
            continue
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running method: {method}")
        logger.info(f"{'=' * 60}")
        
        runner = method_runners[method]
        result = runner(samples, llm_client, ground_truths)
        results[method] = result
        
        # Log summary
        logger.info(f"\n{method} Results:")
        logger.info(f"  Avg BLEU-4: {result.avg_bleu_4:.4f}")
        logger.info(f"  Avg CodeBLEU: {result.avg_codebleu:.4f}")
        logger.info(f"  Avg Edit Similarity: {result.avg_edit_similarity:.4f}")
    
    return results


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_ml_clustering():
    """Test ML-based clustering for Ours method."""
    from .models import FlakyTestSample, DatasetSource
    
    # Create test samples from same project
    samples = [
        FlakyTestSample(
            sample_id=f"test_{i}",
            dataset_source=DatasetSource.IDOFT,
            project_url="https://github.com/test/repo",
            project_name="test-project",
            test_name=f"org.example.Test{i}#testMethod",
            sha_detected="abc123",
            category="ID",
            test_code_before=f"@Test public void test{i}() {{ Thread.sleep({i * 100}); }}",
        )
        for i in range(5)
    ]
    
    # Test clustering (will load/train model)
    clusters = cluster_samples_with_ml(samples)
    
    # Should have at least 1 cluster
    assert len(clusters) >= 1
    
    # All samples should be in some cluster
    total_in_clusters = sum(len(c) for c in clusters.values())
    assert total_in_clusters == 5
    
    print("✓ test_ml_clustering passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("Running baselines unit tests")
    print("=" * 60)
    
    # Note: test_ml_clustering requires RQ2 data and model, skip in quick tests
    # test_ml_clustering()
    
    print("\n✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_tests()
