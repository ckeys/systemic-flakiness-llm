"""
RQ3: LLM-Generated Repair Suggestions for Systemic Flaky Tests

This package implements the RQ3 experiment:
- Data preparation from IDoFT dataset (1000 samples)
- GitHub PR code extraction
- RQ2 clustering + RQ1 diagnosis pipeline
- Repair generation
- Evaluation metrics (BLEU, CodeBLEU, Exact Match)

Usage:
    # Run unit tests
    python -m rq3.data_loader
    python -m rq3.github_extractor
    python -m rq3.repair_generator
    python -m rq3.evaluation
    python -m rq3.baselines
    
    # Run experiment
    python -m rq3.run_experiment --samples 100 --dry-run
"""

from .config import (
    DEFAULT_SAMPLING_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    BASELINES,
    RQ3_RESULTS_DIR,
    RQ3_LOGS_DIR,
)

from .models import (
    FlakyTestSample,
    RepairGenerationResult,
    RepairEvaluationResult,
    ExperimentResult,
    DatasetSource,
    RepairStatus,
)

from .data_loader import (
    load_and_sample_idoft,
    get_idoft_statistics,
)

from .evaluation import (
    compute_bleu_4,
    compute_codebleu,
    compute_edit_similarity,
    evaluate_repair,
    aggregate_results,
)

__all__ = [
    # Config
    "DEFAULT_SAMPLING_CONFIG",
    "DEFAULT_EVALUATION_CONFIG",
    "BASELINES",
    "RQ3_RESULTS_DIR",
    "RQ3_LOGS_DIR",
    # Models
    "FlakyTestSample",
    "RepairGenerationResult",
    "RepairEvaluationResult",
    "ExperimentResult",
    "DatasetSource",
    "RepairStatus",
    # Data loading
    "load_and_sample_idoft",
    "get_idoft_statistics",
    # Evaluation
    "compute_bleu_4",
    "compute_codebleu",
    "compute_edit_similarity",
    "evaluate_repair",
    "aggregate_results",
]
