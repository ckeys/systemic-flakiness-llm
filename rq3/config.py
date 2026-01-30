"""
RQ3 Configuration: LLM-Generated Repair Suggestions

This module contains configuration for the RQ3 experiment:
- IDoFT dataset paths
- Sampling configuration (1000 samples from IDoFT)
- Evaluation settings
- Experiment runs configuration (3 runs for averaging)
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths
RQ3_ROOT = Path(__file__).parent
SRC_ROOT = RQ3_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

# Dataset paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
IDOFT_DIR = DATASETS_DIR / "idoft"
IDOFT_PR_DATA = IDOFT_DIR / "pr-data.csv"
IDOFT_GR_DATA = IDOFT_DIR / "gr-data.csv"

# Systemic Flakiness dataset (for reference)
SYSTEMIC_FLAKINESS_DIR = DATASETS_DIR / "systemic-flakiness" / "clusters_experiment"

# Output paths
RQ3_OUTPUT_DIR = SRC_ROOT / "output" / "rq3"
RQ3_RESULTS_DIR = RQ3_OUTPUT_DIR / "results"
RQ3_LOGS_DIR = RQ3_OUTPUT_DIR / "logs"
RQ3_CACHE_DIR = RQ3_OUTPUT_DIR / "cache"

# Create output directories
for dir_path in [RQ3_OUTPUT_DIR, RQ3_RESULTS_DIR, RQ3_LOGS_DIR, RQ3_CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================

@dataclass
class SamplingConfig:
    """Configuration for dataset sampling."""
    
    # Total samples (all from IDoFT)
    total_samples: int = 1000
    
    # Number of experiment runs for averaging
    num_runs: int = 3
    
    # IDoFT category-based stratified sampling
    # Based on persistent GitHub data (1,404 samples with diff)
    idoft_category_allocation: Dict[str, int] = None
    
    def __post_init__(self):
        if self.idoft_category_allocation is None:
            # Based on IDoFT persistent data distribution
            self.idoft_category_allocation = {
                "ID": 700,       # Implementation-Dependent (largest, 1471 available)
                "OD-Vic": 120,   # Order-Dependent Victim (173 available)
                "OD": 70,        # Order-Dependent (88 available)
                "NOD": 24,       # Non-Deterministic (24 available)
                "NIO": 23,       # Non-Idempotent-Outcome (23 available)
                "OD-Brit": 20,   # Order-Dependent Brittle
                "NDOD": 15,      # Non-Deterministic Order-Dependent
                "TZD": 15,       # Timezone-Dependent
                "UD": 13,        # Unknown-Dependent
            }

# Default sampling config
DEFAULT_SAMPLING_CONFIG = SamplingConfig()


# ============================================================================
# FLAKY TEST CATEGORIES
# ============================================================================

# IDoFT flaky test categories with descriptions
FLAKY_CATEGORIES = {
    "ID": "Implementation-Dependent: Test depends on implementation details",
    "OD": "Order-Dependent: Test outcome depends on execution order",
    "OD-Vic": "Order-Dependent Victim: Test fails when run after certain tests",
    "OD-Brit": "Order-Dependent Brittle: Test fails when run before certain tests",
    "NOD": "Non-Deterministic: Test has random/non-deterministic behavior",
    "NIO": "Non-Idempotent-Outcome: Test outcome changes on re-execution",
    "NDOD": "Non-Deterministic Order-Dependent: Combination of NOD and OD",
    "TZD": "Timezone-Dependent: Test depends on system timezone",
    "UD": "Unknown-Dependent: Unknown dependency causing flakiness",
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for repair evaluation."""
    
    # Code-level metrics thresholds
    bleu_threshold: float = 0.40
    codebleu_threshold: float = 0.45
    
    # Functional validation (subset)
    functional_validation_samples: int = 100
    
    # Random seed for reproducibility
    random_seed: int = 42

DEFAULT_EVALUATION_CONFIG = EvaluationConfig()


# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================

# Baselines for RQ3
BASELINES = {
    "B1_zero_shot": {
        "name": "Zero-Shot Repair",
        "description": "Direct LLM repair without diagnosis",
        "uses_clustering": False,
        "uses_diagnosis": False,
    },
    "B2_individual": {
        "name": "Individual Diagnosis + Repair",
        "description": "Single test diagnosis then repair",
        "uses_clustering": False,
        "uses_diagnosis": True,
    },
    "Ours": {
        "name": "Full Pipeline (RQ2 + RQ1 + Repair)",
        "description": "Clustering + Collective Diagnosis + Repair",
        "uses_clustering": True,
        "uses_diagnosis": True,
    },
}


# ============================================================================
# GITHUB API CONFIGURATION
# ============================================================================

# GitHub API rate limiting
GITHUB_RATE_LIMIT_DELAY = 1.0  # seconds between requests
GITHUB_MAX_RETRIES = 3

# GitHub API token (optional, increases rate limit)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
