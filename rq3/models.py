"""
RQ3 Data Models

Data classes for representing flaky tests, repairs, and evaluation results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class DatasetSource(Enum):
    """Source dataset for a flaky test sample."""
    IDOFT = "idoft"
    LUO_2014 = "luo_2014"
    SYSTEMIC_FLAKINESS = "systemic_flakiness"


class RepairStatus(Enum):
    """Status of a repair in the dataset."""
    ACCEPTED = "Accepted"
    OPENED = "Opened"
    REJECTED = "Rejected"
    UNKNOWN = "Unknown"


@dataclass
class FlakyTestSample:
    """
    Represents a single flaky test sample with ground truth repair.
    
    This is the core data structure for RQ3 experiments.
    """
    # Identification
    sample_id: str
    dataset_source: DatasetSource
    
    # Test information
    project_url: str
    project_name: str
    test_name: str  # Fully-qualified test name
    sha_detected: str  # Commit SHA where flakiness was detected
    
    # Flaky test category (from IDoFT dataset)
    category: str  # e.g., "ID", "OD", "NOD" - behavioral category from IDoFT
    
    # Exception category (inferred by RQ2 Feature Extractor)
    # e.g., "Networking", "Filesystem", "Timeout", "Concurrency"
    exception_category: Optional[str] = None
    
    # Test code (before fix)
    test_code_before: Optional[str] = None
    
    # Ground truth repair
    pr_link: Optional[str] = None
    repair_status: RepairStatus = RepairStatus.UNKNOWN
    test_code_after: Optional[str] = None  # Code after fix
    repair_diff: Optional[str] = None  # Git diff of the fix
    
    # Additional context
    stack_trace: Optional[str] = None
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_ground_truth(self) -> bool:
        """Check if this sample has ground truth repair code."""
        return self.test_code_after is not None or self.repair_diff is not None
    
    @property
    def test_class(self) -> str:
        """Extract test class from fully-qualified name."""
        if "#" in self.test_name:
            return self.test_name.split("#")[0]
        return self.test_name.rsplit(".", 1)[0] if "." in self.test_name else self.test_name
    
    @property
    def test_method(self) -> str:
        """Extract test method from fully-qualified name."""
        if "#" in self.test_name:
            return self.test_name.split("#")[1]
        return self.test_name.rsplit(".", 1)[1] if "." in self.test_name else self.test_name


@dataclass
class RepairGenerationResult:
    """
    Result of LLM repair generation for a single sample.
    """
    sample_id: str
    
    # Input context
    test_code: str
    diagnosis: Optional[str] = None  # From RQ1
    cluster_info: Optional[str] = None  # From RQ2
    
    # LLM output
    generated_repair: Optional[str] = None
    repair_explanation: Optional[str] = None
    
    # Generation metadata
    method: str = "Ours"  # "Ours", "B1_zero_shot", "B2_individual"
    model_name: str = ""
    tokens_used: int = 0
    generation_time_ms: int = 0
    
    # Error handling
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if repair generation was successful."""
        return self.generated_repair is not None and self.error is None


@dataclass
class RepairEvaluationResult:
    """
    Evaluation result comparing generated repair to ground truth.
    """
    sample_id: str
    
    # Code-level metrics
    bleu_4: float = 0.0
    codebleu: float = 0.0
    edit_similarity: float = 0.0  # 1 - normalized_edit_distance
    
    # Optional: AST-level metrics
    ast_match_rate: Optional[float] = None
    
    # Optional: Functional validation (for subset)
    compiles: Optional[bool] = None
    
    # Metadata
    method: str = ""
    ground_truth_available: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "bleu_4": self.bleu_4,
            "codebleu": self.codebleu,
            "edit_similarity": self.edit_similarity,
            "ast_match_rate": self.ast_match_rate,
            "compiles": self.compiles,
            "method": self.method,
            "ground_truth_available": self.ground_truth_available,
        }


@dataclass
class ExperimentResult:
    """
    Aggregated results for an experiment run.
    """
    method: str
    total_samples: int
    
    # Aggregated metrics
    avg_bleu_4: float = 0.0
    avg_codebleu: float = 0.0
    avg_edit_similarity: float = 0.0
    
    # Per-category breakdown
    category_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Individual results
    individual_results: List[RepairEvaluationResult] = field(default_factory=list)
    
    # Cost tracking
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "total_samples": self.total_samples,
            "avg_bleu_4": self.avg_bleu_4,
            "avg_codebleu": self.avg_codebleu,
            "avg_edit_similarity": self.avg_edit_similarity,
            "category_results": self.category_results,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }
