"""
RQ3 Data Loader

Load and prepare data from IDoFT dataset (1000 samples).
Handles:
- CSV parsing for IDoFT pr-data.csv
- Stratified sampling by category
- GitHub PR code extraction
"""

import csv
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .config import (
    IDOFT_PR_DATA,
    DEFAULT_SAMPLING_CONFIG,
    RQ3_CACHE_DIR,
)
from .models import (
    FlakyTestSample,
    DatasetSource,
    RepairStatus,
)

logger = logging.getLogger(__name__)


def parse_repair_status(status_str: str) -> RepairStatus:
    """Parse repair status string to enum."""
    status_map = {
        "Accepted": RepairStatus.ACCEPTED,
        "Opened": RepairStatus.OPENED,
        "Rejected": RepairStatus.REJECTED,
    }
    return status_map.get(status_str, RepairStatus.UNKNOWN)


def load_idoft_data(csv_path: Path = IDOFT_PR_DATA) -> List[Dict]:
    """
    Load IDoFT pr-data.csv file.
    
    Args:
        csv_path: Path to pr-data.csv
        
    Returns:
        List of dictionaries with test data
    """
    if not csv_path.exists():
        logger.error(f"IDoFT data file not found: {csv_path}")
        return []
    
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    logger.info(f"Loaded {len(data)} records from IDoFT")
    return data


def filter_accepted_fixes(data: List[Dict]) -> List[Dict]:
    """
    Filter IDoFT data to only include tests with accepted fixes.
    
    Args:
        data: Raw IDoFT data
        
    Returns:
        Filtered data with only "Accepted" status
    """
    accepted = [row for row in data if row.get("Status") == "Accepted"]
    logger.info(f"Filtered to {len(accepted)} tests with accepted fixes")
    return accepted


def get_category_distribution(data: List[Dict]) -> Dict[str, int]:
    """
    Get distribution of flaky test categories.
    
    Args:
        data: IDoFT data
        
    Returns:
        Dictionary mapping category to count
    """
    distribution = defaultdict(int)
    for row in data:
        category = row.get("Category", "Unknown")
        distribution[category] += 1
    return dict(distribution)


def stratified_sample(
    data: List[Dict],
    category_allocation: Dict[str, int],
    random_seed: int = 42
) -> List[Dict]:
    """
    Perform stratified sampling based on category allocation.
    
    Args:
        data: IDoFT data (filtered to accepted fixes)
        category_allocation: Dict mapping category to desired sample count
        random_seed: Random seed for reproducibility
        
    Returns:
        Sampled data
    """
    random.seed(random_seed)
    
    # Group by category
    by_category = defaultdict(list)
    for row in data:
        category = row.get("Category", "Unknown")
        by_category[category].append(row)
    
    # Sample from each category
    sampled = []
    for category, target_count in category_allocation.items():
        available = by_category.get(category, [])
        
        if len(available) <= target_count:
            # Take all available
            sampled.extend(available)
            if len(available) < target_count:
                logger.warning(
                    f"Category '{category}': requested {target_count}, "
                    f"only {len(available)} available"
                )
        else:
            # Random sample
            sampled.extend(random.sample(available, target_count))
    
    logger.info(f"Stratified sampling: {len(sampled)} samples selected")
    return sampled


def convert_to_samples(
    data: List[Dict],
    dataset_source: DatasetSource = DatasetSource.IDOFT
) -> List[FlakyTestSample]:
    """
    Convert raw IDoFT data to FlakyTestSample objects.
    
    Args:
        data: Raw IDoFT data rows
        dataset_source: Source dataset identifier
        
    Returns:
        List of FlakyTestSample objects
    """
    samples = []
    
    for i, row in enumerate(data):
        project_url = row.get("Project URL", "")
        
        # Extract project name from URL
        # e.g., "https://github.com/apache/hbase" -> "apache-hbase"
        project_name = project_url.rstrip("/").split("/")[-2:]
        project_name = "-".join(project_name) if len(project_name) == 2 else project_url
        
        sample = FlakyTestSample(
            sample_id=f"{dataset_source.value}_{i:04d}",
            dataset_source=dataset_source,
            project_url=project_url,
            project_name=project_name,
            test_name=row.get("Fully-Qualified Test Name (packageName.ClassName.methodName)", ""),
            sha_detected=row.get("SHA Detected", ""),
            category=row.get("Category", "Unknown"),
            pr_link=row.get("PR Link", None),
            repair_status=parse_repair_status(row.get("Status", "")),
            metadata={
                "module_path": row.get("Module Path", ""),
                "notes": row.get("Notes", ""),
            }
        )
        samples.append(sample)
    
    return samples


def load_and_sample_idoft(
    target_samples: int = 800,
    category_allocation: Optional[Dict[str, int]] = None,
    random_seed: int = 42
) -> List[FlakyTestSample]:
    """
    Load IDoFT data and perform stratified sampling.
    
    This is the main entry point for loading IDoFT data for RQ3.
    
    Args:
        target_samples: Target number of samples
        category_allocation: Optional custom category allocation
        random_seed: Random seed for reproducibility
        
    Returns:
        List of FlakyTestSample objects
    """
    # Load raw data
    raw_data = load_idoft_data()
    
    if not raw_data:
        return []
    
    # Filter to accepted fixes only
    accepted_data = filter_accepted_fixes(raw_data)
    
    # Log category distribution
    distribution = get_category_distribution(accepted_data)
    logger.info(f"Category distribution in accepted fixes: {distribution}")
    
    # Use default allocation if not provided
    if category_allocation is None:
        category_allocation = DEFAULT_SAMPLING_CONFIG.idoft_category_allocation
    
    # Perform stratified sampling
    sampled_data = stratified_sample(
        accepted_data,
        category_allocation,
        random_seed
    )
    
    # Convert to samples
    samples = convert_to_samples(sampled_data)
    
    return samples


def get_idoft_statistics() -> Dict:
    """
    Get statistics about the IDoFT dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    raw_data = load_idoft_data()
    
    if not raw_data:
        return {"error": "Could not load IDoFT data"}
    
    # Status distribution
    status_dist = defaultdict(int)
    for row in raw_data:
        status_dist[row.get("Status", "Unknown")] += 1
    
    # Category distribution (accepted only)
    accepted = filter_accepted_fixes(raw_data)
    category_dist = get_category_distribution(accepted)
    
    # Project distribution (accepted only)
    project_dist = defaultdict(int)
    for row in accepted:
        url = row.get("Project URL", "")
        project = url.rstrip("/").split("/")[-1] if url else "Unknown"
        project_dist[project] += 1
    
    return {
        "total_records": len(raw_data),
        "status_distribution": dict(status_dist),
        "accepted_count": len(accepted),
        "category_distribution": category_dist,
        "unique_projects": len(project_dist),
        "top_projects": dict(sorted(project_dist.items(), key=lambda x: -x[1])[:10]),
    }


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_parse_repair_status():
    """Test repair status parsing."""
    assert parse_repair_status("Accepted") == RepairStatus.ACCEPTED
    assert parse_repair_status("Opened") == RepairStatus.OPENED
    assert parse_repair_status("Rejected") == RepairStatus.REJECTED
    assert parse_repair_status("Unknown") == RepairStatus.UNKNOWN
    assert parse_repair_status("") == RepairStatus.UNKNOWN
    print("✓ test_parse_repair_status passed")


def test_stratified_sample():
    """Test stratified sampling."""
    # Create mock data
    mock_data = [
        {"Category": "ID", "test": f"test_id_{i}"} for i in range(100)
    ] + [
        {"Category": "OD", "test": f"test_od_{i}"} for i in range(50)
    ] + [
        {"Category": "NOD", "test": f"test_nod_{i}"} for i in range(30)
    ]
    
    allocation = {"ID": 20, "OD": 10, "NOD": 5}
    
    sampled = stratified_sample(mock_data, allocation, random_seed=42)
    
    # Check total count
    assert len(sampled) == 35, f"Expected 35, got {len(sampled)}"
    
    # Check category counts
    category_counts = defaultdict(int)
    for row in sampled:
        category_counts[row["Category"]] += 1
    
    assert category_counts["ID"] == 20, f"Expected 20 ID, got {category_counts['ID']}"
    assert category_counts["OD"] == 10, f"Expected 10 OD, got {category_counts['OD']}"
    assert category_counts["NOD"] == 5, f"Expected 5 NOD, got {category_counts['NOD']}"
    
    print("✓ test_stratified_sample passed")


def test_convert_to_samples():
    """Test conversion to FlakyTestSample."""
    mock_data = [{
        "Project URL": "https://github.com/apache/hbase",
        "Fully-Qualified Test Name (packageName.ClassName.methodName)": "org.apache.hadoop.hbase.TestClass#testMethod",
        "SHA Detected": "abc123",
        "Category": "ID",
        "Status": "Accepted",
        "PR Link": "https://github.com/apache/hbase/pull/123",
    }]
    
    samples = convert_to_samples(mock_data)
    
    assert len(samples) == 1
    sample = samples[0]
    
    assert sample.project_name == "apache-hbase"
    assert sample.test_name == "org.apache.hadoop.hbase.TestClass#testMethod"
    assert sample.category == "ID"
    assert sample.repair_status == RepairStatus.ACCEPTED
    assert sample.test_class == "org.apache.hadoop.hbase.TestClass"
    assert sample.test_method == "testMethod"
    
    print("✓ test_convert_to_samples passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("Running data_loader unit tests")
    print("=" * 60)
    
    test_parse_repair_status()
    test_stratified_sample()
    test_convert_to_samples()
    
    print("\n✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run unit tests
    run_all_tests()
    
    # Print dataset statistics
    print("\n" + "=" * 60)
    print("IDoFT Dataset Statistics")
    print("=" * 60)
    
    stats = get_idoft_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
