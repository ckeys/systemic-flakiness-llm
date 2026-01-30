"""
RQ2 Clustering Methods: Pre-clustering and Baselines

This module implements:
1. Our Hybrid Method: Signature-based pre-clustering
2. Baseline B1: Random Clustering
3. Baseline B2: Test Class-based Clustering
4. Baseline B3: Exception Type-only Clustering
5. Baseline B4: Embedding-based Clustering
6. Baseline B5: Pure LLM Clustering
"""

from __future__ import annotations

import random
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from rq2_feature_extractor import TestFeatures, FeatureExtractor
from llm_client import LLMClient

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PredictedCluster:
    """A predicted cluster of tests."""
    cluster_id: int
    tests: List[str] = field(default_factory=list)  # Test names
    signature: str = ""                              # Clustering signature (for our method)
    predicted_category: str = ""                     # Predicted root cause category
    method: str = ""                                 # Which method created this cluster


@dataclass
class ClusteringResult:
    """Result of a clustering method."""
    method_name: str
    clusters: List[PredictedCluster]
    test_to_cluster: Dict[str, int]  # Mapping from test name to cluster ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_clusters(self) -> int:
        return len(self.clusters)
    
    @property
    def num_tests(self) -> int:
        return len(self.test_to_cluster)
    
    def get_cluster_sizes(self) -> List[int]:
        return [len(c.tests) for c in self.clusters]


# ============================================================================
# OUR METHOD: SIGNATURE-BASED PRE-CLUSTERING
# ============================================================================

def signature_based_clustering(
    features_list: List[TestFeatures],
    min_cluster_size: int = 1
) -> ClusteringResult:
    """
    Our Hybrid Method (Phase 2): Signature-based pre-clustering.
    
    Groups tests by their signature (exception_type:category:entry_point).
    
    Args:
        features_list: List of TestFeatures from feature extraction
        min_cluster_size: Minimum cluster size (default 1, set to 2 to filter singletons)
        
    Returns:
        ClusteringResult with predicted clusters
    """
    # Group by signature
    signature_groups: Dict[str, List[TestFeatures]] = defaultdict(list)
    
    for features in features_list:
        signature_groups[features.signature].append(features)
    
    # Create clusters
    clusters = []
    test_to_cluster = {}
    cluster_id = 0
    
    for signature, group in signature_groups.items():
        if len(group) < min_cluster_size:
            continue
        
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
    
    # Handle tests not in any cluster (singletons when min_cluster_size > 1)
    for features in features_list:
        if features.test_name not in test_to_cluster:
            # Create singleton cluster
            cluster = PredictedCluster(
                cluster_id=cluster_id,
                tests=[features.test_name],
                signature=features.signature,
                predicted_category=features.exception_category,
                method="signature"
            )
            clusters.append(cluster)
            test_to_cluster[features.test_name] = cluster_id
            cluster_id += 1
    
    return ClusteringResult(
        method_name="Hybrid-Signature",
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={
            "num_unique_signatures": len(signature_groups),
            "min_cluster_size": min_cluster_size
        }
    )


# ============================================================================
# BASELINE B1: RANDOM CLUSTERING
# ============================================================================

def random_clustering(
    test_names: List[str],
    num_clusters: int,
    seed: Optional[int] = None
) -> ClusteringResult:
    """
    Baseline B1: Random Clustering.
    
    Randomly assigns tests to K clusters.
    
    Args:
        test_names: List of test names
        num_clusters: Number of clusters (K)
        seed: Random seed for reproducibility
        
    Returns:
        ClusteringResult with random clusters
    """
    if seed is not None:
        random.seed(seed)
    
    clusters = [
        PredictedCluster(cluster_id=i, tests=[], method="random")
        for i in range(num_clusters)
    ]
    test_to_cluster = {}
    
    for test_name in test_names:
        cluster_id = random.randint(0, num_clusters - 1)
        clusters[cluster_id].tests.append(test_name)
        test_to_cluster[test_name] = cluster_id
    
    # Remove empty clusters
    clusters = [c for c in clusters if c.tests]
    
    # Re-index
    for i, cluster in enumerate(clusters):
        old_id = cluster.cluster_id
        cluster.cluster_id = i
        for test in cluster.tests:
            test_to_cluster[test] = i
    
    return ClusteringResult(
        method_name="B1-Random",
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={"num_clusters_requested": num_clusters, "seed": seed}
    )


# ============================================================================
# BASELINE B2: TEST CLASS-BASED CLUSTERING
# ============================================================================

def test_class_clustering(test_names: List[str]) -> ClusteringResult:
    """
    Baseline B2: Test Class-based Clustering.
    
    Groups tests by their test class name.
    
    Args:
        test_names: List of test names (e.g., "org.package.Class#method")
        
    Returns:
        ClusteringResult with class-based clusters
    """
    class_groups: Dict[str, List[str]] = defaultdict(list)
    
    for test_name in test_names:
        # Extract class name
        if "#" in test_name:
            test_class = test_name.rsplit("#", 1)[0]
        else:
            test_class = test_name.rsplit(".", 1)[0] if "." in test_name else test_name
        
        class_groups[test_class].append(test_name)
    
    clusters = []
    test_to_cluster = {}
    
    for i, (test_class, tests) in enumerate(class_groups.items()):
        cluster = PredictedCluster(
            cluster_id=i,
            tests=tests,
            signature=test_class,
            method="test_class"
        )
        clusters.append(cluster)
        
        for test in tests:
            test_to_cluster[test] = i
    
    return ClusteringResult(
        method_name="B2-TestClass",
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={"num_unique_classes": len(class_groups)}
    )


# ============================================================================
# BASELINE B3: EXCEPTION TYPE-ONLY CLUSTERING
# ============================================================================

def exception_type_clustering(features_list: List[TestFeatures]) -> ClusteringResult:
    """
    Baseline B3: Exception Type-only Clustering.
    
    Groups tests only by their Exception type (ignoring message, stack trace).
    
    Args:
        features_list: List of TestFeatures
        
    Returns:
        ClusteringResult with exception-type-based clusters
    """
    type_groups: Dict[str, List[str]] = defaultdict(list)
    
    for features in features_list:
        exc_type = features.exception_simple_name or "Unknown"
        type_groups[exc_type].append(features.test_name)
    
    clusters = []
    test_to_cluster = {}
    
    for i, (exc_type, tests) in enumerate(type_groups.items()):
        cluster = PredictedCluster(
            cluster_id=i,
            tests=tests,
            signature=exc_type,
            method="exception_type"
        )
        clusters.append(cluster)
        
        for test in tests:
            test_to_cluster[test] = i
    
    return ClusteringResult(
        method_name="B3-ExceptionType",
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={"num_unique_types": len(type_groups)}
    )


# ============================================================================
# BASELINE B4: EMBEDDING-BASED CLUSTERING
# ============================================================================

def embedding_clustering(
    features_list: List[TestFeatures],
    stack_traces: Dict[str, str],
    distance_threshold: float = 0.5,  # Reduced from 1.5 - cosine distance range is 0-2
    model_name: str = "all-MiniLM-L6-v2"
) -> ClusteringResult:
    """
    Baseline B4: Embedding-based Clustering.
    
    Uses sentence embeddings to cluster tests based on their exception info.
    
    Args:
        features_list: List of TestFeatures
        stack_traces: Dict mapping test name to stack trace
        distance_threshold: Distance threshold for clustering
        model_name: Sentence transformer model name
        
    Returns:
        ClusteringResult with embedding-based clusters
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.error("Run: pip install sentence-transformers scikit-learn")
        # Return fallback: each test in its own cluster
        return _fallback_singleton_clustering(features_list, "B4-Embedding")
    
    if not features_list:
        return ClusteringResult(
            method_name="B4-Embedding",
            clusters=[],
            test_to_cluster={},
            metadata={"error": "No features provided"}
        )
    
    # Load model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Build text representations
    texts = []
    for features in features_list:
        trace = stack_traces.get(features.test_name, "")
        text = f"{features.exception_type}: {features.exception_message}\n{trace[:500]}"
        texts.append(text)
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} tests...")
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # Hierarchical clustering
    logger.info("Running hierarchical clustering...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="average",
        metric="cosine"
    )
    labels = clustering.fit_predict(embeddings)
    
    # Build clusters
    label_to_tests: Dict[int, List[str]] = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_tests[label].append(features_list[i].test_name)
    
    clusters = []
    test_to_cluster = {}
    
    for i, (label, tests) in enumerate(label_to_tests.items()):
        cluster = PredictedCluster(
            cluster_id=i,
            tests=tests,
            method="embedding"
        )
        clusters.append(cluster)
        
        for test in tests:
            test_to_cluster[test] = i
    
    return ClusteringResult(
        method_name="B4-Embedding",
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={
            "model": model_name,
            "distance_threshold": distance_threshold,
            "num_clusters": len(clusters)
        }
    )


# ============================================================================
# BASELINE B5: PURE LLM CLUSTERING
# ============================================================================

def pure_llm_clustering(
    features_list: List[TestFeatures],
    stack_traces: Dict[str, str],
    llm_client: LLMClient,
    max_tests_per_batch: int = 20
) -> ClusteringResult:
    """
    Baseline B5: Pure LLM Clustering.
    
    Directly asks LLM to cluster tests based on their exception info.
    Due to context limits, processes tests in batches.
    
    Args:
        features_list: List of TestFeatures
        stack_traces: Dict mapping test name to stack trace
        llm_client: LLM client for clustering
        max_tests_per_batch: Maximum tests per LLM call
        
    Returns:
        ClusteringResult with LLM-based clusters
    """
    if not features_list:
        return ClusteringResult(
            method_name="B5-PureLLM",
            clusters=[],
            test_to_cluster={},
            metadata={"error": "No features provided"}
        )
    
    # Set component for cost tracking
    llm_client.set_component("B5_pure_llm_clustering")
    
    all_clusters: List[List[str]] = []
    
    # Process in batches
    for batch_start in range(0, len(features_list), max_tests_per_batch):
        batch = features_list[batch_start:batch_start + max_tests_per_batch]
        
        # Build prompt
        tests_info = []
        for i, features in enumerate(batch):
            exc_info = f"{features.exception_type}: {features.exception_message}"
            tests_info.append(f"[{i+1}] {features.test_name}\n    Exception: {exc_info}")
        
        prompt = f"""You are analyzing {len(batch)} flaky tests to group them by shared root cause.

Tests:
{chr(10).join(tests_info)}

Group these tests into clusters where tests in the same cluster likely share the same root cause for their flakiness.

Reply ONLY in this format (use test numbers):
Group 1: [1, 3, 5]
Group 2: [2, 4]
Group 3: [6]
...

Important: Every test number must appear in exactly one group."""

        try:
            response = llm_client.generate(prompt)
            batch_clusters = _parse_llm_clustering_response(response, batch)
            all_clusters.extend(batch_clusters)
            
        except Exception as e:
            logger.warning(f"LLM clustering failed for batch: {e}")
            # Fallback: each test in its own cluster
            for features in batch:
                all_clusters.append([features.test_name])
    
    # Convert to ClusteringResult
    clusters = []
    test_to_cluster = {}
    
    for i, test_list in enumerate(all_clusters):
        cluster = PredictedCluster(
            cluster_id=i,
            tests=test_list,
            method="pure_llm"
        )
        clusters.append(cluster)
        
        for test in test_list:
            test_to_cluster[test] = i
    
    return ClusteringResult(
        method_name="B5-PureLLM",
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={
            "max_tests_per_batch": max_tests_per_batch,
            "num_batches": (len(features_list) + max_tests_per_batch - 1) // max_tests_per_batch
        }
    )


def _parse_llm_clustering_response(
    response: str, 
    batch: List[TestFeatures]
) -> List[List[str]]:
    """Parse LLM clustering response into list of test name lists."""
    import re
    
    clusters = []
    assigned = set()
    
    # Pattern: Group N: [1, 2, 3] or Group N: 1, 2, 3
    for line in response.strip().split("\n"):
        # Extract numbers from the line
        numbers = re.findall(r"\d+", line)
        if numbers and "group" in line.lower():
            group_tests = []
            for num_str in numbers[1:]:  # Skip group number
                idx = int(num_str) - 1  # Convert to 0-based
                if 0 <= idx < len(batch) and idx not in assigned:
                    group_tests.append(batch[idx].test_name)
                    assigned.add(idx)
            if group_tests:
                clusters.append(group_tests)
    
    # Handle unassigned tests
    for i, features in enumerate(batch):
        if i not in assigned:
            clusters.append([features.test_name])
    
    return clusters


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _fallback_singleton_clustering(
    features_list: List[TestFeatures],
    method_name: str
) -> ClusteringResult:
    """Fallback: put each test in its own cluster."""
    clusters = []
    test_to_cluster = {}
    
    for i, features in enumerate(features_list):
        cluster = PredictedCluster(
            cluster_id=i,
            tests=[features.test_name],
            method="fallback"
        )
        clusters.append(cluster)
        test_to_cluster[features.test_name] = i
    
    return ClusteringResult(
        method_name=method_name,
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={"fallback": True}
    )


def run_all_baselines(
    features_list: List[TestFeatures],
    stack_traces: Dict[str, str],
    num_ground_truth_clusters: int,
    llm_client: Optional[LLMClient] = None,
    random_seed: int = 42
) -> Dict[str, ClusteringResult]:
    """
    Run all baseline methods and return results.
    
    Args:
        features_list: List of TestFeatures
        stack_traces: Dict mapping test name to stack trace
        num_ground_truth_clusters: Number of clusters in ground truth (for B1)
        llm_client: Optional LLM client for B5
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict mapping method name to ClusteringResult
    """
    test_names = [f.test_name for f in features_list]
    results = {}
    
    # B1: Random Clustering
    logger.info("Running B1: Random Clustering...")
    results["B1-Random"] = random_clustering(
        test_names, 
        num_ground_truth_clusters, 
        seed=random_seed
    )
    
    # B2: Test Class-based Clustering
    logger.info("Running B2: Test Class-based Clustering...")
    results["B2-TestClass"] = test_class_clustering(test_names)
    
    # B3: Exception Type-only Clustering
    logger.info("Running B3: Exception Type-only Clustering...")
    results["B3-ExceptionType"] = exception_type_clustering(features_list)
    
    # B4: Embedding-based Clustering
    logger.info("Running B4: Embedding-based Clustering...")
    results["B4-Embedding"] = embedding_clustering(features_list, stack_traces)
    
    # B5: Pure LLM Clustering (if LLM client provided)
    if llm_client:
        logger.info("Running B5: Pure LLM Clustering...")
        results["B5-PureLLM"] = pure_llm_clustering(
            features_list, 
            stack_traces, 
            llm_client
        )
    else:
        logger.info("Skipping B5: No LLM client provided")
    
    # Our Method: Signature-based Clustering
    logger.info("Running Our Method: Signature-based Clustering...")
    results["Hybrid-Signature"] = signature_based_clustering(features_list)
    
    return results


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create some test features
    test_features = [
        TestFeatures(
            test_name="org.example.TestA#test1",
            project="test-project",
            test_class="org.example.TestA",
            test_method="test1",
            exception_type="java.net.UnknownHostException",
            exception_simple_name="UnknownHostException",
            exception_message="host not found",
            exception_category="Networking",
            entry_point="org.example.NetworkClient",
            signature="UnknownHostException:Networking:org.example.NetworkClient"
        ),
        TestFeatures(
            test_name="org.example.TestA#test2",
            project="test-project",
            test_class="org.example.TestA",
            test_method="test2",
            exception_type="java.net.UnknownHostException",
            exception_simple_name="UnknownHostException",
            exception_message="dns failure",
            exception_category="Networking",
            entry_point="org.example.NetworkClient",
            signature="UnknownHostException:Networking:org.example.NetworkClient"
        ),
        TestFeatures(
            test_name="org.example.TestB#test1",
            project="test-project",
            test_class="org.example.TestB",
            test_method="test1",
            exception_type="java.io.FileNotFoundException",
            exception_simple_name="FileNotFoundException",
            exception_message="file not found",
            exception_category="Filesystem",
            entry_point="org.example.FileHandler",
            signature="FileNotFoundException:Filesystem:org.example.FileHandler"
        ),
    ]
    
    test_names = [f.test_name for f in test_features]
    stack_traces = {f.test_name: f"Exception: {f.exception_message}" for f in test_features}
    
    print("Testing Clustering Methods:")
    print("=" * 60)
    
    # Test B1
    result = random_clustering(test_names, 2, seed=42)
    print(f"\nB1 Random: {result.num_clusters} clusters")
    for c in result.clusters:
        print(f"  Cluster {c.cluster_id}: {c.tests}")
    
    # Test B2
    result = test_class_clustering(test_names)
    print(f"\nB2 Test Class: {result.num_clusters} clusters")
    for c in result.clusters:
        print(f"  Cluster {c.cluster_id}: {c.tests}")
    
    # Test B3
    result = exception_type_clustering(test_features)
    print(f"\nB3 Exception Type: {result.num_clusters} clusters")
    for c in result.clusters:
        print(f"  Cluster {c.cluster_id} ({c.signature}): {c.tests}")
    
    # Test Our Method
    result = signature_based_clustering(test_features)
    print(f"\nOur Method (Signature): {result.num_clusters} clusters")
    for c in result.clusters:
        print(f"  Cluster {c.cluster_id} ({c.signature}): {c.tests}")
