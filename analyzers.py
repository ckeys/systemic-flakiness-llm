"""
Analyzers for RQ1 Experiment

This module implements the two analysis strategies:
1. Individual Analysis: Analyze each test separately, then aggregate
2. Collective Analysis: Analyze all tests together as a cluster
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from data_loader import Cluster, TestCase
from llm_client import (
    LLMClient,
    SYSTEM_PROMPT,
    INDIVIDUAL_ANALYSIS_PROMPT,
    AGGREGATION_PROMPT,
    COLLECTIVE_ANALYSIS_PROMPT,
    format_test_for_collective_prompt,
    format_individual_diagnoses
)
from config import MAX_TESTS_PER_CLUSTER

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    """Result of a root cause diagnosis."""
    cluster_project: str
    cluster_id: int
    diagnosis: str
    method: str  # "individual" or "collective"
    
    # Additional metadata
    num_tests_analyzed: int = 0
    individual_diagnoses: Optional[list[tuple[str, str]]] = None  # For individual method
    

class IndividualAnalyzer:
    """
    Analyzer that processes each test independently, then aggregates results.
    
    This is the baseline approach that mimics how existing LLM-based methods
    (FlakyFix, FlakyDoctor, etc.) work - analyzing tests one at a time.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def analyze_single_test(self, test: TestCase) -> str:
        """
        Analyze a single test and return its diagnosis.
        
        Args:
            test: TestCase to analyze
            
        Returns:
            Diagnosis string
        """
        # Prepare the prompt
        source_code = test.source_code or "[Source code not available]"
        stack_trace = test.stack_traces[0] if test.stack_traces else "[Stack trace not available]"
        
        prompt = INDIVIDUAL_ANALYSIS_PROMPT.format(
            test_name=test.name,
            source_code=source_code,
            stack_trace=stack_trace
        )
        
        diagnosis = self.llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        return diagnosis.strip()
    
    def aggregate_diagnoses(self, diagnoses: list[tuple[str, str]]) -> str:
        """
        Aggregate individual diagnoses into a shared root cause.
        
        Args:
            diagnoses: List of (test_name, diagnosis) tuples
            
        Returns:
            Aggregated diagnosis string
        """
        formatted_diagnoses = format_individual_diagnoses(diagnoses)
        
        prompt = AGGREGATION_PROMPT.format(
            num_tests=len(diagnoses),
            individual_diagnoses=formatted_diagnoses
        )
        
        aggregated = self.llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        return aggregated.strip()
    
    def analyze_cluster(self, cluster: Cluster, max_tests: int = MAX_TESTS_PER_CLUSTER) -> DiagnosisResult:
        """
        Analyze a cluster using the individual analysis strategy.
        
        Args:
            cluster: Cluster to analyze
            max_tests: Maximum number of tests to analyze
            
        Returns:
            DiagnosisResult with the aggregated diagnosis
        """
        logger.info(f"Individual analysis: {cluster.project}/cluster{cluster.cluster_id} ({cluster.size} tests)")
        
        # Select tests to analyze
        tests_to_analyze = cluster.tests[:max_tests]
        
        # Analyze each test individually
        individual_diagnoses = []
        for test in tests_to_analyze:
            try:
                diagnosis = self.analyze_single_test(test)
                individual_diagnoses.append((test.name, diagnosis))
                logger.debug(f"  Analyzed: {test.name}")
            except Exception as e:
                logger.warning(f"  Failed to analyze {test.name}: {e}")
                individual_diagnoses.append((test.name, f"[Analysis failed: {e}]"))
        
        # Aggregate the diagnoses
        if len(individual_diagnoses) == 1:
            # Only one test, use its diagnosis directly
            aggregated_diagnosis = individual_diagnoses[0][1]
        else:
            aggregated_diagnosis = self.aggregate_diagnoses(individual_diagnoses)
        
        return DiagnosisResult(
            cluster_project=cluster.project,
            cluster_id=cluster.cluster_id,
            diagnosis=aggregated_diagnosis,
            method="individual",
            num_tests_analyzed=len(tests_to_analyze),
            individual_diagnoses=individual_diagnoses
        )


class CollectiveAnalyzer:
    """
    Analyzer that processes all tests in a cluster together.
    
    This is the treatment approach that leverages the insight that tests
    failing together likely share a root cause.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def analyze_cluster(self, cluster: Cluster, max_tests: int = MAX_TESTS_PER_CLUSTER) -> DiagnosisResult:
        """
        Analyze a cluster using the collective analysis strategy.
        
        Args:
            cluster: Cluster to analyze
            max_tests: Maximum number of tests to include
            
        Returns:
            DiagnosisResult with the collective diagnosis
        """
        logger.info(f"Collective analysis: {cluster.project}/cluster{cluster.cluster_id} ({cluster.size} tests)")
        
        # Select tests to analyze
        tests_to_analyze = cluster.tests[:max_tests]
        
        # Format all tests for the prompt
        test_details = []
        for i, test in enumerate(tests_to_analyze):
            test_details.append(format_test_for_collective_prompt(test, i))
        
        prompt = COLLECTIVE_ANALYSIS_PROMPT.format(
            num_tests=len(tests_to_analyze),
            test_details="\n\n".join(test_details)
        )
        
        # Generate collective diagnosis
        diagnosis = self.llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        
        return DiagnosisResult(
            cluster_project=cluster.project,
            cluster_id=cluster.cluster_id,
            diagnosis=diagnosis.strip(),
            method="collective",
            num_tests_analyzed=len(tests_to_analyze)
        )


def select_representative_tests(cluster: Cluster, max_tests: int) -> list[TestCase]:
    """
    Select representative tests from a large cluster.
    
    Strategy:
    1. Prioritize tests with both source code and stack traces
    2. Try to get diverse stack traces (different error types)
    
    Args:
        cluster: Cluster to select from
        max_tests: Maximum number of tests to select
        
    Returns:
        List of selected TestCase objects
    """
    if cluster.size <= max_tests:
        return cluster.tests
    
    # Score tests by data availability
    scored_tests = []
    for test in cluster.tests:
        score = 0
        if test.source_code:
            score += 2
        if test.stack_traces:
            score += 1
        scored_tests.append((score, test))
    
    # Sort by score (descending) and select top tests
    scored_tests.sort(key=lambda x: x[0], reverse=True)
    selected = [test for _, test in scored_tests[:max_tests]]
    
    return selected


if __name__ == "__main__":
    # Test the analyzers with a mock client
    from data_loader import load_all_clusters
    
    print("Loading clusters...")
    clusters = load_all_clusters()
    
    if clusters:
        sample_cluster = clusters[0]
        print(f"\nSample cluster: {sample_cluster.project}/cluster{sample_cluster.cluster_id}")
        print(f"  Size: {sample_cluster.size}")
        print(f"  Ground Truth (Q3): {sample_cluster.q3_answer}")
        
        # Show what would be analyzed
        tests_to_analyze = select_representative_tests(sample_cluster, MAX_TESTS_PER_CLUSTER)
        print(f"\n  Tests to analyze ({len(tests_to_analyze)}):")
        for test in tests_to_analyze:
            has_source = "✓" if test.source_code else "✗"
            has_trace = "✓" if test.stack_traces else "✗"
            print(f"    {test.name} [source:{has_source}, trace:{has_trace}]")

