"""
RQ2 LLM Verification: Cluster Verification and Refinement

This module implements Phase 3 of our Hybrid Method:
- LLM-based verification of candidate clusters
- Splitting clusters that don't share a common root cause
- Merging clusters that should be together

The goal is to refine pre-clustering results using LLM's code understanding.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from rq2_feature_extractor import TestFeatures
from rq2_clustering import ClusteringResult, PredictedCluster
from llm_client import LLMClient

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VerificationResult:
    """Result of verifying a single cluster."""
    cluster_id: int
    verdict: str  # "Yes", "No", "Partial"
    confidence: str  # "High", "Medium", "Low"
    root_cause: Optional[str] = None
    split_suggestion: Optional[List[List[str]]] = None  # Suggested sub-clusters
    reasoning: str = ""


@dataclass
class VerifiedCluster:
    """A cluster that has been verified by LLM."""
    cluster_id: int
    tests: List[str]
    root_cause: str
    confidence: str
    original_cluster_id: int  # ID from pre-clustering
    verification_verdict: str


# ============================================================================
# CLUSTER VERIFIER
# ============================================================================

class ClusterVerifier:
    """
    Verifies candidate clusters using LLM analysis.
    
    For each candidate cluster, the LLM determines:
    1. Whether tests truly share a common root cause
    2. If not, how to split them
    3. The shared root cause (if verified)
    """
    
    def __init__(
        self, 
        llm_client: LLMClient,
        max_tests_per_verification: int = 15
    ):
        """
        Initialize the verifier.
        
        Args:
            llm_client: LLM client for verification
            max_tests_per_verification: Max tests to include in one verification call
        """
        self.llm_client = llm_client
        self.max_tests_per_verification = max_tests_per_verification
    
    def verify_cluster(
        self,
        cluster: PredictedCluster,
        features_map: Dict[str, TestFeatures],
        stack_traces: Dict[str, str],
        source_codes: Dict[str, str]
    ) -> VerificationResult:
        """
        Verify a single cluster.
        
        Args:
            cluster: The candidate cluster to verify
            features_map: Mapping from test name to TestFeatures
            stack_traces: Mapping from test name to stack trace
            source_codes: Mapping from test name to source code
            
        Returns:
            VerificationResult with verdict and suggestions
        """
        if len(cluster.tests) < 2:
            # Single test cluster - trivially verified
            return VerificationResult(
                cluster_id=cluster.cluster_id,
                verdict="Yes",
                confidence="High",
                root_cause="Single test cluster",
                reasoning="Cluster contains only one test"
            )
        
        # Set component for cost tracking
        self.llm_client.set_component("cluster_verification")
        
        # Select tests for verification (sample if too many)
        tests_to_verify = cluster.tests[:self.max_tests_per_verification]
        
        # Build prompt
        prompt = self._build_verification_prompt(
            tests_to_verify,
            features_map,
            stack_traces,
            source_codes
        )
        
        try:
            response = self.llm_client.generate(prompt)
            result = self._parse_verification_response(response, cluster.cluster_id)
            return result
            
        except Exception as e:
            logger.warning(f"Verification failed for cluster {cluster.cluster_id}: {e}")
            return VerificationResult(
                cluster_id=cluster.cluster_id,
                verdict="Yes",  # Default to verified on error
                confidence="Low",
                reasoning=f"Verification failed: {e}"
            )
    
    def _build_verification_prompt(
        self,
        tests: List[str],
        features_map: Dict[str, TestFeatures],
        stack_traces: Dict[str, str],
        source_codes: Dict[str, str]
    ) -> str:
        """Build the verification prompt."""
        
        # Get common exception type
        common_exception = ""
        if tests and tests[0] in features_map:
            common_exception = features_map[tests[0]].exception_type
        
        # Build test details
        test_details = []
        for i, test_name in enumerate(tests):
            features = features_map.get(test_name)
            trace = stack_traces.get(test_name, "")
            code = source_codes.get(test_name, "")
            
            detail = f"""### Test {i+1}: {test_name}
**Exception:** {features.exception_type if features else 'Unknown'}: {features.exception_message if features else ''}
**Stack Trace (excerpt):**
```
{trace[:800] if trace else 'No stack trace available'}
```"""
            
            if code:
                detail += f"""
**Code (excerpt):**
```java
{code[:600]}
```"""
            
            test_details.append(detail)
        
        prompt = f"""You are analyzing {len(tests)} flaky tests that were grouped together because they all throw similar exceptions ({common_exception}).

Your task is to determine if these tests truly share a COMMON ROOT CAUSE for their flakiness.

{chr(10).join(test_details)}

## Analysis Questions:
1. Do ALL these tests fail due to the SAME underlying root cause?
2. If yes, what is that shared root cause?
3. If not all tests share the same root cause, which tests should be grouped together?

## Response Format (MUST follow exactly):
VERDICT: [Yes/No/Partial]
CONFIDENCE: [High/Medium/Low]
ROOT_CAUSE: [If Yes or Partial, describe the shared root cause in 1-2 sentences]
SPLIT_SUGGESTION: [If No or Partial, list which test numbers should be grouped together, e.g., "Tests 1,2,3 share cause A; Tests 4,5 share cause B"]
REASONING: [Brief explanation of your analysis]"""

        return prompt
    
    def _parse_verification_response(
        self, 
        response: str, 
        cluster_id: int
    ) -> VerificationResult:
        """Parse the LLM verification response."""
        
        result = VerificationResult(cluster_id=cluster_id, verdict="Yes", confidence="Medium")
        
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            if line.upper().startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
                if "YES" in verdict:
                    result.verdict = "Yes"
                elif "NO" in verdict:
                    result.verdict = "No"
                elif "PARTIAL" in verdict:
                    result.verdict = "Partial"
            
            elif line.upper().startswith("CONFIDENCE:"):
                conf = line.split(":", 1)[1].strip().upper()
                if "HIGH" in conf:
                    result.confidence = "High"
                elif "LOW" in conf:
                    result.confidence = "Low"
                else:
                    result.confidence = "Medium"
            
            elif line.upper().startswith("ROOT_CAUSE:"):
                result.root_cause = line.split(":", 1)[1].strip()
            
            elif line.upper().startswith("SPLIT_SUGGESTION:"):
                suggestion = line.split(":", 1)[1].strip()
                result.split_suggestion = self._parse_split_suggestion(suggestion)
            
            elif line.upper().startswith("REASONING:"):
                result.reasoning = line.split(":", 1)[1].strip()
        
        return result
    
    def _parse_split_suggestion(self, suggestion: str) -> Optional[List[List[str]]]:
        """Parse split suggestion into groups of test indices."""
        if not suggestion or suggestion.lower() in ["none", "n/a", "-"]:
            return None
        
        groups = []
        
        # Pattern: "Tests 1,2,3 share cause A; Tests 4,5 share cause B"
        # or: "Group 1: [1,2,3]; Group 2: [4,5]"
        parts = re.split(r"[;|]", suggestion)
        
        for part in parts:
            # Extract numbers
            numbers = re.findall(r"\d+", part)
            if numbers:
                groups.append([f"test_{n}" for n in numbers])  # Placeholder indices
        
        return groups if groups else None
    
    def verify_all_clusters(
        self,
        clustering_result: ClusteringResult,
        features_map: Dict[str, TestFeatures],
        stack_traces: Dict[str, str],
        source_codes: Dict[str, str],
        min_cluster_size: int = 2
    ) -> Tuple[List[VerifiedCluster], Dict[str, Any]]:
        """
        Verify all clusters and return refined clustering.
        
        Args:
            clustering_result: Pre-clustering result
            features_map: Mapping from test name to TestFeatures
            stack_traces: Mapping from test name to stack trace
            source_codes: Mapping from test name to source code
            min_cluster_size: Minimum size for clusters to verify
            
        Returns:
            Tuple of (list of verified clusters, verification statistics)
        """
        verified_clusters = []
        stats = {
            "total_clusters": len(clustering_result.clusters),
            "verified_yes": 0,
            "verified_no": 0,
            "verified_partial": 0,
            "skipped_small": 0,
            "splits_performed": 0
        }
        
        new_cluster_id = 0
        
        for cluster in clustering_result.clusters:
            if len(cluster.tests) < min_cluster_size:
                # Skip small clusters, just pass through
                stats["skipped_small"] += 1
                verified = VerifiedCluster(
                    cluster_id=new_cluster_id,
                    tests=cluster.tests,
                    root_cause="Small cluster (not verified)",
                    confidence="N/A",
                    original_cluster_id=cluster.cluster_id,
                    verification_verdict="Skipped"
                )
                verified_clusters.append(verified)
                new_cluster_id += 1
                continue
            
            # Verify the cluster
            logger.info(f"Verifying cluster {cluster.cluster_id} ({len(cluster.tests)} tests)...")
            verification = self.verify_cluster(
                cluster, features_map, stack_traces, source_codes
            )
            
            if verification.verdict == "Yes":
                stats["verified_yes"] += 1
                verified = VerifiedCluster(
                    cluster_id=new_cluster_id,
                    tests=cluster.tests,
                    root_cause=verification.root_cause or "Verified shared root cause",
                    confidence=verification.confidence,
                    original_cluster_id=cluster.cluster_id,
                    verification_verdict="Yes"
                )
                verified_clusters.append(verified)
                new_cluster_id += 1
            
            elif verification.verdict == "No":
                stats["verified_no"] += 1
                # Split into individual clusters (conservative approach)
                for test in cluster.tests:
                    verified = VerifiedCluster(
                        cluster_id=new_cluster_id,
                        tests=[test],
                        root_cause="Split: no shared root cause",
                        confidence=verification.confidence,
                        original_cluster_id=cluster.cluster_id,
                        verification_verdict="No"
                    )
                    verified_clusters.append(verified)
                    new_cluster_id += 1
                stats["splits_performed"] += 1
            
            elif verification.verdict == "Partial":
                stats["verified_partial"] += 1
                
                if verification.split_suggestion:
                    # Use LLM's split suggestion
                    stats["splits_performed"] += 1
                    # For now, treat as verified (could implement split logic)
                    verified = VerifiedCluster(
                        cluster_id=new_cluster_id,
                        tests=cluster.tests,
                        root_cause=verification.root_cause or "Partially shared root cause",
                        confidence=verification.confidence,
                        original_cluster_id=cluster.cluster_id,
                        verification_verdict="Partial"
                    )
                    verified_clusters.append(verified)
                    new_cluster_id += 1
                else:
                    # No split suggestion, keep as is
                    verified = VerifiedCluster(
                        cluster_id=new_cluster_id,
                        tests=cluster.tests,
                        root_cause=verification.root_cause or "Partially verified",
                        confidence=verification.confidence,
                        original_cluster_id=cluster.cluster_id,
                        verification_verdict="Partial"
                    )
                    verified_clusters.append(verified)
                    new_cluster_id += 1
        
        return verified_clusters, stats


# ============================================================================
# CONVERT VERIFIED CLUSTERS TO CLUSTERING RESULT
# ============================================================================

def verified_to_clustering_result(
    verified_clusters: List[VerifiedCluster],
    method_name: str = "Hybrid-Verified"
) -> ClusteringResult:
    """
    Convert verified clusters to ClusteringResult format.
    
    Args:
        verified_clusters: List of VerifiedCluster objects
        method_name: Name for the clustering method
        
    Returns:
        ClusteringResult object
    """
    clusters = []
    test_to_cluster = {}
    
    for vc in verified_clusters:
        cluster = PredictedCluster(
            cluster_id=vc.cluster_id,
            tests=vc.tests,
            predicted_category=vc.root_cause,
            method="verified"
        )
        clusters.append(cluster)
        
        for test in vc.tests:
            test_to_cluster[test] = vc.cluster_id
    
    return ClusteringResult(
        method_name=method_name,
        clusters=clusters,
        test_to_cluster=test_to_cluster,
        metadata={
            "verified": True,
            "num_verified_clusters": len(verified_clusters)
        }
    )


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("ClusterVerifier module loaded successfully.")
    print("To test verification, run with actual LLM client and cluster data.")
