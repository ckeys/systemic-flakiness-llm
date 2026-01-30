"""
RQ2 Feature Extractor: Three-Tier Exception Categorization

This module implements the feature extraction pipeline for RQ2:
1. Tier 1: Exact match on known Exception types
2. Tier 2: Keyword matching on Exception type + message
3. Tier 3: LLM batch classification for unknown Exceptions

The goal is to extract features from flaky tests that can be used for clustering
without requiring historical failure statistics.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from llm_client import LLMClient, create_llm_client

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TestFeatures:
    """Features extracted from a flaky test for clustering."""
    test_name: str
    project: str
    
    # Test structure
    test_class: str = ""
    test_method: str = ""
    
    # Exception features
    exception_type: str = ""           # e.g., "java.net.UnknownHostException"
    exception_simple_name: str = ""    # e.g., "UnknownHostException"
    exception_message: str = ""        # e.g., "Temporary failure in name resolution"
    exception_category: str = ""       # e.g., "Networking", "Filesystem", "Timeout"
    
    # Stack trace features
    key_classes: List[str] = field(default_factory=list)  # Non-standard library classes
    entry_point: str = ""              # First project-specific class in stack trace
    stack_depth: int = 0               # Depth of stack trace
    
    # Clustering signature
    signature: str = ""                # Composite key for pre-clustering
    
    # Raw data references
    has_source_code: bool = False
    has_stack_trace: bool = False
    
    # Tier info (for analysis)
    categorization_tier: int = 0       # 1, 2, or 3 - which tier was used


# ============================================================================
# THREE-TIER EXCEPTION CATEGORIZER
# ============================================================================

class ThreeTierExceptionCategorizer:
    """
    Three-tier Exception categorization system:
    - Tier 1: Exact match on known Exception types (O(1))
    - Tier 2: Keyword matching on type + message (O(n))  
    - Tier 3: LLM batch classification for unknowns
    """
    
    # Tier 1: Known Exception type mappings
    TIER1_RULES: Dict[str, List[str]] = {
        "Networking": [
            # Java standard
            "UnknownHostException", "SocketException", "ConnectException",
            "SocketTimeoutException", "NoRouteToHostException", "BindException",
            "PortUnreachableException", "ProtocolException",
            # SSL/TLS
            "SSLException", "SSLHandshakeException", "SSLPeerUnverifiedException",
            "CertificateException", "CertPathValidatorException",
            # HTTP clients
            "HttpHostConnectException", "HttpRetryException",
            "ClientProtocolException", "HttpResponseException",
            # Connection pools
            "ConnectionPoolTimeoutException", "ConnectTimeoutException",
        ],
        "Filesystem": [
            "FileNotFoundException", "NoSuchFileException", "AccessDeniedException",
            "FileAlreadyExistsException", "DirectoryNotEmptyException",
            "FileSystemException", "NotDirectoryException", "FileSystemLoopException",
            "AtomicMoveNotSupportedException", "ReadOnlyFileSystemException",
        ],
        "Timeout": [
            "TimeoutException", "InterruptedIOException",
            "SocketTimeoutException",  # Also in Networking, but timeout is primary
        ],
        "Concurrency": [
            "InterruptedException", "ConcurrentModificationException",
            "RejectedExecutionException", "BrokenBarrierException",
            "ExecutionException", "CancellationException",
            "IllegalMonitorStateException", "ThreadDeath",
        ],
        "Configuration": [
            "NoClassDefFoundError", "ClassNotFoundException", "NoSuchMethodError",
            "NoSuchFieldError", "IncompatibleClassChangeError", "LinkageError",
            "UnsupportedClassVersionError", "VerifyError", "ClassFormatError",
            "AbstractMethodError", "IllegalAccessError", "InstantiationError",
        ],
        "Resource": [
            "OutOfMemoryError", "StackOverflowError",
            "IllegalStateException",  # Often resource-related
        ],
        "Assertion": [
            "AssertionError", "AssertionFailedError", "ComparisonFailure",
        ],
    }
    
    # Tier 2: Keyword patterns for message analysis
    TIER2_KEYWORDS: Dict[str, List[str]] = {
        "Networking": [
            "connection", "connect", "network", "socket", "dns", "host",
            "unreachable", "refused", "reset", "ssl", "tls", "http", "https",
            "proxy", "firewall", "port", "address", "resolve", "lookup",
            "handshake", "certificate", "peer", "remote",
        ],
        "Filesystem": [
            "file", "directory", "path", "permission", "disk", "storage",
            "read", "write", "delete", "create", "exists", "locked", "busy",
        ],
        "Timeout": [
            "timeout", "timed out", "deadline", "expired", "waiting",
        ],
        "Concurrency": [
            "thread", "lock", "concurrent", "race", "deadlock", "synchron",
            "atomic", "volatile", "mutex", "semaphore", "barrier",
        ],
        "Configuration": [
            "class not found", "method not found", "no such method",
            "version", "compatibility", "classpath", "missing",
        ],
        "Resource": [
            "memory", "heap", "stack", "pool", "exhausted", "limit",
            "capacity", "overflow", "leak",
        ],
    }
    
    # Standard library packages to filter out from key_classes
    STANDARD_PACKAGES = [
        "java.", "javax.", "sun.", "com.sun.", "jdk.",
        "org.junit", "junit.", "org.testng",
        "org.apache.maven", "org.gradle",
        "org.mockito", "org.easymock", "org.powermock",
    ]
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the categorizer.
        
        Args:
            llm_client: Optional LLM client for Tier 3 classification.
                       If not provided, Tier 3 will return "Other".
        """
        self.llm_client = llm_client
        self.llm_cache: Dict[str, str] = {}  # Cache LLM results
        
        # Build reverse lookup for Tier 1
        self._tier1_lookup: Dict[str, str] = {}
        for category, exceptions in self.TIER1_RULES.items():
            for exc in exceptions:
                self._tier1_lookup[exc.lower()] = category
    
    def categorize(
        self, 
        exception_type: str, 
        exception_message: str,
        stack_trace: str = ""
    ) -> Tuple[str, int]:
        """
        Categorize an Exception using the three-tier system.
        
        Args:
            exception_type: Full Exception type (e.g., "java.net.UnknownHostException")
            exception_message: Exception message
            stack_trace: Optional stack trace for additional context
            
        Returns:
            Tuple of (category, tier_used)
        """
        # Extract simple name
        simple_name = exception_type.split(".")[-1] if "." in exception_type else exception_type
        
        # Tier 1: Exact match
        category = self._tier1_match(simple_name)
        if category:
            return category, 1
        
        # Tier 2: Keyword match
        category = self._tier2_match(exception_type, exception_message)
        if category:
            return category, 2
        
        # Tier 3: LLM classification
        category = self._tier3_llm_classify(exception_type, exception_message, stack_trace)
        return category, 3
    
    def _tier1_match(self, simple_name: str) -> Optional[str]:
        """Tier 1: Exact match on Exception simple name."""
        return self._tier1_lookup.get(simple_name.lower())
    
    def _tier2_match(self, exception_type: str, message: str) -> Optional[str]:
        """Tier 2: Keyword matching on type + message."""
        text = f"{exception_type} {message}".lower()
        
        # Check each category's keywords
        for category, keywords in self.TIER2_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                # Special case: "timeout" in networking context
                if category == "Timeout" and any(
                    nw in text for nw in ["socket", "connect", "http"]
                ):
                    return "Networking"
                return category
        
        return None
    
    def _tier3_llm_classify(
        self, 
        exception_type: str, 
        message: str, 
        stack_trace: str
    ) -> str:
        """Tier 3: LLM classification for unknown Exceptions."""
        if not self.llm_client:
            return "Other"
        
        # Check cache
        cache_key = f"{exception_type}:{message[:100]}"
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
        
        # Set component for cost tracking
        self.llm_client.set_component("tier3_classification")
        
        # Build prompt
        prompt = f"""Classify this Java Exception into ONE of these categories:
- Networking: Network, connection, DNS, SSL/TLS, HTTP issues
- Filesystem: File, directory, path, permission issues  
- Timeout: Timeout, deadline issues (not network-related)
- Concurrency: Thread, lock, race condition, synchronization issues
- Configuration: Class loading, version, classpath issues
- Resource: Memory, pool exhaustion issues
- Assertion: Test assertion failures
- Other: None of the above

Exception Type: {exception_type}
Exception Message: {message}

Reply with ONLY the category name, nothing else."""

        try:
            response = self.llm_client.generate(prompt)
            category = response.strip()
            
            # Validate response
            valid_categories = [
                "Networking", "Filesystem", "Timeout", "Concurrency",
                "Configuration", "Resource", "Assertion", "Other"
            ]
            if category not in valid_categories:
                # Try to extract category from response
                for valid in valid_categories:
                    if valid.lower() in category.lower():
                        category = valid
                        break
                else:
                    category = "Other"
            
            # Cache result
            self.llm_cache[cache_key] = category
            return category
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return "Other"
    
    def batch_categorize(
        self, 
        exceptions: List[Dict[str, str]]
    ) -> Dict[str, Tuple[str, int]]:
        """
        Batch categorize multiple Exceptions efficiently.
        
        Args:
            exceptions: List of dicts with 'type', 'message', 'stack_trace' keys
            
        Returns:
            Dict mapping exception key to (category, tier)
        """
        results = {}
        unknown_for_llm = []
        
        for exc in exceptions:
            exc_type = exc.get("type", "")
            exc_msg = exc.get("message", "")
            exc_trace = exc.get("stack_trace", "")
            key = f"{exc_type}:{exc_msg[:50]}"
            
            # Try Tier 1
            simple_name = exc_type.split(".")[-1] if "." in exc_type else exc_type
            category = self._tier1_match(simple_name)
            if category:
                results[key] = (category, 1)
                continue
            
            # Try Tier 2
            category = self._tier2_match(exc_type, exc_msg)
            if category:
                results[key] = (category, 2)
                continue
            
            # Collect for batch LLM
            unknown_for_llm.append((key, exc_type, exc_msg, exc_trace))
        
        # Batch LLM classification
        if unknown_for_llm and self.llm_client:
            llm_results = self._batch_llm_classify(unknown_for_llm)
            for key, category in llm_results.items():
                results[key] = (category, 3)
        else:
            # No LLM, mark as Other
            for key, _, _, _ in unknown_for_llm:
                results[key] = ("Other", 3)
        
        return results
    
    def _batch_llm_classify(
        self, 
        unknowns: List[Tuple[str, str, str, str]]
    ) -> Dict[str, str]:
        """Batch classify unknown Exceptions with a single LLM call."""
        if not unknowns:
            return {}
        
        # Set component for cost tracking
        self.llm_client.set_component("tier3_batch_classification")
        
        # Build batch prompt
        items = []
        for i, (key, exc_type, exc_msg, _) in enumerate(unknowns):
            items.append(f"[{i+1}] Type: {exc_type}\n    Message: {exc_msg[:100]}")
        
        prompt = f"""Classify each of these {len(unknowns)} Java Exceptions into ONE category each.

Categories:
- Networking: Network, connection, DNS, SSL/TLS, HTTP issues
- Filesystem: File, directory, path, permission issues
- Timeout: Timeout, deadline issues (not network-related)
- Concurrency: Thread, lock, race condition issues
- Configuration: Class loading, version, classpath issues
- Resource: Memory, pool exhaustion issues
- Assertion: Test assertion failures
- Other: None of the above

Exceptions:
{chr(10).join(items)}

Reply with ONLY the classifications in this format (one per line):
[1] CategoryName
[2] CategoryName
..."""

        try:
            response = self.llm_client.generate(prompt)
            
            # Parse response
            results = {}
            for line in response.strip().split("\n"):
                match = re.match(r"\[(\d+)\]\s*(\w+)", line)
                if match:
                    idx = int(match.group(1)) - 1
                    category = match.group(2)
                    if 0 <= idx < len(unknowns):
                        key = unknowns[idx][0]
                        # Validate category
                        valid_categories = [
                            "Networking", "Filesystem", "Timeout", "Concurrency",
                            "Configuration", "Resource", "Assertion", "Other"
                        ]
                        if category in valid_categories:
                            results[key] = category
                            self.llm_cache[key] = category
                        else:
                            results[key] = "Other"
            
            # Fill in any missing
            for key, _, _, _ in unknowns:
                if key not in results:
                    results[key] = "Other"
            
            return results
            
        except Exception as e:
            logger.warning(f"Batch LLM classification failed: {e}")
            return {key: "Other" for key, _, _, _ in unknowns}
    
    def is_standard_library(self, class_name: str) -> bool:
        """Check if a class is from standard library."""
        return any(class_name.startswith(pkg) for pkg in self.STANDARD_PACKAGES)


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Extract features from flaky tests for clustering.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the feature extractor.
        
        Args:
            llm_client: Optional LLM client for Tier 3 classification
        """
        self.categorizer = ThreeTierExceptionCategorizer(llm_client)
    
    def extract_features(self, test_case: Any, project: str) -> TestFeatures:
        """
        Extract features from a single test case.
        
        Args:
            test_case: TestCase object from data_loader
            project: Project name
            
        Returns:
            TestFeatures object
        """
        features = TestFeatures(
            test_name=test_case.name,
            project=project,
            test_class=test_case.class_name,
            test_method=test_case.method_name,
            has_source_code=bool(test_case.source_code),
            has_stack_trace=bool(test_case.stack_traces),
        )
        
        # Extract exception info from stack trace
        if test_case.stack_traces:
            stack_trace = test_case.stack_traces[0]
            
            # Parse exception type and message
            exc_type, exc_msg = self._parse_exception_from_trace(stack_trace)
            features.exception_type = exc_type
            features.exception_simple_name = exc_type.split(".")[-1] if "." in exc_type else exc_type
            features.exception_message = exc_msg
            
            # Extract key classes and entry point
            key_classes = self._extract_key_classes(stack_trace)
            features.key_classes = key_classes
            features.entry_point = key_classes[0] if key_classes else ""
            features.stack_depth = stack_trace.count("\n\tat ")
            
            # Categorize exception
            category, tier = self.categorizer.categorize(exc_type, exc_msg, stack_trace)
            features.exception_category = category
            features.categorization_tier = tier
        
        # If no stack trace, try to infer category from source code
        elif test_case.source_code:
            category = self._infer_category_from_code(test_case.source_code)
            features.exception_category = category
            features.categorization_tier = 4  # Tier 4: Code-based inference
        
        # Build signature for pre-clustering
        features.signature = self._build_signature(features)
        
        return features
    
    def _infer_category_from_code(self, source_code: str) -> str:
        """
        Infer exception category from source code when no stack trace is available.
        
        Args:
            source_code: Test method source code
            
        Returns:
            Inferred category string
        """
        code_lower = source_code.lower()
        
        # Check for networking-related code
        networking_patterns = [
            "socket", "http", "url", "connection", "client", "server",
            "request", "response", "network", "dns", "host", "port"
        ]
        if any(p in code_lower for p in networking_patterns):
            return "Networking"
        
        # Check for filesystem-related code
        filesystem_patterns = [
            "file", "path", "directory", "inputstream", "outputstream",
            "reader", "writer", "filesystem", "tempfile"
        ]
        if any(p in code_lower for p in filesystem_patterns):
            return "Filesystem"
        
        # Check for concurrency-related code
        concurrency_patterns = [
            "thread", "executor", "concurrent", "async", "await",
            "lock", "synchronized", "latch", "barrier", "semaphore"
        ]
        if any(p in code_lower for p in concurrency_patterns):
            return "Concurrency"
        
        # Check for timeout-related code
        timeout_patterns = [
            "timeout", "wait", "sleep", "delay", "deadline"
        ]
        if any(p in code_lower for p in timeout_patterns):
            return "Timeout"
        
        # Check for assertion-related code
        assertion_patterns = [
            "assert", "expect", "verify", "should"
        ]
        if any(p in code_lower for p in assertion_patterns):
            return "Assertion"
        
        return "Unknown"
    
    def extract_features_batch(
        self, 
        test_cases: List[Any], 
        project: str
    ) -> List[TestFeatures]:
        """
        Extract features from multiple test cases efficiently.
        
        Args:
            test_cases: List of TestCase objects
            project: Project name
            
        Returns:
            List of TestFeatures objects
        """
        # First pass: extract basic features and collect exceptions
        features_list = []
        exceptions_to_classify = []
        
        for test_case in test_cases:
            features = TestFeatures(
                test_name=test_case.name,
                project=project,
                test_class=test_case.class_name,
                test_method=test_case.method_name,
                has_source_code=bool(test_case.source_code),
                has_stack_trace=bool(test_case.stack_traces),
            )
            
            if test_case.stack_traces:
                stack_trace = test_case.stack_traces[0]
                exc_type, exc_msg = self._parse_exception_from_trace(stack_trace)
                features.exception_type = exc_type
                features.exception_simple_name = exc_type.split(".")[-1] if "." in exc_type else exc_type
                features.exception_message = exc_msg
                
                key_classes = self._extract_key_classes(stack_trace)
                features.key_classes = key_classes
                features.entry_point = key_classes[0] if key_classes else ""
                features.stack_depth = stack_trace.count("\n\tat ")
                
                exceptions_to_classify.append({
                    "type": exc_type,
                    "message": exc_msg,
                    "stack_trace": stack_trace,
                    "index": len(features_list)
                })
            
            features_list.append(features)
        
        # Batch categorize exceptions
        if exceptions_to_classify:
            exc_inputs = [
                {"type": e["type"], "message": e["message"], "stack_trace": e["stack_trace"]}
                for e in exceptions_to_classify
            ]
            category_results = self.categorizer.batch_categorize(exc_inputs)
            
            for exc_info in exceptions_to_classify:
                key = f"{exc_info['type']}:{exc_info['message'][:50]}"
                if key in category_results:
                    category, tier = category_results[key]
                    features_list[exc_info["index"]].exception_category = category
                    features_list[exc_info["index"]].categorization_tier = tier
        
        # Build signatures
        for features in features_list:
            features.signature = self._build_signature(features)
        
        return features_list
    
    def _parse_exception_from_trace(self, stack_trace: str) -> Tuple[str, str]:
        """
        Parse exception type and message from stack trace.
        
        Args:
            stack_trace: Stack trace string
            
        Returns:
            Tuple of (exception_type, exception_message)
        """
        if not stack_trace:
            return "", ""
        
        lines = stack_trace.strip().split("\n")
        if not lines:
            return "", ""
        
        first_line = lines[0].strip()
        
        # Pattern: ExceptionType: message
        # or: ExceptionType
        if ":" in first_line:
            parts = first_line.split(":", 1)
            exc_type = parts[0].strip()
            exc_msg = parts[1].strip() if len(parts) > 1 else ""
        else:
            exc_type = first_line
            exc_msg = ""
        
        # Handle "Caused by:" prefix
        if exc_type.lower().startswith("caused by"):
            exc_type = exc_type[9:].strip()
            if exc_type.startswith(":"):
                exc_type = exc_type[1:].strip()
        
        return exc_type, exc_msg
    
    def _extract_key_classes(self, stack_trace: str) -> List[str]:
        """
        Extract key (non-standard-library) classes from stack trace.
        
        Args:
            stack_trace: Stack trace string
            
        Returns:
            List of key class names
        """
        key_classes = []
        
        # Pattern: at package.Class.method(File.java:line)
        pattern = r"at\s+([\w.]+)\.[^(]+\("
        
        for match in re.finditer(pattern, stack_trace):
            full_class = match.group(1)
            if not self.categorizer.is_standard_library(full_class):
                if full_class not in key_classes:
                    key_classes.append(full_class)
                    if len(key_classes) >= 5:  # Limit to top 5
                        break
        
        return key_classes
    
    def _build_signature(self, features: TestFeatures) -> str:
        """
        Build a signature for pre-clustering.
        
        Strategy: Project-Aware Coarse-Grained Clustering
        
        Key insight from Ground Truth analysis:
        - Alluxio-alluxio: 113 tests → 1 cluster (all tests share same root cause)
        - spring-projects-spring-boot: 147 tests → 6 clusters
        
        This suggests:
        - Project is the PRIMARY grouping factor
        - Exception category is SECONDARY (to split projects with multiple root causes)
        - Test class is TOO fine-grained (causes over-segmentation)
        
        Args:
            features: TestFeatures object
            
        Returns:
            Signature string
        """
        # Project is the primary grouping factor
        project = features.project or "Unknown"
        
        # Exception category as secondary factor
        category = features.exception_category or "Unknown"
        
        # Only split by strong categories that clearly indicate different root causes
        # Networking, Filesystem, Timeout, Concurrency are distinct failure modes
        strong_categories = ["Networking", "Filesystem", "Timeout", "Concurrency"]
        
        if category in strong_categories:
            return f"{project}:{category}"
        else:
            # Group all other categories together within the project
            return f"{project}:Other"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_categorization_statistics(features_list: List[TestFeatures]) -> Dict[str, Any]:
    """
    Get statistics about the categorization process.
    
    Args:
        features_list: List of TestFeatures objects
        
    Returns:
        Dictionary with statistics
    """
    total = len(features_list)
    if total == 0:
        return {}
    
    tier_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    category_counts: Dict[str, int] = {}
    
    for f in features_list:
        tier_counts[f.categorization_tier] = tier_counts.get(f.categorization_tier, 0) + 1
        cat = f.exception_category or "Unknown"
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return {
        "total_tests": total,
        "tier1_count": tier_counts[1],
        "tier1_pct": tier_counts[1] / total * 100,
        "tier2_count": tier_counts[2],
        "tier2_pct": tier_counts[2] / total * 100,
        "tier3_count": tier_counts[3],
        "tier3_pct": tier_counts[3] / total * 100,
        "tier4_count": tier_counts[4],
        "tier4_pct": tier_counts[4] / total * 100,
        "no_info_count": tier_counts[0],
        "no_info_pct": tier_counts[0] / total * 100,
        "category_distribution": category_counts,
    }


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test the categorizer
    logging.basicConfig(level=logging.INFO)
    
    categorizer = ThreeTierExceptionCategorizer()
    
    test_cases = [
        ("java.net.UnknownHostException", "Temporary failure in name resolution"),
        ("java.io.FileNotFoundException", "/tmp/test.txt (No such file or directory)"),
        ("java.lang.NoClassDefFoundError", "Could not initialize class X"),
        ("java.io.IOException", "Connection refused"),
        ("org.custom.MyException", "Something went wrong"),
    ]
    
    print("Testing Three-Tier Exception Categorizer:")
    print("-" * 60)
    
    for exc_type, exc_msg in test_cases:
        category, tier = categorizer.categorize(exc_type, exc_msg)
        print(f"Type: {exc_type}")
        print(f"Message: {exc_msg}")
        print(f"Category: {category} (Tier {tier})")
        print("-" * 60)
