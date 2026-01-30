"""
Data Loader for RQ1 Experiment

This module handles loading cluster data from the Systemic Flakiness dataset,
including:
- Cluster information (test names, run IDs)
- Ground truth (human-annotated Q3 answers)
- Test source code
- Stack traces
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from config import (
    MANUAL_ANALYSIS_DIR,
    SOURCES_DIR,
    PROJECTS_WITH_CLUSTERS,
    MAX_TRACES_PER_TEST
)


@dataclass
class TestCase:
    """Represents a single flaky test case."""
    name: str                           # Full test name (e.g., "org.apache...#testMethod")
    project: str                        # Project name
    source_code: Optional[str] = None   # Test method source code
    stack_traces: list[str] = field(default_factory=list)  # Sample stack traces
    
    @property
    def class_name(self) -> str:
        """Extract class name from test name."""
        if "#" in self.name:
            return self.name.split("#")[0]
        return self.name.rsplit(".", 1)[0] if "." in self.name else self.name
    
    @property
    def method_name(self) -> str:
        """Extract method name from test name."""
        if "#" in self.name:
            return self.name.split("#")[1]
        return self.name.rsplit(".", 1)[1] if "." in self.name else self.name
    
    @property
    def source_file_name(self) -> str:
        """Get the expected source file name in the dataset."""
        # Convert "org.package.Class#method" to "org.package.Class-method.java"
        return f"{self.class_name}-{self.method_name}.java"


@dataclass
class Cluster:
    """Represents a cluster of co-failing flaky tests."""
    project: str                        # Project name
    cluster_id: int                     # Cluster number
    tests: list[TestCase] = field(default_factory=list)  # Tests in this cluster
    run_ids: list[int] = field(default_factory=list)     # Run IDs where failures occurred
    
    # Ground truth from human annotation
    q1_answer: Optional[str] = None     # Extent of shared root causes
    q2_answer: Optional[str] = None     # Causal relationships
    q3_answer: Optional[str] = None     # Root causes (MAIN GROUND TRUTH)
    q4_answer: Optional[str] = None     # Single action possible?
    q5_answer: Optional[str] = None     # Repair actions
    q6_answer: Optional[str] = None     # Additional notes
    
    @property
    def size(self) -> int:
        """Number of tests in the cluster."""
        return len(self.tests)
    
    @property
    def cluster_path(self) -> Path:
        """Path to the cluster directory in manual_analysis."""
        return MANUAL_ANALYSIS_DIR / self.project / f"cluster{self.cluster_id}"


def parse_info_file(info_path: Path) -> tuple[list[str], list[int]]:
    """
    Parse the info.txt file to extract test names and run IDs.
    
    Args:
        info_path: Path to the info.txt file
        
    Returns:
        Tuple of (test_names, run_ids)
    """
    test_names = []
    run_ids = []
    current_section = None
    
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "test_names:":
                current_section = "tests"
            elif line == "run_ids:":
                current_section = "runs"
            elif line and current_section == "tests":
                test_names.append(line)
            elif line and current_section == "runs":
                try:
                    run_ids.append(int(line))
                except ValueError:
                    pass
    
    return test_names, run_ids


def parse_cluster_form(form_path: Path) -> dict[str, str]:
    """
    Parse the cluster_form.txt file to extract Q&A answers.
    
    Args:
        form_path: Path to the cluster_form.txt file
        
    Returns:
        Dictionary with Q1-Q6 answers
    """
    answers = {}
    current_question = None
    current_answer_lines = []
    
    # Patterns to match questions
    question_patterns = {
        "Q1": r"Q1\.",
        "Q2": r"Q2\.",
        "Q3": r"Q3\.",
        "Q4": r"Q4\.",
        "Q5": r"Q5\.",
        "Q6": r"Q6\."
    }
    
    with open(form_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line starts a new question
        new_question = None
        for q_name, pattern in question_patterns.items():
            if re.match(pattern, line):
                new_question = q_name
                break
        
        if new_question:
            # Save previous question's answer
            if current_question:
                answer = "\n".join(current_answer_lines).strip()
                answers[current_question] = answer if answer and answer != "..." else None
            
            current_question = new_question
            current_answer_lines = []
            # Skip the question line and any following empty lines
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        elif current_question:
            # Check if we've hit another question marker
            is_next_question = False
            for pattern in question_patterns.values():
                if re.match(pattern, line):
                    is_next_question = True
                    break
            
            if not is_next_question and line:
                current_answer_lines.append(line)
        
        i += 1
    
    # Save the last question's answer
    if current_question:
        answer = "\n".join(current_answer_lines).strip()
        answers[current_question] = answer if answer and answer != "..." else None
    
    return answers


def load_source_code(project: str, test_name: str) -> Optional[str]:
    """
    Load the source code for a test method.
    
    Args:
        project: Project name
        test_name: Full test name (e.g., "org.package.Class#method")
        
    Returns:
        Source code string or None if not found
    """
    # Convert test name to file name
    if "#" in test_name:
        class_name, method_name = test_name.split("#")
    else:
        parts = test_name.rsplit(".", 1)
        class_name = parts[0] if len(parts) > 1 else test_name
        method_name = parts[1] if len(parts) > 1 else ""
    
    file_name = f"{class_name}-{method_name}.java"
    source_path = SOURCES_DIR / project / "flakyMethods" / file_name
    
    if source_path.exists():
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()
    
    return None


def load_stack_traces(cluster_path: Path, test_index: int, max_traces: int = MAX_TRACES_PER_TEST) -> list[str]:
    """
    Load stack traces for a specific test in a cluster.
    
    Args:
        cluster_path: Path to the cluster directory
        test_index: Index of the test (0-based)
        max_traces: Maximum number of traces to load
        
    Returns:
        List of stack trace strings
    """
    traces = []
    
    # Traces are stored in directories named traces0, traces1, etc.
    # The number corresponds to the test index in the info.txt file
    traces_dir = cluster_path / f"traces{test_index}"
    
    if not traces_dir.exists():
        return traces
    
    # Load sample files
    sample_files = sorted(traces_dir.glob("sample*.txt"))[:max_traces]
    
    for sample_file in sample_files:
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract just the trace_text part
            if "trace_text:" in content:
                trace_start = content.index("trace_text:") + len("trace_text:")
                trace = content[trace_start:].strip()
                traces.append(trace)
            else:
                traces.append(content.strip())
    
    return traces


def load_cluster(project: str, cluster_id: int) -> Optional[Cluster]:
    """
    Load a complete cluster with all its data.
    
    Args:
        project: Project name
        cluster_id: Cluster number
        
    Returns:
        Cluster object or None if not found
    """
    cluster_path = MANUAL_ANALYSIS_DIR / project / f"cluster{cluster_id}"
    
    if not cluster_path.exists():
        return None
    
    # Load info.txt
    info_path = cluster_path / "info.txt"
    if not info_path.exists():
        return None
    
    test_names, run_ids = parse_info_file(info_path)
    
    # Load cluster_form.txt (ground truth)
    form_path = cluster_path / "cluster_form.txt"
    answers = {}
    if form_path.exists():
        answers = parse_cluster_form(form_path)
    
    # Create cluster object
    cluster = Cluster(
        project=project,
        cluster_id=cluster_id,
        run_ids=run_ids,
        q1_answer=answers.get("Q1"),
        q2_answer=answers.get("Q2"),
        q3_answer=answers.get("Q3"),
        q4_answer=answers.get("Q4"),
        q5_answer=answers.get("Q5"),
        q6_answer=answers.get("Q6")
    )
    
    # Load test cases
    for i, test_name in enumerate(test_names):
        test_case = TestCase(
            name=test_name,
            project=project,
            source_code=load_source_code(project, test_name),
            stack_traces=load_stack_traces(cluster_path, i)
        )
        cluster.tests.append(test_case)
    
    return cluster


def discover_clusters(project: str) -> list[int]:
    """
    Discover all cluster IDs for a project.
    
    Args:
        project: Project name
        
    Returns:
        List of cluster IDs
    """
    project_path = MANUAL_ANALYSIS_DIR / project
    if not project_path.exists():
        return []
    
    cluster_ids = []
    for item in project_path.iterdir():
        if item.is_dir() and item.name.startswith("cluster"):
            try:
                cluster_id = int(item.name.replace("cluster", ""))
                cluster_ids.append(cluster_id)
            except ValueError:
                pass
    
    return sorted(cluster_ids)


def load_all_clusters(require_ground_truth: bool = True) -> list[Cluster]:
    """
    Load all clusters from all projects.
    
    Args:
        require_ground_truth: If True, only include clusters with Q3 ground truth.
                             Set to False for RQ2 (clustering evaluation) to include
                             all 45 clusters / 606 tests as in the EMSE paper.
    
    Returns:
        List of all Cluster objects
    """
    all_clusters = []
    
    for project in PROJECTS_WITH_CLUSTERS:
        cluster_ids = discover_clusters(project)
        for cluster_id in cluster_ids:
            cluster = load_cluster(project, cluster_id)
            if cluster:
                if require_ground_truth and not cluster.q3_answer:
                    continue  # Skip clusters without Q3 for RQ1 (diagnosis evaluation)
                all_clusters.append(cluster)
    
    return all_clusters


def get_cluster_statistics(clusters: list[Cluster]) -> dict:
    """
    Calculate statistics about the loaded clusters.
    
    Args:
        clusters: List of Cluster objects
        
    Returns:
        Dictionary with statistics
    """
    if not clusters:
        return {}
    
    sizes = [c.size for c in clusters]
    tests_with_source = sum(
        1 for c in clusters for t in c.tests if t.source_code
    )
    tests_with_traces = sum(
        1 for c in clusters for t in c.tests if t.stack_traces
    )
    total_tests = sum(c.size for c in clusters)
    
    return {
        "total_clusters": len(clusters),
        "total_tests": total_tests,
        "projects": len(set(c.project for c in clusters)),
        "min_cluster_size": min(sizes),
        "max_cluster_size": max(sizes),
        "avg_cluster_size": sum(sizes) / len(sizes),
        "tests_with_source": tests_with_source,
        "tests_with_traces": tests_with_traces,
        "source_coverage": tests_with_source / total_tests if total_tests > 0 else 0,
        "trace_coverage": tests_with_traces / total_tests if total_tests > 0 else 0
    }


if __name__ == "__main__":
    # Test the data loader
    print("Loading all clusters...")
    clusters = load_all_clusters()
    
    print(f"\nLoaded {len(clusters)} clusters")
    
    stats = get_cluster_statistics(clusters)
    print("\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Print sample cluster info
    if clusters:
        print("\nSample cluster:")
        sample = clusters[0]
        print(f"  Project: {sample.project}")
        print(f"  Cluster ID: {sample.cluster_id}")
        print(f"  Size: {sample.size}")
        print(f"  Q3 (Ground Truth): {sample.q3_answer}")
        if sample.tests:
            print(f"  First test: {sample.tests[0].name}")
            if sample.tests[0].source_code:
                print(f"  Source code preview: {sample.tests[0].source_code[:100]}...")
            if sample.tests[0].stack_traces:
                print(f"  Stack trace preview: {sample.tests[0].stack_traces[0][:100]}...")

