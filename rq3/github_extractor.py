"""
GitHub PR Code Extractor

Extract repair code (before/after) from GitHub PRs.
This module fetches the actual code changes from PR links in IDoFT dataset.
"""

import os
import re
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from urllib.parse import urlparse
from dataclasses import dataclass

import os
from dotenv import load_dotenv

# Load .env file from src directory
load_dotenv(Path(__file__).parent.parent / ".env")

from .config import (
    GITHUB_RATE_LIMIT_DELAY,
    GITHUB_MAX_RETRIES,
    RQ3_CACHE_DIR,
)
from .models import FlakyTestSample

# Get GitHub token from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

logger = logging.getLogger(__name__)


@dataclass
class PRCodePair:
    """Code before and after a PR fix."""
    file_path: str
    code_before: str
    code_after: str
    diff: str
    pr_number: int
    repo_owner: str
    repo_name: str
    base_sha: Optional[str] = None  # Base commit SHA (before the fix)
    head_sha: Optional[str] = None  # Head commit SHA (after the fix)


def parse_pr_url(pr_url: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse GitHub PR URL to extract owner, repo, and PR number.
    
    Args:
        pr_url: GitHub PR URL (e.g., "https://github.com/apache/hbase/pull/123")
        
    Returns:
        Tuple of (owner, repo, pr_number) or None if invalid
    """
    if not pr_url:
        return None
    
    # Pattern: https://github.com/{owner}/{repo}/pull/{number}
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, pr_url)
    
    if match:
        owner, repo, pr_number = match.groups()
        return owner, repo, int(pr_number)
    
    return None


def get_cache_key(owner: str, repo: str, pr_number: int) -> str:
    """Generate cache key for a PR."""
    return f"{owner}_{repo}_{pr_number}"


def load_from_cache(cache_key: str) -> Optional[Dict]:
    """Load PR data from cache."""
    cache_file = RQ3_CACHE_DIR / f"pr_{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
    return None


def save_to_cache(cache_key: str, data: Dict):
    """Save PR data to cache."""
    cache_file = RQ3_CACHE_DIR / f"pr_{cache_key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file}: {e}")


def fetch_pr_metadata(
    owner: str,
    repo: str,
    pr_number: int,
    use_cache: bool = True
) -> Optional[Dict]:
    """
    Fetch PR metadata including base and head SHA.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: PR number
        use_cache: Whether to use cached data
        
    Returns:
        PR metadata dictionary or None if failed
    """
    try:
        import requests
    except ImportError:
        logger.error("requests package not installed. Run: pip install requests")
        return None
    
    cache_key = f"{get_cache_key(owner, repo, pr_number)}_metadata"
    
    # Check cache
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached:
            logger.debug(f"Using cached metadata for {cache_key}")
            return cached
    
    # Build API URL for PR details
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant metadata
            metadata = {
                "base_sha": data.get("base", {}).get("sha"),
                "head_sha": data.get("head", {}).get("sha"),
                "base_ref": data.get("base", {}).get("ref"),
                "head_ref": data.get("head", {}).get("ref"),
                "merged": data.get("merged", False),
                "merge_commit_sha": data.get("merge_commit_sha"),
            }
            
            # Cache the result
            save_to_cache(cache_key, metadata)
            
            return metadata
            
        else:
            logger.warning(f"Failed to fetch PR metadata: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to fetch PR metadata: {e}")
        return None


def fetch_pr_files(
    owner: str,
    repo: str,
    pr_number: int,
    use_cache: bool = True
) -> Optional[List[Dict]]:
    """
    Fetch files changed in a PR using GitHub API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: PR number
        use_cache: Whether to use cached data
        
    Returns:
        List of file change dictionaries or None if failed
    """
    try:
        import requests
    except ImportError:
        logger.error("requests package not installed. Run: pip install requests")
        return None
    
    cache_key = get_cache_key(owner, repo, pr_number)
    
    # Check cache
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached:
            logger.debug(f"Using cached data for {cache_key}")
            return cached.get("files", [])
    
    # Build API URL
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    
    # Set up headers
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    # Fetch with retry
    for attempt in range(GITHUB_MAX_RETRIES):
        try:
            response = requests.get(api_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                files = response.json()
                
                # Cache the result
                save_to_cache(cache_key, {"files": files, "pr_number": pr_number})
                
                return files
                
            elif response.status_code == 403:
                # Rate limited
                logger.warning(f"Rate limited. Waiting {GITHUB_RATE_LIMIT_DELAY * 10}s...")
                time.sleep(GITHUB_RATE_LIMIT_DELAY * 10)
                
            elif response.status_code == 404:
                logger.warning(f"PR not found: {owner}/{repo}#{pr_number}")
                return None
                
            else:
                logger.warning(f"API error {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            
        time.sleep(GITHUB_RATE_LIMIT_DELAY)
    
    return None


def fetch_file_content(
    owner: str,
    repo: str,
    file_path: str,
    ref: str  # commit SHA or branch
) -> Optional[str]:
    """
    Fetch file content at a specific commit.
    
    Args:
        owner: Repository owner
        repo: Repository name
        file_path: Path to file in repo
        ref: Commit SHA or branch name
        
    Returns:
        File content as string or None if failed
    """
    try:
        import requests
        import base64
    except ImportError:
        logger.error("requests package not installed")
        return None
    
    # Build API URL
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    params = {"ref": ref}
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Content is base64 encoded
            if data.get("encoding") == "base64":
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
            else:
                return data.get("content", "")
                
        elif response.status_code == 404:
            logger.debug(f"File not found: {file_path} at {ref}")
            return None
            
        else:
            logger.warning(f"Failed to fetch file: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to fetch file content: {e}")
        return None


def extract_test_file_changes(
    pr_files: List[Dict],
    test_name: str
) -> List[Dict]:
    """
    Extract file changes related to a specific test.
    
    Args:
        pr_files: List of files changed in PR
        test_name: Fully-qualified test name
        
    Returns:
        List of relevant file changes
    """
    # Convert test name to potential file path patterns
    # e.g., "org.apache.hbase.TestClass#testMethod" -> "TestClass.java"
    test_class = test_name.split("#")[0] if "#" in test_name else test_name
    simple_class_name = test_class.split(".")[-1]
    
    relevant_files = []
    
    for file_info in pr_files:
        filename = file_info.get("filename", "")
        
        # Check if this file might contain the test
        if filename.endswith(".java"):
            # Check if class name matches
            if simple_class_name in filename:
                relevant_files.append(file_info)
            # Also check if it's in a test directory
            elif "/test/" in filename or "/tests/" in filename:
                relevant_files.append(file_info)
    
    return relevant_files


def fetch_test_code_after(
    owner: str,
    repo: str,
    pr_number: int,
    file_path: str,
    use_cache: bool = True
) -> Optional[Tuple[str, str]]:
    """
    Fetch the test code AFTER the PR fix (the complete fixed code).
    
    This fetches the file content at the head commit (after the fix was applied).
    This is the Ground Truth for code repair evaluation.
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: PR number
        file_path: Path to the test file in the repo
        use_cache: Whether to use cached data
        
    Returns:
        Tuple of (code_after, head_sha) or None if failed
    """
    # First, get PR metadata to find the head SHA
    metadata = fetch_pr_metadata(owner, repo, pr_number, use_cache)
    
    if not metadata or not metadata.get("head_sha"):
        logger.warning(f"Could not get head SHA for PR {owner}/{repo}#{pr_number}")
        return None
    
    head_sha = metadata["head_sha"]
    
    # Fetch file content at head commit (after fix)
    code_after = fetch_file_content(owner, repo, file_path, head_sha)
    
    if code_after:
        return code_after, head_sha
    else:
        logger.warning(f"Could not fetch file {file_path} at {head_sha}")
        return None


def fetch_test_code_before(
    owner: str,
    repo: str,
    pr_number: int,
    file_path: str,
    use_cache: bool = True
) -> Optional[Tuple[str, str]]:
    """
    Fetch the test code BEFORE the PR fix.
    
    This fetches the file content at the base commit (before the fix was applied).
    
    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: PR number
        file_path: Path to the test file in the repo
        use_cache: Whether to use cached data
        
    Returns:
        Tuple of (code_before, base_sha) or None if failed
    """
    # First, get PR metadata to find the base SHA
    metadata = fetch_pr_metadata(owner, repo, pr_number, use_cache)
    
    if not metadata or not metadata.get("base_sha"):
        logger.warning(f"Could not get base SHA for PR {owner}/{repo}#{pr_number}")
        return None
    
    base_sha = metadata["base_sha"]
    
    # Fetch file content at base commit
    code_before = fetch_file_content(owner, repo, file_path, base_sha)
    
    if code_before:
        return code_before, base_sha
    else:
        logger.warning(f"Could not fetch file {file_path} at {base_sha}")
        return None


def extract_repair_pair(
    sample: FlakyTestSample,
    use_cache: bool = True
) -> Optional[PRCodePair]:
    """
    Extract before/after code pair from a PR for a flaky test sample.
    
    Args:
        sample: FlakyTestSample with PR link
        use_cache: Whether to use cached data
        
    Returns:
        PRCodePair with before/after code or None if failed
    """
    if not sample.pr_link:
        logger.debug(f"No PR link for sample {sample.sample_id}")
        return None
    
    # Parse PR URL
    parsed = parse_pr_url(sample.pr_link)
    if not parsed:
        logger.warning(f"Could not parse PR URL: {sample.pr_link}")
        return None
    
    owner, repo, pr_number = parsed
    
    # Fetch PR files
    pr_files = fetch_pr_files(owner, repo, pr_number, use_cache)
    if not pr_files:
        return None
    
    # Find test-related file changes
    test_files = extract_test_file_changes(pr_files, sample.test_name)
    
    if not test_files:
        logger.debug(f"No test file changes found in PR for {sample.test_name}")
        return None
    
    # Use the first relevant file (could be improved to find exact match)
    file_info = test_files[0]
    
    return PRCodePair(
        file_path=file_info.get("filename", ""),
        code_before="",  # Would need additional API calls
        code_after="",   # Would need additional API calls
        diff=file_info.get("patch", ""),
        pr_number=pr_number,
        repo_owner=owner,
        repo_name=repo,
    )


def batch_extract_repairs(
    samples: List[FlakyTestSample],
    max_samples: Optional[int] = None,
    delay_between_requests: float = 1.0
) -> Dict[str, PRCodePair]:
    """
    Extract repair pairs for multiple samples.
    
    Args:
        samples: List of FlakyTestSample objects
        max_samples: Maximum number of samples to process
        delay_between_requests: Delay between API requests
        
    Returns:
        Dictionary mapping sample_id to PRCodePair
    """
    results = {}
    
    samples_to_process = samples[:max_samples] if max_samples else samples
    
    for i, sample in enumerate(samples_to_process):
        if i > 0 and i % 10 == 0:
            logger.info(f"Processed {i}/{len(samples_to_process)} samples")
        
        try:
            repair_pair = extract_repair_pair(sample)
            if repair_pair:
                results[sample.sample_id] = repair_pair
                
        except Exception as e:
            logger.warning(f"Failed to extract repair for {sample.sample_id}: {e}")
        
        time.sleep(delay_between_requests)
    
    logger.info(f"Extracted {len(results)} repair pairs from {len(samples_to_process)} samples")
    return results


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_parse_pr_url():
    """Test PR URL parsing."""
    # Valid URLs
    result = parse_pr_url("https://github.com/apache/hbase/pull/123")
    assert result == ("apache", "hbase", 123), f"Got {result}"
    
    result = parse_pr_url("https://github.com/owner/repo/pull/456")
    assert result == ("owner", "repo", 456), f"Got {result}"
    
    # With trailing slash
    result = parse_pr_url("https://github.com/apache/hbase/pull/123/")
    assert result == ("apache", "hbase", 123), f"Got {result}"
    
    # Invalid URLs
    assert parse_pr_url("") is None
    assert parse_pr_url("https://github.com/apache/hbase") is None
    assert parse_pr_url("not a url") is None
    
    print("✓ test_parse_pr_url passed")


def test_extract_test_file_changes():
    """Test extraction of test file changes."""
    mock_files = [
        {"filename": "src/main/java/org/example/Service.java"},
        {"filename": "src/test/java/org/example/TestService.java"},
        {"filename": "src/test/java/org/example/TestClass.java"},
        {"filename": "README.md"},
    ]
    
    # Test with class name - should find TestClass.java and also TestService.java (both in test/)
    result = extract_test_file_changes(mock_files, "org.example.TestClass#testMethod")
    # The function finds files by class name match OR by being in test directory
    assert len(result) >= 1
    filenames = [r["filename"] for r in result]
    assert "src/test/java/org/example/TestClass.java" in filenames
    
    # Test with different class
    result = extract_test_file_changes(mock_files, "org.example.TestService#test")
    assert len(result) >= 1
    filenames = [r["filename"] for r in result]
    assert "src/test/java/org/example/TestService.java" in filenames
    
    print("✓ test_extract_test_file_changes passed")


def test_get_cache_key():
    """Test cache key generation."""
    key = get_cache_key("apache", "hbase", 123)
    assert key == "apache_hbase_123"
    
    print("✓ test_get_cache_key passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("Running github_extractor unit tests")
    print("=" * 60)
    
    test_parse_pr_url()
    test_extract_test_file_changes()
    test_get_cache_key()
    
    print("\n✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_tests()
