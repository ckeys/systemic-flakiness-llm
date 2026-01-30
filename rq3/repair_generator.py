"""
RQ3 Repair Generator

Generate repair suggestions using LLM with different configurations:
- Ours: Clustering + Collective Diagnosis + Repair
- B1: Zero-Shot (no diagnosis)
- B2: Individual Diagnosis + Repair
- B3: No Clustering + Collective Diagnosis + Repair
"""

import re
import time
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    from llm_client import LLMClient

from .models import (
    FlakyTestSample,
    RepairGenerationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CODE EXTRACTION UTILITIES
# ============================================================================

def extract_test_method(full_code: str, test_name: str, max_fallback_len: int = 3000) -> str:
    """
    Extract specific test methods from the complete test class code.
    
    Args:
        full_code: Complete test class code
        test_name: Test name, format like "org.example.TestClass.testMethod"
        max_fallback_len: If method is not found, truncate to this length
    
    Returns:
        Extracted test method code, or truncated code if not found
    """
    if not full_code:
        return "// Test code not available"
    
    # Extract method name from test_name
    if '#' in test_name:
        method_name = test_name.split('#')[-1]
    elif '.' in test_name:
        method_name = test_name.split('.')[-1]
    else:
        method_name = test_name
    
    # Try matching test method with regex
    patterns = [
        # Match @Test annotation + method
        rf'(@Test[^\n]*\n\s*(?:public\s+)?(?:void\s+)?{re.escape(method_name)}\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{{)',
        # Match regular method definition
        rf'((?:public|private|protected)?\s*(?:static)?\s*\w+\s+{re.escape(method_name)}\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{{)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_code)
        if match:
            start = match.start()
            # Find the end position of the method (matching braces)
            brace_count = 0
            end = start
            in_string = False
            escape_next = False
            
            for i, char in enumerate(full_code[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
            
            if end > start:
                # Look forward for annotations
                annotation_start = start
                lines_before = full_code[:start].split('\n')
                for line in reversed(lines_before[-5:]):
                    stripped = line.strip()
                    if stripped.startswith('@') or stripped.startswith('//') or stripped == '':
                        pos = full_code.rfind(line, 0, start)
                        if pos >= 0:
                            annotation_start = pos
                    else:
                        break
                
                return full_code[annotation_start:end].strip()
    
    # If specific method not found, return truncated code
    if len(full_code) <= max_fallback_len:
        return full_code
    
    return full_code[:max_fallback_len] + "\n// ... (code truncated)"


def get_extracted_test_code(sample: "FlakyTestSample") -> str:
    """
    Get extracted test code for all baselines.
    
    Args:
        sample: FlakyTestSample object
    
    Returns:
        Extracted test method code
    """
    return extract_test_method(
        sample.test_code_before or "",
        sample.test_name,
        max_fallback_len=3000
    )


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

SYSTEM_PROMPT_REPAIR = """You are an expert software engineer specializing in fixing flaky tests.
Your task is to generate the fixed code for a flaky test based on the provided information.

Focus on:
1. Understanding the root cause of the flakiness
2. Generating a minimal, targeted fix
3. Ensuring the fix addresses the underlying issue, not just symptoms

Output the COMPLETE fixed test method. Do NOT use diff format."""


ZERO_SHOT_REPAIR_PROMPT = """Fix this flaky test.

## Test Name
{test_name}

## Test Category
{category}

## Project
{project_name}

## Original Test Code
```java
{test_code}
```

## Task
Generate the complete fixed version of the test method above to eliminate flakiness.
The fix should be minimal and targeted to address the root cause.

Output ONLY the complete fixed test method code in a Java code block.
Do NOT include any explanations, just the code."""


INDIVIDUAL_REPAIR_PROMPT = """Fix this flaky test based on the diagnosis.

## Test Name
{test_name}

## Test Category
{category}

## Project
{project_name}

## Original Test Code
```java
{test_code}
```

## Root Cause Diagnosis
{diagnosis}

## Task
Based on the diagnosis above, generate the complete fixed version of the test method to eliminate flakiness.
The fix should be minimal and targeted to address the diagnosed root cause.

Output ONLY the complete fixed test method code in a Java code block.
Do NOT include any explanations, just the code."""


COLLECTIVE_REPAIR_PROMPT = """Fix this flaky test. Use the specific diagnosis provided, while ensuring the fix is consistent with project-wide patterns.

## Test Name
{test_name}

## Original Test Code
```java
{test_code}
```

## Relevant Diagnosis Section
{diagnosis}

## Task
Generate the complete fixed version of the test method {test_name} to eliminate flakiness.
- **CRITICAL**: ONLY fix the test named "{test_name}".
- **Goal**: Apply the fix recommended in the "SPECIFIC ISSUE" section, ensuring it doesn't conflict with "SYSTEMIC PATTERNS".
- The fix must be minimal, clean, and address the root cause.

Output ONLY the complete fixed test method code in a Java code block. No explanations."""


# ============================================================================
# REPAIR GENERATION FUNCTIONS
# ============================================================================

def generate_zero_shot_repair(
    sample: FlakyTestSample,
    llm_client: "LLMClient"
) -> RepairGenerationResult:
    """
    B1: Zero-Shot Repair - Direct LLM repair without diagnosis.
    
    Args:
        sample: Flaky test sample
        llm_client: LLM client for generation
        
    Returns:
        RepairGenerationResult
    """
    start_time = time.time()
    
    # Set component for cost tracking
    llm_client.set_component("B1_zero_shot_repair")
    
    # Build prompt - use extracted test method instead of the entire file
    extracted_code = get_extracted_test_code(sample)
    prompt = ZERO_SHOT_REPAIR_PROMPT.format(
        test_name=sample.test_name,
        category=sample.category,
        project_name=sample.project_name,
        test_code=extracted_code,
    )
    
    try:
        response = llm_client.generate(prompt, SYSTEM_PROMPT_REPAIR)
        generated_repair = extract_code_from_response(response)
        
        return RepairGenerationResult(
            sample_id=sample.sample_id,
            test_code=sample.test_code_before or "",
            diagnosis=None,
            cluster_info=None,
            generated_repair=generated_repair,
            repair_explanation=None,
            method="B1_zero_shot",
            model_name=llm_client.get_model_name(),
            generation_time_ms=int((time.time() - start_time) * 1000),
        )
        
    except Exception as e:
        logger.error(f"Zero-shot repair failed for {sample.sample_id}: {e}")
        return RepairGenerationResult(
            sample_id=sample.sample_id,
            test_code=sample.test_code_before or "",
            method="B1_zero_shot",
            model_name=llm_client.get_model_name(),
            error=str(e),
        )


def generate_individual_repair(
    sample: FlakyTestSample,
    diagnosis: str,
    llm_client: "LLMClient"
) -> RepairGenerationResult:
    """
    B2: Individual Diagnosis + Repair.
    
    Args:
        sample: Flaky test sample
        diagnosis: Individual diagnosis from RQ1
        llm_client: LLM client for generation
        
    Returns:
        RepairGenerationResult
    """
    start_time = time.time()
    
    llm_client.set_component("B2_individual_repair")
    
    # Use extracted test method instead of the entire file
    extracted_code = get_extracted_test_code(sample)
    prompt = INDIVIDUAL_REPAIR_PROMPT.format(
        test_name=sample.test_name,
        category=sample.category,
        project_name=sample.project_name,
        test_code=extracted_code,
        diagnosis=diagnosis,
    )
    
    try:
        response = llm_client.generate(prompt, SYSTEM_PROMPT_REPAIR)
        generated_repair = extract_code_from_response(response)
        
        return RepairGenerationResult(
            sample_id=sample.sample_id,
            test_code=sample.test_code_before or "",
            diagnosis=diagnosis,
            cluster_info=None,
            generated_repair=generated_repair,
            method="B2_individual",
            model_name=llm_client.get_model_name(),
            generation_time_ms=int((time.time() - start_time) * 1000),
        )
        
    except Exception as e:
        logger.error(f"Individual repair failed for {sample.sample_id}: {e}")
        return RepairGenerationResult(
            sample_id=sample.sample_id,
            test_code=sample.test_code_before or "",
            diagnosis=diagnosis,
            method="B2_individual",
            model_name=llm_client.get_model_name(),
            error=str(e),
        )


def generate_collective_repair(
    sample: FlakyTestSample,
    diagnosis: str,
    cluster_tests: List[str],
    llm_client: "LLMClient"
) -> RepairGenerationResult:
    """
    Ours: Clustering + Collective Diagnosis + Repair (Improved Logic).
    """
    start_time = time.time()
    llm_client.set_component("Ours_improved_repair")
    
    # Extract code
    extracted_code = get_extracted_test_code(sample)
    
    # Enhanced Prompt: Guide LLM to first locate the specific diagnosis, then refer to cluster patterns
    improved_prompt = f"""Fix this flaky test using the provided cluster-wide diagnosis.

## Test Name
{sample.test_name}

## Original Test Code
```java
{extracted_code}
```

## Diagnosis Analysis
{diagnosis}

## Task
1. From the "Diagnosis Analysis", identify the specific root cause and fix strategy for "{sample.test_name}".
2. Identify any systemic patterns shared across the cluster.
3. Generate a unified diff patch to fix "{sample.test_name}" by applying the specific fix and ensuring it follows the systemic patterns.

Output ONLY the unified diff patch. No explanations."""

    try:
        response = llm_client.generate(improved_prompt, SYSTEM_PROMPT_REPAIR)
        generated_repair = extract_code_from_response(response)
        
        return RepairGenerationResult(
            sample_id=sample.sample_id,
            test_code=sample.test_code_before or "",
            diagnosis=diagnosis,
            cluster_info=f"Cluster of {len(cluster_tests) + 1} tests",
            generated_repair=generated_repair,
            method="Ours",
            model_name=llm_client.get_model_name(),
            generation_time_ms=int((time.time() - start_time) * 1000),
        )
    except Exception as e:
        logger.error(f"Improved collective repair failed: {e}")
        return RepairGenerationResult(sample_id=sample.sample_id, method="Ours", error=str(e))


def extract_code_from_response(response: str) -> str:
    """
    Extract code block from LLM response.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Extracted code or original response if no code block found
    """
    if not response:
        return ""
    
    # Try to extract code from markdown code block
    import re
    
    # Pattern for ```java ... ``` or ``` ... ```
    patterns = [
        r"```java\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no code block, return the response as-is
    return response.strip()


# ============================================================================
# BATCH GENERATION
# ============================================================================

def batch_generate_repairs(
    samples: List[FlakyTestSample],
    method: str,
    llm_client: "LLMClient",
    diagnoses: Optional[Dict[str, str]] = None,
    clusters: Optional[Dict[str, List[str]]] = None,
) -> List[RepairGenerationResult]:
    """
    Generate repairs for multiple samples.
    
    Args:
        samples: List of flaky test samples
        method: "B1_zero_shot", "B2_individual", or "Ours"
        llm_client: LLM client for generation
        diagnoses: Optional dict mapping sample_id to diagnosis
        clusters: Optional dict mapping sample_id to cluster test names
        
    Returns:
        List of RepairGenerationResult
    """
    results = []
    
    for i, sample in enumerate(samples):
        if i > 0 and i % 50 == 0:
            logger.info(f"Generated repairs for {i}/{len(samples)} samples")
        
        if method == "B1_zero_shot":
            result = generate_zero_shot_repair(sample, llm_client)
            
        elif method == "B2_individual":
            diagnosis = diagnoses.get(sample.sample_id, "") if diagnoses else ""
            result = generate_individual_repair(sample, diagnosis, llm_client)
            
        elif method == "Ours":
            diagnosis = diagnoses.get(sample.sample_id, "") if diagnoses else ""
            cluster_tests = clusters.get(sample.sample_id, []) if clusters else []
            result = generate_collective_repair(sample, diagnosis, cluster_tests, llm_client)
            
        else:
            logger.error(f"Unknown method: {method}")
            continue
        
        results.append(result)
    
    logger.info(f"Generated {len(results)} repairs using method {method}")
    return results


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_extract_code_from_response():
    """Test code extraction from LLM response."""
    # Test with java code block
    response1 = """Here's the fix:

```java
@Test
public void testMethod() {
    // Fixed code
}
```

This should work now."""
    
    expected1 = """@Test
public void testMethod() {
    // Fixed code
}"""
    
    assert extract_code_from_response(response1) == expected1
    
    # Test with generic code block
    response2 = """```
public void test() {}
```"""
    
    assert extract_code_from_response(response2) == "public void test() {}"
    
    # Test without code block
    response3 = "Just some text without code"
    assert extract_code_from_response(response3) == response3
    
    # Test empty response
    assert extract_code_from_response("") == ""
    
    print("✓ test_extract_code_from_response passed")


def test_prompt_formatting():
    """Test prompt template formatting."""
    from .models import FlakyTestSample, DatasetSource
    
    sample = FlakyTestSample(
        sample_id="test_001",
        dataset_source=DatasetSource.IDOFT,
        project_url="https://github.com/test/repo",
        project_name="test-repo",
        test_name="org.example.TestClass#testMethod",
        sha_detected="abc123",
        category="ID",
        test_code_before="@Test public void testMethod() {}",
        stack_trace="java.lang.AssertionError at line 10",
    )
    
    # Test zero-shot prompt
    prompt = ZERO_SHOT_REPAIR_PROMPT.format(
        test_name=sample.test_name,
        category=sample.category,
        project_name=sample.project_name,
        test_code=sample.test_code_before,
    )
    
    assert "org.example.TestClass#testMethod" in prompt
    assert "@Test public void testMethod() {}" in prompt
    assert "ID" in prompt
    
    print("✓ test_prompt_formatting passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("Running repair_generator unit tests")
    print("=" * 60)
    
    test_extract_code_from_response()
    test_prompt_formatting()
    
    print("\n✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_tests()
