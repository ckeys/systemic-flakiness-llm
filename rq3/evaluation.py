"""
RQ3 Evaluation Metrics

Pure code-level evaluation metrics (no LLM-as-Judge):
- BLEU-4
- CodeBLEU
- Edit Distance (Normalized)
- Syntax Validity
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

from .models import (
    RepairGenerationResult,
    RepairEvaluationResult,
    ExperimentResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DIFF PROCESSING
# ============================================================================

def extract_added_code_from_diff(diff: str) -> str:
    """
    Extract only the added code lines from a unified diff.
    
    This removes:
    - @@ line number markers
    - diff headers (---, +++, diff --git, index)
    - Context lines (lines without +/-)
    - Deleted lines (- prefix)
    
    Only keeps:
    - Added lines (+ prefix), with the + removed
    
    This is the standard approach in code repair papers for comparing
    generated repairs with ground truth patches.
    
    Args:
        diff: Unified diff string
        
    Returns:
        String containing only the added code lines
    """
    if not diff:
        return ""
    
    lines = diff.split('\n')
    added_lines = []
    
    for line in lines:
        # Skip diff metadata
        if line.startswith('@@'):
            continue
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('diff --git'):
            continue
        if line.startswith('index '):
            continue
        
        # Extract added lines (remove + prefix)
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])
    
    return '\n'.join(added_lines)


def extract_all_code_from_diff(diff: str, include_context: bool = False) -> str:
    """
    Extract all code from a unified diff (added + optionally context).
    
    Args:
        diff: Unified diff string
        include_context: Whether to include context lines
        
    Returns:
        String containing the extracted code
    """
    if not diff:
        return ""
    
    lines = diff.split('\n')
    code_lines = []
    
    for line in lines:
        # Skip diff metadata
        if line.startswith('@@'):
            continue
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('diff --git'):
            continue
        if line.startswith('index '):
            continue
        
        # Added lines
        if line.startswith('+') and not line.startswith('+++'):
            code_lines.append(line[1:])
        # Context lines
        elif include_context and line.startswith(' '):
            code_lines.append(line[1:])
    
    return '\n'.join(code_lines)


def extract_changed_lines_from_diff(diff: str) -> Tuple[str, str]:
    """
    Extract changed lines from a unified diff.
    
    Returns both removed (-) and added (+) lines separately.
    This is useful for comparing what was changed vs what was generated.
    
    Args:
        diff: Unified diff string
        
    Returns:
        Tuple of (removed_lines, added_lines)
    """
    if not diff:
        return "", ""
    
    lines = diff.split('\n')
    removed_lines = []
    added_lines = []
    
    for line in lines:
        # Skip diff metadata
        if line.startswith('@@'):
            continue
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('diff --git'):
            continue
        if line.startswith('index '):
            continue
        
        # Removed lines
        if line.startswith('-') and not line.startswith('---'):
            removed_lines.append(line[1:])
        # Added lines
        elif line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])
    
    return '\n'.join(removed_lines), '\n'.join(added_lines)


def extract_modified_method_from_code(
    full_code: str, 
    diff: str,
    context_lines: int = 3
) -> str:
    """
    Extract only the modified method/section from full code based on diff.
    
    This helps focus evaluation on just the changed portion rather than
    the entire file.
    
    Args:
        full_code: Complete source code
        diff: Unified diff showing changes
        context_lines: Number of context lines around changes
        
    Returns:
        Extracted code section containing the changes
    """
    if not diff or not full_code:
        return full_code
    
    # Parse diff to find line numbers of changes
    changed_line_numbers = set()
    
    lines = diff.split('\n')
    current_line = 0
    
    for line in lines:
        # Parse @@ hunk headers to get line numbers
        # Format: @@ -old_start,old_count +new_start,new_count @@
        if line.startswith('@@'):
            import re
            match = re.search(r'\+(\d+)', line)
            if match:
                current_line = int(match.group(1)) - 1  # 0-indexed
            continue
        
        # Track line numbers for added/modified lines
        if line.startswith('+') and not line.startswith('+++'):
            changed_line_numbers.add(current_line)
            current_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            # Deleted lines don't increment in new file
            pass
        elif line.startswith(' '):
            # Context line
            current_line += 1
        elif not line.startswith('diff') and not line.startswith('index'):
            current_line += 1
    
    if not changed_line_numbers:
        return full_code
    
    # Calculate range to extract
    min_line = max(0, min(changed_line_numbers) - context_lines)
    max_line = max(changed_line_numbers) + context_lines
    
    # Extract lines
    code_lines = full_code.split('\n')
    extracted = code_lines[min_line:max_line + 1]
    
    return '\n'.join(extracted)


# ============================================================================
# CODE NORMALIZATION
# ============================================================================

def normalize_code(code: str) -> str:
    """
    Normalize code for comparison.
    
    - Remove leading/trailing whitespace
    - Normalize line endings
    - Remove empty lines
    - Normalize indentation (optional)
    """
    if not code:
        return ""
    
    # Normalize line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    
    # Split into lines, strip each, remove empty
    lines = [line.strip() for line in code.split("\n")]
    lines = [line for line in lines if line]
    
    return "\n".join(lines)


def compute_exact_match(generated: str, ground_truth: str) -> bool:
    """
    Check if generated code exactly matches ground truth after normalization.
    
    Args:
        generated: Generated repair code
        ground_truth: Ground truth repair code
        
    Returns:
        True if exact match
    """
    return normalize_code(generated) == normalize_code(ground_truth)


# ============================================================================
# BLEU-4
# ============================================================================

def tokenize_code(code: str) -> List[str]:
    """
    Tokenize code for BLEU calculation.
    
    Uses a simple tokenization that splits on whitespace and punctuation.
    """
    if not code:
        return []
    
    # Split on whitespace and common code punctuation
    # Keep operators and punctuation as separate tokens
    pattern = r'(\s+|[{}()\[\];,.<>=!&|+\-*/^%@#])'
    tokens = re.split(pattern, code)
    
    # Filter empty tokens and whitespace-only tokens
    tokens = [t for t in tokens if t and not t.isspace()]
    
    return tokens


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Compute n-grams from token list."""
    if len(tokens) < n:
        return Counter()
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    
    return Counter(ngrams)


def compute_bleu_4(
    generated: str,
    ground_truth: str,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """
    Compute BLEU-4 score between generated and ground truth code.
    
    Args:
        generated: Generated code
        ground_truth: Ground truth code
        weights: Weights for 1-4 gram precision (default: uniform)
        
    Returns:
        BLEU-4 score (0-1)
    """
    gen_tokens = tokenize_code(generated)
    ref_tokens = tokenize_code(ground_truth)
    
    if not gen_tokens or not ref_tokens:
        return 0.0
    
    # Compute n-gram precisions
    precisions = []
    
    for n in range(1, 5):
        gen_ngrams = compute_ngrams(gen_tokens, n)
        ref_ngrams = compute_ngrams(ref_tokens, n)
        
        if not gen_ngrams:
            precisions.append(0.0)
            continue
        
        # Count matches (clipped by reference count)
        matches = 0
        for ngram, count in gen_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        total = sum(gen_ngrams.values())
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)
    
    # Check for zero precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    import math
    log_precisions = [math.log(p) for p in precisions]
    weighted_log_precision = sum(w * lp for w, lp in zip(weights, log_precisions))
    
    # Brevity penalty
    bp = 1.0
    if len(gen_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(gen_tokens))
    
    bleu = bp * math.exp(weighted_log_precision)
    
    return min(bleu, 1.0)


# ============================================================================
# EDIT DISTANCE (NORMALIZED)
# ============================================================================

def compute_edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    Uses dynamic programming for efficiency.
    """
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    m, n = len(s1), len(s2)
    
    # Use two rows for space efficiency
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr_row[0] = i
        
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,      # deletion
                curr_row[j - 1] + 1,  # insertion
                prev_row[j - 1] + cost  # substitution
            )
        
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[n]


def compute_edit_similarity(generated: str, ground_truth: str) -> float:
    """
    Compute normalized edit similarity (1 - normalized_edit_distance).
    
    Args:
        generated: Generated code
        ground_truth: Ground truth code
        
    Returns:
        Similarity score (0-1), where 1 means identical
    """
    if not generated and not ground_truth:
        return 1.0
    
    # Normalize first
    gen_norm = normalize_code(generated)
    gt_norm = normalize_code(ground_truth)
    
    edit_dist = compute_edit_distance(gen_norm, gt_norm)
    max_len = max(len(gen_norm), len(gt_norm))
    
    if max_len == 0:
        return 1.0
    
    similarity = 1.0 - (edit_dist / max_len)
    return max(0.0, similarity)


# ============================================================================
# CODEBLEU (Simplified)
# ============================================================================

def compute_codebleu(
    generated: str,
    ground_truth: str,
    lang: str = "java"
) -> float:
    """
    Compute CodeBLEU score (simplified version).
    
    Full CodeBLEU requires AST parsing and data flow analysis.
    This simplified version combines:
    - Token BLEU (50%)
    - Keyword match (25%)
    - Structural similarity (25%)
    
    Args:
        generated: Generated code
        ground_truth: Ground truth code
        lang: Programming language
        
    Returns:
        CodeBLEU score (0-1)
    """
    # Component 1: Token BLEU (standard BLEU-4)
    token_bleu = compute_bleu_4(generated, ground_truth)
    
    # Component 2: Keyword match
    keyword_match = compute_keyword_match(generated, ground_truth, lang)
    
    # Component 3: Structural similarity (based on brackets/braces)
    structural_sim = compute_structural_similarity(generated, ground_truth)
    
    # Weighted combination
    codebleu = 0.50 * token_bleu + 0.25 * keyword_match + 0.25 * structural_sim
    
    return codebleu


def compute_keyword_match(generated: str, ground_truth: str, lang: str = "java") -> float:
    """Compute keyword overlap between generated and ground truth."""
    # Java keywords
    java_keywords = {
        "public", "private", "protected", "static", "final", "abstract",
        "class", "interface", "extends", "implements", "import", "package",
        "void", "int", "long", "double", "float", "boolean", "char", "byte",
        "if", "else", "for", "while", "do", "switch", "case", "break",
        "continue", "return", "try", "catch", "finally", "throw", "throws",
        "new", "this", "super", "null", "true", "false",
        "@Test", "@Before", "@After", "@BeforeEach", "@AfterEach",
        "assert", "assertEquals", "assertTrue", "assertFalse", "assertNotNull",
    }
    
    gen_tokens = set(tokenize_code(generated))
    gt_tokens = set(tokenize_code(ground_truth))
    
    gen_keywords = gen_tokens & java_keywords
    gt_keywords = gt_tokens & java_keywords
    
    if not gt_keywords:
        return 1.0 if not gen_keywords else 0.5
    
    # Jaccard similarity of keywords
    intersection = len(gen_keywords & gt_keywords)
    union = len(gen_keywords | gt_keywords)
    
    return intersection / union if union > 0 else 0.0


def compute_structural_similarity(generated: str, ground_truth: str) -> float:
    """Compute structural similarity based on bracket patterns."""
    def extract_structure(code: str) -> str:
        """Extract structural elements (brackets, braces, etc.)."""
        structural_chars = set("{}()[]")
        return "".join(c for c in code if c in structural_chars)
    
    gen_struct = extract_structure(generated)
    gt_struct = extract_structure(ground_truth)
    
    if not gt_struct:
        return 1.0 if not gen_struct else 0.5
    
    # Use edit similarity on structural patterns
    return compute_edit_similarity(gen_struct, gt_struct)


# ============================================================================
# SYNTAX VALIDITY
# ============================================================================

def check_java_syntax(code: str) -> bool:
    """
    Basic Java syntax check (heuristic-based).
    
    Checks:
    - Balanced braces
    - Balanced parentheses
    - Balanced brackets
    - Basic structure
    
    Note: This is not a full syntax check, just basic validation.
    """
    if not code:
        return False
    
    # Check balanced delimiters
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    
    for char in code:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    
    if stack:
        return False
    
    # Check for basic Java structure
    # Should have at least one method-like pattern
    has_method = bool(re.search(r'\w+\s*\([^)]*\)\s*\{', code))
    
    return has_method


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def normalize_diff(diff: str) -> str:
    """
    Normalize a diff for comparison - keep only the actual code changes.
    
    - Remove markdown code block markers (```diff, ```)
    - Remove diff headers (diff --git, index, ---, +++)
    - Remove @@ line number markers (we don't care about line numbers)
    - Remove context lines (lines starting with space)
    - Keep only +/- lines (the actual changes)
    - Remove the +/- prefix to compare just the code
    
    Args:
        diff: Unified diff string
        
    Returns:
        Normalized diff string containing only the changed code
    """
    if not diff:
        return ""
    
    lines = diff.split('\n')
    normalized_lines = []
    
    for line in lines:
        # Skip markdown code block markers
        if line.strip().startswith('```'):
            continue
        # Skip file headers
        if line.startswith('diff --git'):
            continue
        if line.startswith('index '):
            continue
        if line.startswith('---') or line.startswith('+++'):
            continue
        # Skip @@ line number markers (we don't care about line numbers)
        if line.startswith('@@'):
            continue
        # Skip context lines (we only care about actual changes)
        if line.startswith(' '):
            continue
        # Skip empty lines
        if not line.strip():
            continue
        
        # Keep +/- lines but remove the prefix to compare just the code
        if line.startswith('+') or line.startswith('-'):
            # Keep the +/- prefix to distinguish additions from deletions
            normalized_lines.append(line.rstrip())
    
    return '\n'.join(normalized_lines)


def evaluate_repair_diff_compare(
    generated_diff: str,
    ground_truth_diff: str,
    sample_id: str,
    method: str = "",
) -> RepairEvaluationResult:
    """
    Evaluate generated repair by comparing complete diffs.
    
    This compares the entire diff content (including context, additions, deletions)
    rather than just the added lines.
    
    Args:
        generated_diff: Generated diff (unified format)
        ground_truth_diff: Ground truth diff (unified format)
        sample_id: Sample identifier
        method: Method name for tracking
        
    Returns:
        RepairEvaluationResult with metrics computed on full diff comparison
    """
    # Normalize both diffs
    gen_normalized = normalize_diff(generated_diff)
    gt_normalized = normalize_diff(ground_truth_diff)
    
    return RepairEvaluationResult(
        sample_id=sample_id,
        bleu_4=compute_bleu_4(gen_normalized, gt_normalized),
        codebleu=compute_codebleu(gen_normalized, gt_normalized),
        edit_similarity=compute_edit_similarity(gen_normalized, gt_normalized),
        method=method,
        ground_truth_available=bool(gt_normalized),
    )


def evaluate_repair_diff_only(
    generated: str,
    ground_truth_diff: str,
    sample_id: str,
    method: str = "",
) -> RepairEvaluationResult:
    """
    Evaluate generated repair by comparing ONLY the changed lines from diff.
    
    This is a more focused evaluation that:
    1. Extracts only the added lines (+) from the ground truth diff
    2. Compares them with the generated repair
    
    This approach is useful when:
    - You want to evaluate the quality of the actual fix
    - Not penalize for differences in unchanged code
    - Focus on the repair logic itself
    
    Args:
        generated: Generated repair code
        ground_truth_diff: Ground truth diff (unified format)
        sample_id: Sample identifier
        method: Method name for tracking
        
    Returns:
        RepairEvaluationResult with metrics computed on diff-only comparison
    """
    # Extract only added lines from ground truth diff
    _, gt_added = extract_changed_lines_from_diff(ground_truth_diff)
    
    # For generated code, try to extract if it looks like a diff
    if generated.startswith('@@') or generated.startswith('diff --git'):
        _, gen_code = extract_changed_lines_from_diff(generated)
    elif generated.startswith('+'):
        # Might be just added lines
        gen_code = extract_added_code_from_diff(generated)
    else:
        # Assume it's already the repair code
        gen_code = generated
    
    return RepairEvaluationResult(
        sample_id=sample_id,
        bleu_4=compute_bleu_4(gen_code, gt_added),
        codebleu=compute_codebleu(gen_code, gt_added),
        edit_similarity=compute_edit_similarity(gen_code, gt_added),
        method=method,
        ground_truth_available=bool(gt_added),
    )


def evaluate_repair(
    generated: str,
    ground_truth: str,
    sample_id: str,
    method: str = "",
    ground_truth_is_complete_code: Optional[bool] = None
) -> RepairEvaluationResult:
    """
    Evaluate a generated repair against ground truth.
    
    Args:
        generated: Generated repair code
        ground_truth: Ground truth repair code
        sample_id: Sample identifier
        method: Method name for tracking
        ground_truth_is_complete_code: If True, ground_truth is the complete fixed code
                                       If False, ground_truth is a diff
                                       If None, automatically detect
        
    Returns:
        RepairEvaluationResult with all metrics
    """
    # Determine how to process ground truth
    is_gt_diff = False
    if ground_truth_is_complete_code is False:
        is_gt_diff = True
    elif ground_truth_is_complete_code is None:
        # Heuristic to detect diff
        if ground_truth.startswith('@@') or ground_truth.startswith('---') or ground_truth.startswith('diff --git'):
            is_gt_diff = True
    
    if is_gt_diff:
        # Ground truth is a diff, extract the added code
        gt_code = extract_added_code_from_diff(ground_truth)
    else:
        # Ground truth is already complete code
        gt_code = ground_truth
    
    # Process generated code (extract if it looks like a diff)
    if generated.startswith('@@') or generated.startswith('+') or generated.startswith('diff --git'):
        gen_code = extract_added_code_from_diff(generated)
    else:
        gen_code = generated
    
    return RepairEvaluationResult(
        sample_id=sample_id,
        bleu_4=compute_bleu_4(gen_code, gt_code),
        codebleu=compute_codebleu(gen_code, gt_code),
        edit_similarity=compute_edit_similarity(gen_code, gt_code),
        method=method,
        ground_truth_available=bool(gt_code),
    )


def batch_evaluate(
    results: List[RepairGenerationResult],
    ground_truths: Dict[str, str],
) -> List[RepairEvaluationResult]:
    """
    Evaluate multiple repair results.
    
    Args:
        results: List of RepairGenerationResult
        ground_truths: Dict mapping sample_id to ground truth code
        
    Returns:
        List of RepairEvaluationResult
    """
    evaluations = []
    
    for result in results:
        gt = ground_truths.get(result.sample_id, "")
        generated = result.generated_repair or ""
        
        evaluation = evaluate_repair(
            generated=generated,
            ground_truth=gt,
            sample_id=result.sample_id,
            method=result.method,
        )
        evaluations.append(evaluation)
    
    return evaluations


def aggregate_results(
    evaluations: List[RepairEvaluationResult],
    method: str
) -> ExperimentResult:
    """
    Aggregate evaluation results into summary statistics.
    
    Args:
        evaluations: List of RepairEvaluationResult
        method: Method name
        
    Returns:
        ExperimentResult with aggregated metrics
    """
    if not evaluations:
        return ExperimentResult(method=method, total_samples=0)
    
    n = len(evaluations)
    
    return ExperimentResult(
        method=method,
        total_samples=n,
        avg_bleu_4=sum(e.bleu_4 for e in evaluations) / n,
        avg_codebleu=sum(e.codebleu for e in evaluations) / n,
        avg_edit_similarity=sum(e.edit_similarity for e in evaluations) / n,
        individual_results=evaluations,
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_normalize_code():
    """Test code normalization."""
    code1 = "  @Test\n  public void test() {\n  }\n  "
    expected = "@Test\npublic void test() {\n}"
    assert normalize_code(code1) == expected
    
    # Test with different line endings
    code2 = "@Test\r\npublic void test() {}\r\n"
    assert "@Test" in normalize_code(code2)
    
    print("✓ test_normalize_code passed")


def test_compute_exact_match():
    """Test exact match computation."""
    code1 = "@Test\npublic void test() {}"
    code2 = "  @Test\n  public void test() {}  "
    code3 = "@Test\npublic void different() {}"
    
    assert compute_exact_match(code1, code2) is True
    assert compute_exact_match(code1, code3) is False
    
    print("✓ test_compute_exact_match passed")


def test_tokenize_code():
    """Test code tokenization."""
    code = "public void test() { return x + 1; }"
    tokens = tokenize_code(code)
    
    assert "public" in tokens
    assert "void" in tokens
    assert "test" in tokens
    assert "(" in tokens
    assert ")" in tokens
    assert "{" in tokens
    assert "return" in tokens
    assert "+" in tokens
    
    print("✓ test_tokenize_code passed")


def test_compute_bleu_4():
    """Test BLEU-4 computation."""
    # Identical code should have high BLEU
    code = "@Test public void test() { assertEquals(1, 1); }"
    bleu = compute_bleu_4(code, code)
    assert bleu > 0.99, f"Expected ~1.0, got {bleu}"
    
    # Completely different code should have low BLEU
    code1 = "public void methodA() { return 1; }"
    code2 = "private int methodB() { return 2; }"
    bleu = compute_bleu_4(code1, code2)
    assert bleu < 0.5, f"Expected < 0.5, got {bleu}"
    
    # Empty code
    assert compute_bleu_4("", "some code") == 0.0
    
    print("✓ test_compute_bleu_4 passed")


def test_compute_edit_distance():
    """Test edit distance computation."""
    assert compute_edit_distance("", "") == 0
    assert compute_edit_distance("abc", "") == 3
    assert compute_edit_distance("", "abc") == 3
    assert compute_edit_distance("abc", "abc") == 0
    assert compute_edit_distance("abc", "abd") == 1
    assert compute_edit_distance("abc", "adc") == 1
    assert compute_edit_distance("abc", "axc") == 1
    
    print("✓ test_compute_edit_distance passed")


def test_compute_edit_similarity():
    """Test edit similarity computation."""
    # Identical strings
    sim = compute_edit_similarity("abc", "abc")
    assert sim == 1.0, f"Expected 1.0, got {sim}"
    
    # One character difference
    sim = compute_edit_similarity("abc", "abd")
    assert 0.6 < sim < 0.7, f"Expected ~0.67, got {sim}"
    
    # Completely different
    sim = compute_edit_similarity("abc", "xyz")
    assert sim == 0.0, f"Expected 0.0, got {sim}"
    
    print("✓ test_compute_edit_similarity passed")


def test_check_java_syntax():
    """Test Java syntax checking."""
    # Valid code
    assert check_java_syntax("@Test public void test() { int x = 1; }") is True
    assert check_java_syntax("public void method() {}") is True
    
    # Invalid - unbalanced braces
    assert check_java_syntax("public void test() {") is False
    assert check_java_syntax("public void test() }") is False
    
    # Invalid - no method structure
    assert check_java_syntax("just some text") is False
    
    print("✓ test_check_java_syntax passed")


def test_compute_codebleu():
    """Test CodeBLEU computation."""
    # Identical code
    code = "@Test public void test() { assertEquals(1, 1); }"
    codebleu = compute_codebleu(code, code)
    assert codebleu > 0.95, f"Expected > 0.95, got {codebleu}"
    
    # Similar code
    code1 = "@Test public void testA() { assertEquals(1, 1); }"
    code2 = "@Test public void testB() { assertEquals(2, 2); }"
    codebleu = compute_codebleu(code1, code2)
    assert 0.5 < codebleu < 1.0, f"Expected 0.5-1.0, got {codebleu}"
    
    print("✓ test_compute_codebleu passed")


def test_evaluate_repair():
    """Test full repair evaluation."""
    generated = "@Test public void test() { assertEquals(1, 1); }"
    ground_truth = "@Test public void test() { assertEquals(1, 1); }"
    
    result = evaluate_repair(generated, ground_truth, "test_001", "Ours")
    
    assert result.bleu_4 > 0.99
    assert result.codebleu > 0.95
    assert result.edit_similarity > 0.99
    assert result.sample_id == "test_001"
    assert result.method == "Ours"
    
    print("✓ test_evaluate_repair passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("Running evaluation unit tests")
    print("=" * 60)
    
    test_normalize_code()
    test_compute_exact_match()
    test_tokenize_code()
    test_compute_bleu_4()
    test_compute_edit_distance()
    test_compute_edit_similarity()
    test_check_java_syntax()
    test_compute_codebleu()
    test_evaluate_repair()
    
    print("\n✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_tests()
