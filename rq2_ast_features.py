#!/usr/bin/env python3
"""
RQ2 AST Feature Extraction using Tree-sitter

This module extracts AST-based features from Java test code using tree-sitter,
a production-grade incremental parsing library used by GitHub and VS Code.

Reference: https://tree-sitter.github.io/tree-sitter/
"""

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

# Initialize Java parser
JAVA_LANGUAGE = Language(tsjava.language())
_parser = None

def get_parser() -> Parser:
    """Get or create the Java parser (singleton)."""
    global _parser
    if _parser is None:
        _parser = Parser(JAVA_LANGUAGE)
    return _parser


@dataclass
class JavaASTFeatures:
    """
    AST features extracted from a Java test method.
    
    These features capture structural and semantic properties that may indicate
    shared root causes for flaky test failures.
    """
    # Method invocations
    method_calls: Set[str] = field(default_factory=set)
    method_call_count: int = 0
    
    # Assertions (JUnit)
    assertion_types: Set[str] = field(default_factory=set)
    assertion_count: int = 0
    
    # Control flow
    if_count: int = 0
    for_count: int = 0
    while_count: int = 0
    try_count: int = 0
    catch_count: int = 0
    
    # Annotations
    annotations: Set[str] = field(default_factory=set)
    
    # Exception handling
    caught_exceptions: Set[str] = field(default_factory=set)
    thrown_exceptions: Set[str] = field(default_factory=set)
    
    # Object creation
    instantiated_classes: Set[str] = field(default_factory=set)
    
    # Field/variable access
    field_accesses: Set[str] = field(default_factory=set)
    
    # Literals
    string_literal_count: int = 0
    number_literal_count: int = 0
    
    # Code complexity
    node_count: int = 0
    max_depth: int = 0
    
    # Domain-specific indicators (from class names and method calls)
    has_network_api: bool = False
    has_file_api: bool = False
    has_thread_api: bool = False
    has_time_api: bool = False


# Known assertion methods (JUnit 4/5, Hamcrest, AssertJ)
ASSERTION_METHODS = {
    'assertEquals', 'assertNotEquals', 'assertTrue', 'assertFalse',
    'assertNull', 'assertNotNull', 'assertSame', 'assertNotSame',
    'assertArrayEquals', 'assertThrows', 'assertDoesNotThrow',
    'assertThat', 'assertAll', 'fail',
    # Hamcrest
    'assertThat', 'is', 'equalTo', 'hasItem', 'containsString',
    # AssertJ
    'assertThat', 'isEqualTo', 'isNotNull', 'contains', 'hasSize',
}

# Network-related APIs
NETWORK_APIS = {
    'HttpClient', 'HttpURLConnection', 'URL', 'Socket', 'ServerSocket',
    'DatagramSocket', 'InetAddress', 'HttpRequest', 'HttpResponse',
    'RestTemplate', 'WebClient', 'OkHttpClient', 'CloseableHttpClient',
    'connect', 'openConnection', 'getInputStream', 'getOutputStream',
}

# File system APIs
FILE_APIS = {
    'File', 'Path', 'Files', 'FileInputStream', 'FileOutputStream',
    'BufferedReader', 'BufferedWriter', 'FileReader', 'FileWriter',
    'RandomAccessFile', 'FileChannel', 'DirectoryStream',
    'createFile', 'createDirectory', 'delete', 'exists', 'readAllBytes',
}

# Threading/concurrency APIs
THREAD_APIS = {
    'Thread', 'Runnable', 'Callable', 'Future', 'CompletableFuture',
    'ExecutorService', 'Executors', 'ThreadPoolExecutor',
    'Lock', 'ReentrantLock', 'Semaphore', 'CountDownLatch', 'CyclicBarrier',
    'synchronized', 'wait', 'notify', 'notifyAll', 'join',
    'AtomicInteger', 'AtomicBoolean', 'ConcurrentHashMap',
}

# Time-related APIs
TIME_APIS = {
    'sleep', 'wait', 'timeout', 'Timeout', 'Duration',
    'System.currentTimeMillis', 'System.nanoTime',
    'Instant', 'LocalDateTime', 'ZonedDateTime',
    'ScheduledExecutorService', 'Timer', 'TimerTask',
    'awaitTermination', 'awaitility',
}


def extract_ast_features(source_code: str) -> JavaASTFeatures:
    """
    Extract AST features from Java source code using tree-sitter.
    
    Args:
        source_code: Java source code (can be a method or full class)
        
    Returns:
        JavaASTFeatures dataclass with extracted features
    """
    parser = get_parser()
    tree = parser.parse(bytes(source_code, 'utf8'))
    
    features = JavaASTFeatures()
    
    def visit(node: Node, depth: int = 0):
        """Recursively visit AST nodes and extract features."""
        features.node_count += 1
        features.max_depth = max(features.max_depth, depth)
        
        node_type = node.type
        
        # Method invocations
        if node_type == 'method_invocation':
            method_name = _get_method_name(node)
            if method_name:
                features.method_calls.add(method_name)
                features.method_call_count += 1
                
                # Check for assertions
                if method_name in ASSERTION_METHODS or method_name.startswith('assert'):
                    features.assertion_types.add(method_name)
                    features.assertion_count += 1
                
                # Check for domain-specific APIs
                if method_name in NETWORK_APIS:
                    features.has_network_api = True
                if method_name in FILE_APIS:
                    features.has_file_api = True
                if method_name in THREAD_APIS:
                    features.has_thread_api = True
                if method_name in TIME_APIS:
                    features.has_time_api = True
        
        # Control flow
        elif node_type == 'if_statement':
            features.if_count += 1
        elif node_type == 'for_statement' or node_type == 'enhanced_for_statement':
            features.for_count += 1
        elif node_type == 'while_statement':
            features.while_count += 1
        elif node_type == 'try_statement':
            features.try_count += 1
        elif node_type == 'catch_clause':
            features.catch_count += 1
            # Extract caught exception type
            catch_type = _get_catch_type(node)
            if catch_type:
                features.caught_exceptions.add(catch_type)
        
        # Annotations
        elif node_type == 'marker_annotation' or node_type == 'annotation':
            annotation_name = _get_annotation_name(node)
            if annotation_name:
                features.annotations.add(annotation_name)
        
        # Object instantiation
        elif node_type == 'object_creation_expression':
            class_name = _get_created_class(node)
            if class_name:
                features.instantiated_classes.add(class_name)
                
                # Check for domain-specific classes
                if class_name in NETWORK_APIS:
                    features.has_network_api = True
                if class_name in FILE_APIS:
                    features.has_file_api = True
                if class_name in THREAD_APIS:
                    features.has_thread_api = True
        
        # Field access
        elif node_type == 'field_access':
            field_name = _get_field_name(node)
            if field_name:
                features.field_accesses.add(field_name)
        
        # Literals
        elif node_type == 'string_literal':
            features.string_literal_count += 1
        elif node_type in ('decimal_integer_literal', 'decimal_floating_point_literal'):
            features.number_literal_count += 1
        
        # Throws declaration
        elif node_type == 'throws':
            for child in node.children:
                if child.type == 'type_identifier':
                    features.thrown_exceptions.add(child.text.decode())
        
        # Recurse into children
        for child in node.children:
            visit(child, depth + 1)
    
    visit(tree.root_node)
    return features


def _get_method_name(node: Node) -> str:
    """Extract method name from method_invocation node."""
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode()
        elif child.type == 'field_access':
            # e.g., System.out.println -> println
            for subchild in child.children:
                if subchild.type == 'identifier':
                    return subchild.text.decode()
    return ""


def _get_annotation_name(node: Node) -> str:
    """Extract annotation name from annotation node."""
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode()
    return ""


def _get_catch_type(node: Node) -> str:
    """Extract exception type from catch_clause node."""
    for child in node.children:
        if child.type == 'catch_formal_parameter':
            for subchild in child.children:
                if subchild.type == 'catch_type':
                    for type_child in subchild.children:
                        if type_child.type == 'type_identifier':
                            return type_child.text.decode()
    return ""


def _get_created_class(node: Node) -> str:
    """Extract class name from object_creation_expression node."""
    for child in node.children:
        if child.type == 'type_identifier':
            return child.text.decode()
        elif child.type == 'generic_type':
            for subchild in child.children:
                if subchild.type == 'type_identifier':
                    return subchild.text.decode()
    return ""


def _get_field_name(node: Node) -> str:
    """Extract field name from field_access node."""
    children = list(node.children)
    if len(children) >= 2:
        return children[-1].text.decode()
    return ""


def compute_ast_distance(f1: JavaASTFeatures, f2: JavaASTFeatures) -> List[float]:
    """
    Compute pairwise AST-based distance features between two tests.
    
    All distances are normalized to [0, 1] where 0 = identical, 1 = completely different.
    
    Returns:
        List of 15 distance features
    """
    def jaccard_distance(s1: Set, s2: Set) -> float:
        """Jaccard distance: 1 - |intersection| / |union|"""
        if not s1 and not s2:
            return 0.0  # Both empty = identical
        if not s1 or not s2:
            return 1.0  # One empty = completely different
        return 1 - len(s1 & s2) / len(s1 | s2)
    
    def normalized_diff(v1: int, v2: int) -> float:
        """Normalized absolute difference"""
        if v1 == 0 and v2 == 0:
            return 0.0
        return abs(v1 - v2) / max(v1, v2)
    
    def bool_diff(b1: bool, b2: bool) -> float:
        """Boolean difference: 0 if same, 1 if different"""
        return 0.0 if b1 == b2 else 1.0
    
    features = []
    
    # 1. Method call similarity (Jaccard distance)
    features.append(jaccard_distance(f1.method_calls, f2.method_calls))
    
    # 2. Assertion type similarity
    features.append(jaccard_distance(f1.assertion_types, f2.assertion_types))
    
    # 3. Annotation similarity
    features.append(jaccard_distance(f1.annotations, f2.annotations))
    
    # 4. Instantiated class similarity
    features.append(jaccard_distance(f1.instantiated_classes, f2.instantiated_classes))
    
    # 5. Caught exception similarity
    features.append(jaccard_distance(f1.caught_exceptions, f2.caught_exceptions))
    
    # 6. Control flow difference (normalized sum of if/for/while/try/catch)
    cf1 = f1.if_count + f1.for_count + f1.while_count + f1.try_count + f1.catch_count
    cf2 = f2.if_count + f2.for_count + f2.while_count + f2.try_count + f2.catch_count
    features.append(normalized_diff(cf1, cf2))
    
    # 7. Method call count difference
    features.append(normalized_diff(f1.method_call_count, f2.method_call_count))
    
    # 8. Assertion count difference
    features.append(normalized_diff(f1.assertion_count, f2.assertion_count))
    
    # 9. AST node count difference (complexity)
    features.append(normalized_diff(f1.node_count, f2.node_count))
    
    # 10. AST depth difference
    features.append(normalized_diff(f1.max_depth, f2.max_depth))
    
    # 11-14. Domain API differences (binary)
    features.append(bool_diff(f1.has_network_api, f2.has_network_api))
    features.append(bool_diff(f1.has_file_api, f2.has_file_api))
    features.append(bool_diff(f1.has_thread_api, f2.has_thread_api))
    features.append(bool_diff(f1.has_time_api, f2.has_time_api))
    
    # 15. Field access similarity
    features.append(jaccard_distance(f1.field_accesses, f2.field_accesses))
    
    return features


# Feature names for documentation
AST_FEATURE_NAMES = [
    "ast_method_call_jaccard",      # 1. Jaccard distance of method calls
    "ast_assertion_type_jaccard",   # 2. Jaccard distance of assertion types
    "ast_annotation_jaccard",       # 3. Jaccard distance of annotations
    "ast_class_instantiation_jaccard",  # 4. Jaccard distance of instantiated classes
    "ast_caught_exception_jaccard", # 5. Jaccard distance of caught exceptions
    "ast_control_flow_diff",        # 6. Normalized control flow difference
    "ast_method_call_count_diff",   # 7. Normalized method call count difference
    "ast_assertion_count_diff",     # 8. Normalized assertion count difference
    "ast_node_count_diff",          # 9. Normalized AST node count difference
    "ast_depth_diff",               # 10. Normalized AST depth difference
    "ast_has_network_api_diff",     # 11. Network API usage difference
    "ast_has_file_api_diff",        # 12. File API usage difference
    "ast_has_thread_api_diff",      # 13. Thread API usage difference
    "ast_has_time_api_diff",        # 14. Time API usage difference
    "ast_field_access_jaccard",     # 15. Jaccard distance of field accesses
]


def test_ast_extraction():
    """Test AST feature extraction on sample code."""
    code1 = '''
    @Test
    public void testConnection() {
        HttpClient client = new HttpClient();
        try {
            client.connect("localhost", 8080);
            Thread.sleep(1000);
            assertEquals(200, client.getStatus());
        } catch (IOException e) {
            fail("Connection failed");
        }
    }
    '''
    
    code2 = '''
    @Test
    @Timeout(5)
    public void testFileRead() {
        File file = new File("/tmp/test.txt");
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String content = reader.readLine();
            assertNotNull(content);
            assertTrue(content.length() > 0);
        } catch (FileNotFoundException e) {
            fail("File not found");
        }
    }
    '''
    
    f1 = extract_ast_features(code1)
    f2 = extract_ast_features(code2)
    
    print("=== Code 1 Features ===")
    print(f"Method calls: {f1.method_calls}")
    print(f"Assertions: {f1.assertion_types}")
    print(f"Annotations: {f1.annotations}")
    print(f"Caught exceptions: {f1.caught_exceptions}")
    print(f"Instantiated classes: {f1.instantiated_classes}")
    print(f"Has network API: {f1.has_network_api}")
    print(f"Has file API: {f1.has_file_api}")
    print(f"Has thread API: {f1.has_thread_api}")
    print(f"Has time API: {f1.has_time_api}")
    print(f"Node count: {f1.node_count}, Max depth: {f1.max_depth}")
    
    print("\n=== Code 2 Features ===")
    print(f"Method calls: {f2.method_calls}")
    print(f"Assertions: {f2.assertion_types}")
    print(f"Annotations: {f2.annotations}")
    print(f"Caught exceptions: {f2.caught_exceptions}")
    print(f"Instantiated classes: {f2.instantiated_classes}")
    print(f"Has network API: {f2.has_network_api}")
    print(f"Has file API: {f2.has_file_api}")
    print(f"Has thread API: {f2.has_thread_api}")
    print(f"Has time API: {f2.has_time_api}")
    print(f"Node count: {f2.node_count}, Max depth: {f2.max_depth}")
    
    print("\n=== Distance Features ===")
    distances = compute_ast_distance(f1, f2)
    for name, dist in zip(AST_FEATURE_NAMES, distances):
        print(f"{name}: {dist:.3f}")
    
    print("\n✓ AST extraction test passed!")


if __name__ == "__main__":
    test_ast_extraction()
