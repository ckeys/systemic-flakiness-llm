"""
Configuration for RQ1 Experiment: Automated Root Cause Diagnosis for Systemic Flakiness

API Keys Setup:
    Create a .env file in the src/ directory with the following keys:
    
    # OpenAI (GPT-4o)
    OPENAI_API_KEY=sk-your-openai-key-here
    
    # Anthropic (Claude 3.5 Sonnet)
    ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
    
    # DeepSeek (DeepSeek Coder)
    DEEPSEEK_API_KEY=sk-your-deepseek-key-here
    
    # Together AI (Llama 3.1 70B)
    TOGETHER_API_KEY=your-together-key-here
    
    # Groq (Llama 3.1 70B - faster, has free tier)
    GROQ_API_KEY=gsk_your-groq-key-here

Get API keys from:
    - OpenAI: https://platform.openai.com/api-keys
    - Anthropic: https://console.anthropic.com/
    - DeepSeek: https://platform.deepseek.com/
    - Together AI: https://www.together.ai/
    - Groq: https://console.groq.com/
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    # If .env file cannot be read, continue anyway (env vars can be set externally)
    import sys
    print(f"Warning: Could not load .env file: {e}", file=sys.stderr)

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "systemic-flakiness" / "clusters_experiment"

# Dataset paths
MANUAL_ANALYSIS_DIR = DATASET_ROOT / "manual_analysis"
SOURCES_DIR = DATASET_ROOT / "sources"
SAMPLES_DIR = DATASET_ROOT / "samples"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "src" / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# API Keys (loaded from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model settings
LLM_CONFIG = {
    "openai": {
        "model": "gpt-4o-mini",  # Changed from gpt-4o to save cost (15x cheaper)
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": None,  # Use default
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": None,  # Uses Anthropic SDK
    },
    "deepseek": {
        "model": "deepseek-coder",
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": "https://api.deepseek.com",
    },
    "together": {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": "https://api.together.xyz/v1",
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",  # Updated from deprecated llama-3.1-70b-versatile
        "temperature": 0.1,
        "max_tokens": 2000,
        "base_url": "https://api.groq.com/openai/v1",
    },
}

# Default LLM provider
DEFAULT_LLM_PROVIDER = "openai"

# Provider to friendly name mapping (for display)
PROVIDER_DISPLAY_NAMES = {
    "openai": "GPT-4o-mini",
    "anthropic": "Claude 3.5 Sonnet",
    "deepseek": "DeepSeek Coder",
    "together": "Llama 3.3 70B (Together)",
    "groq": "Llama 3.3 70B (Groq)",
}

# Rate limiting settings
# Minimum seconds between API requests (increase if hitting rate limits)
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "1.0"))

# Maximum retries for rate limit errors
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# Base delay (seconds) for exponential backoff on rate limits
RATE_LIMIT_BASE_DELAY = float(os.getenv("RATE_LIMIT_BASE_DELAY", "5.0"))

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Number of runs per cluster for handling non-determinism
RUNS_PER_CLUSTER = 3

# Maximum number of tests to include in collective analysis
# (to avoid exceeding context limits)
MAX_TESTS_PER_CLUSTER = 10

# Maximum number of stack traces to include per test
MAX_TRACES_PER_TEST = 3

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Semantic similarity scoring scale
SIMILARITY_SCALE = {
    1: "Completely different root causes",
    2: "Related but different specific causes",
    3: "Same general category, different details",
    4: "Same root cause, minor wording differences",
    5: "Essentially identical diagnosis"
}

# Root cause categories for classification
ROOT_CAUSE_CATEGORIES = [
    "Networking",           # DNS, connection issues, socket errors
    "Resource",             # Memory, threads, file handles
    "Configuration",        # Version mismatch, missing config
    "External Dependency",  # Third-party services
    "Concurrency",          # Race conditions, deadlocks
    "Timing",               # Timeouts, delays
    "Filesystem",           # File locks, path issues
    "Other"
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# PROJECTS WITH CLUSTERS (from clusters_table.csv)
# ============================================================================

PROJECTS_WITH_CLUSTERS = [
    "Alluxio-alluxio",
    "apache-ambari",
    "apache-hbase",
    "elasticjob-elastic-job-lite",
    "hector-client-hector",
    "kevinsawicki-http-request",
    "spring-projects-spring-boot",
    "square-okhttp",
    "wildfly-wildfly",
    "wro4j-wro4j"
]

