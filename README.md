# Replication Package: Collective Diagnosis and Repair of Systemic Flakiness

This repository contains the replication package for our research on automated diagnosis and repair of systemic flakiness. Our approach leverages LLMs to identify shared root causes across clusters of flaky tests and generate informed repair suggestions.

## Overview

Systemic flakiness refers to multiple flaky tests sharing a common root cause (e.g., environment configuration, resource leaks). This package implements:
1. **RQ1 (Collective Diagnosis)**: Evaluating LLMs' ability to diagnose shared root causes by analyzing clusters of tests collectively vs. individually.
2. **RQ2 (Automated Clustering)**: Identifying systemic flakiness clusters using a Three-Tier Exception Categorization and AST-based features.
3. **RQ3 (Informed Repair)**: Generating repair suggestions for flaky tests informed by collective diagnosis.

## Project Structure

```text
.
├── config.py              # Global configuration and path management
├── data_loader.py         # Utilities for loading datasets (Systemic Flakiness & IDoFT)
├── llm_client.py          # Unified interface for LLM providers (OpenAI, Anthropic, etc.)
├── requirements.txt       # Python dependencies
├── env.example            # Template for environment variables (API keys)
│
├── RQ1: Collective Diagnosis
│   ├── run_experiment.py      # Main entry point for RQ1 experiments
│   ├── analyzers.py           # Implementation of Individual vs. Collective strategies
│   ├── evaluator.py           # Automated evaluation metrics (Semantic Similarity, BERTScore)
│   └── analyze_results.py     # Scripts for statistical analysis and visualization
│
├── RQ2: Automated Clustering
│   ├── run_rq2_experiment.py  # Main entry point for RQ2 experiments
│   ├── rq2_feature_extractor.py # Three-Tier Exception Categorization logic
│   ├── rq2_ast_features.py      # AST-based feature extraction using tree-sitter
│   ├── rq2_clustering.py        # Implementation of clustering algorithms (Hybrid-Signature)
│   ├── rq2_evaluation.py        # Metrics for clustering (ARI, NMI, Silhouette)
│   └── rq2_verification.py      # LLM-based cluster refinement logic
│
├── RQ3: Informed Repair
│   └── rq3/                   # Self-contained module for repair generation
│       ├── run_experiment.py  # Main entry point for RQ3 experiments
│       ├── repair_generator.py # LLM-based repair generation informed by diagnosis
│       ├── evaluation.py      # Code-level metrics (CodeBLEU, BLEU-4, Edit Sim)
│       ├── baselines.py       # Implementation of B1 (Zero-shot) and B2 (Individual)
│       └── models.py          # Data models for repair samples
│
└── output/                    # Pre-computed results reported in the paper
    ├── results/               # RQ1 & RQ2 raw data and summary reports
    ├── logs/                  # Execution logs for verification
    ├── human_evaluation/      # Human-annotated ground truth and IRR analysis
    └── rq3/                   # RQ3 specific results and evaluation reports
```

## Setup

### 1. Prerequisites
- Python 3.9+
- Java 8+ (for some AST parsing components)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Copy the environment template and add your API keys:
```bash
cp env.example .env
# Edit .env and provide at least one LLM API key (e.g., OPENAI_API_KEY)
```

## Running Experiments

### RQ1: Collective Diagnosis
To evaluate the diagnostic accuracy of LLMs:
```bash
python run_experiment.py --provider openai --max-clusters 45
```

### RQ2: Automated Clustering
To run the clustering pipeline and evaluate against ground truth:
```bash
python run_rq2_experiment.py
```

### RQ3: Informed Repair
To generate and evaluate repair suggestions (using IDoFT dataset):
```bash
cd rq3
python run_experiment.py --samples 1000 --provider openai
```

## Experimental Results (Paper Alignment)

The `output/` directory contains the exact results used in our paper:

- **RQ1 Results**: `output/results/rq1_report_20260123_155158.txt`
- **RQ2 Results**: `output/results/RQ2_EXPERIMENT_REPORT.md` (Best ARI: 0.548)
- **RQ3 Results**: `output/rq3/results/RQ3_EXPERIMENT_REPORT_20260128_164449.md`
- **Human Study**: `output/human_evaluation/agreement_analysis_summary.json`

## Dataset Information

This package expects datasets to be organized in a `../datasets/` directory:
- `systemic-flakiness/`: Contains the 45 co-failing clusters.
- `idoft/`: Contains the IDoFT `pr-data.csv` for repair evaluation.

## Citation

If you use this replication package or our findings in your research, please cite:

```bibtex
@inproceedings{your-citation-here,
  title={Automated Diagnosis and Repair of Systemic Flakiness},
  author={...},
  booktitle={...},
  year={2026}
}
```
