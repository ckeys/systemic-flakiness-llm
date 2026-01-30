# RQ3: LLM-Generated Repair Suggestions

This module implements the RQ3 experiment for generating repair suggestions for flaky tests using the combination of RQ1 (Collective Diagnosis) and RQ2 (Automated Clustering) methods.

## Overview

RQ3 evaluates whether LLM-generated repair suggestions, informed by cluster-aware collective diagnosis, can produce fixes that are similar to developer-written repairs.

### Key Features

- **Pure code-level evaluation** (no LLM-as-Judge)
- **Stratified sampling** from IDoFT dataset
- **Three baselines** for comparison
- **Comprehensive metrics**: Exact Match, BLEU-4, CodeBLEU, Edit Similarity

## Module Structure

```
rq3/
├── __init__.py          # Package exports
├── config.py            # Configuration settings
├── models.py            # Data models (FlakyTestSample, etc.)
├── data_loader.py       # IDoFT data loading and sampling
├── github_extractor.py  # GitHub PR code extraction
├── repair_generator.py  # LLM repair generation
├── evaluation.py        # Evaluation metrics
├── baselines.py         # Baseline implementations
├── run_experiment.py    # Main experiment script
└── README.md            # This file
```

## Usage

### Run Unit Tests

```bash
cd /path/to/src

# Test each module
python -m rq3.data_loader
python -m rq3.github_extractor
python -m rq3.repair_generator
python -m rq3.evaluation
python -m rq3.baselines
```

### Run Experiment

```bash
# Dry run (no LLM calls)
python -m rq3.run_experiment --samples 100 --dry-run

# Full experiment (requires API keys)
python -m rq3.run_experiment --samples 1000 --provider openai

# Run specific methods only
python -m rq3.run_experiment --samples 100 --methods Ours B1_zero_shot
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--samples, -n` | Number of samples | 100 |
| `--provider, -p` | LLM provider (openai, anthropic, etc.) | openai |
| `--methods, -m` | Methods to run | all |
| `--skip-github` | Skip GitHub PR extraction | False |
| `--dry-run` | Only load data, no LLM calls | False |

## Methods

### Ours (Full Pipeline)

```
Flaky Tests → [RQ2: Clustering] → [RQ1: Collective Diagnosis] → Repair
```

1. Cluster tests using Hybrid-Signature (Project:Category)
2. Generate collective diagnosis for each cluster
3. Generate repair informed by cluster context

### Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **B1: Zero-Shot** | Direct LLM repair without diagnosis | Verify diagnosis value |
| **B2: Individual** | Single test diagnosis + repair | Verify collective value |
| **B3: No Clustering** | Category grouping + collective diagnosis | Verify clustering value |

## Evaluation Metrics

All metrics are **pure code-level** (no LLM-as-Judge):

| Metric | Description | Range |
|--------|-------------|-------|
| **Exact Match** | Normalized string equality | 0/1 |
| **BLEU-4** | 4-gram token overlap | 0-1 |
| **CodeBLEU** | Code-aware similarity (AST + keywords) | 0-1 |
| **Edit Similarity** | 1 - normalized edit distance | 0-1 |
| **Syntax Valid** | Basic Java syntax check | 0/1 |

## Dataset

### IDoFT (Primary)

- **Source**: `datasets/idoft/pr-data.csv`
- **Samples**: 800 (stratified by category)
- **Ground Truth**: GitHub PR diffs

### Category Allocation

| Category | Samples | Description |
|----------|---------|-------------|
| ID | 500 | Implementation-Dependent |
| OD-Vic | 100 | Order-Dependent Victim |
| NIO | 80 | Non-Idempotent-Outcome |
| OD | 60 | Order-Dependent |
| NOD | 29 | Non-Deterministic |
| Others | 31 | OD-Brit, TZD, UD, NDOD |

## Expected Results

```
| Method              | Exact Match | BLEU-4 | CodeBLEU | Edit Sim. |
|---------------------|-------------|--------|----------|-----------|
| B1: Zero-Shot       | 5%          | 0.25   | 0.30     | 0.55      |
| B2: Individual      | 8%          | 0.35   | 0.40     | 0.62      |
| B3: No Clustering   | 10%         | 0.38   | 0.43     | 0.65      |
| **Ours (Full)**     | **15%**     | **0.45**| **0.50** | **0.72**  |
```

## Cost Estimation

| Phase | Tokens | Cost |
|-------|--------|------|
| RQ2 Classification | 25K | $0.08 |
| RQ1 Diagnosis | 300K | $1.00 |
| RQ3 Repair Generation | 2M | $6.50 |
| **Total** | **~2.3M** | **~$7.6** |

*Based on GPT-4o-mini pricing ($2.5/1M input, $10/1M output)*

## Output

Results are saved to `src/output/rq3/`:

```
output/rq3/
├── results/
│   ├── RQ3_EXPERIMENT_REPORT_{timestamp}.md
│   └── rq3_results_{timestamp}.json
├── logs/
│   └── rq3_experiment_{timestamp}.log
└── cache/
    └── pr_{owner}_{repo}_{number}.json
```

## Dependencies

- Python 3.9+
- python-dotenv
- requests (for GitHub API)
- openai / anthropic (for LLM calls)

Install with:
```bash
pip install python-dotenv requests openai anthropic
```
