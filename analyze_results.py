"""
Results Analysis and Visualization for RQ1 Experiment

This script analyzes the experiment results and generates visualizations
for the paper.
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import RESULTS_DIR, OUTPUT_DIR, ROOT_CAUSE_CATEGORIES


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(results_file: Path) -> dict:
    """Load experiment results from JSON file."""
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert detailed results to a pandas DataFrame."""
    detailed = results.get("detailed_results", [])
    return pd.DataFrame(detailed)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """
    Perform statistical tests comparing Individual vs Collective analysis.
    
    Returns:
        Dictionary with test results
    """
    # Separate by method
    individual = df[df["method"] == "individual"]["similarity_score"].values
    collective = df[df["method"] == "collective"]["similarity_score"].values
    
    results = {}
    
    # Paired t-test (if same clusters were analyzed)
    if len(individual) == len(collective):
        t_stat, t_pvalue = stats.ttest_rel(collective, individual)
        results["paired_ttest"] = {
            "statistic": t_stat,
            "p_value": t_pvalue,
            "significant": t_pvalue < 0.05
        }
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_pvalue = stats.wilcoxon(collective, individual)
            results["wilcoxon"] = {
                "statistic": w_stat,
                "p_value": w_pvalue,
                "significant": w_pvalue < 0.05
            }
        except ValueError:
            # Wilcoxon fails if all differences are zero
            results["wilcoxon"] = {"error": "All differences are zero"}
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(individual)**2 + np.std(collective)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(collective) - np.mean(individual)) / pooled_std
        results["effect_size"] = {
            "cohens_d": cohens_d,
            "interpretation": interpret_cohens_d(cohens_d)
        }
    
    return results


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def analyze_by_cluster_size(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by cluster size."""
    # Create size categories
    def categorize_size(n):
        if n <= 5:
            return "Small (2-5)"
        elif n <= 15:
            return "Medium (6-15)"
        else:
            return "Large (>15)"
    
    # Group by size category and method
    # Note: We need to add cluster size info from original data
    # This is a placeholder - actual implementation would need cluster size data
    return df.groupby("method")["similarity_score"].agg(["mean", "std", "count"])


def analyze_by_root_cause_category(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by root cause category."""
    return df.groupby(["actual_category", "method"])["similarity_score"].agg(
        ["mean", "std", "count"]
    ).unstack()


# ============================================================================
# VISUALIZATION
# ============================================================================

def set_plot_style():
    """Set consistent plot style for paper figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (10, 6),
        "figure.dpi": 150
    })


def plot_similarity_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create a box plot comparing similarity scores between methods.
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create box plot
    sns.boxplot(
        data=df,
        x="method",
        y="similarity_score",
        palette=["#3498db", "#e74c3c"],
        ax=ax
    )
    
    # Add individual points
    sns.stripplot(
        data=df,
        x="method",
        y="similarity_score",
        color="black",
        alpha=0.3,
        ax=ax
    )
    
    ax.set_xlabel("Analysis Method")
    ax.set_ylabel("Semantic Similarity Score (1-5)")
    ax.set_title("RQ1: Individual vs Collective Analysis")
    ax.set_xticklabels(["Individual", "Collective"])
    ax.set_ylim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_correctness_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create a stacked bar chart comparing correctness rates.
    """
    set_plot_style()
    
    # Calculate correctness rates
    correctness_data = df.groupby(["method", "is_correct"]).size().unstack(fill_value=0)
    correctness_pct = correctness_data.div(correctness_data.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors
    colors = {"Yes": "#2ecc71", "Partial": "#f39c12", "No": "#e74c3c"}
    
    # Create stacked bar chart
    bottom = np.zeros(len(correctness_pct))
    for correctness in ["Yes", "Partial", "No"]:
        if correctness in correctness_pct.columns:
            values = correctness_pct[correctness].values
            ax.bar(
                correctness_pct.index,
                values,
                bottom=bottom,
                label=correctness,
                color=colors[correctness]
            )
            bottom += values
    
    ax.set_xlabel("Analysis Method")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Diagnosis Correctness by Method")
    ax.legend(title="Correctness")
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Create a heatmap showing performance by root cause category.
    """
    set_plot_style()
    
    # Calculate mean similarity by category and method
    pivot = df.pivot_table(
        values="similarity_score",
        index="actual_category",
        columns="method",
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=1,
        vmax=5,
        ax=ax
    )
    
    ax.set_title("Mean Similarity Score by Root Cause Category")
    ax.set_xlabel("Analysis Method")
    ax.set_ylabel("Root Cause Category")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_paired_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create a scatter plot showing paired comparison of methods.
    """
    set_plot_style()
    
    # Pivot to get paired data
    pivot = df.pivot_table(
        values="similarity_score",
        index=["cluster_project", "cluster_id"],
        columns="method"
    ).reset_index()
    
    if "individual" not in pivot.columns or "collective" not in pivot.columns:
        print("Cannot create paired comparison: missing method data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(
        pivot["individual"],
        pivot["collective"],
        alpha=0.6,
        s=100,
        c="#3498db"
    )
    
    # Add diagonal line (y=x)
    ax.plot([1, 5], [1, 5], "k--", alpha=0.5, label="Equal performance")
    
    # Add labels for points above/below diagonal
    above = pivot[pivot["collective"] > pivot["individual"]]
    below = pivot[pivot["collective"] < pivot["individual"]]
    equal = pivot[pivot["collective"] == pivot["individual"]]
    
    ax.set_xlabel("Individual Analysis Score")
    ax.set_ylabel("Collective Analysis Score")
    ax.set_title(f"Paired Comparison\n(Collective better: {len(above)}, Individual better: {len(below)}, Tie: {len(equal)})")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_aspect("equal")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_project_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create a grouped bar chart showing performance by project.
    """
    set_plot_style()
    
    # Calculate mean similarity by project and method
    project_means = df.groupby(["cluster_project", "method"])["similarity_score"].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(project_means))
    width = 0.35
    
    if "individual" in project_means.columns:
        ax.bar(x - width/2, project_means["individual"], width, label="Individual", color="#3498db")
    if "collective" in project_means.columns:
        ax.bar(x + width/2, project_means["collective"], width, label="Collective", color="#e74c3c")
    
    ax.set_xlabel("Project")
    ax.set_ylabel("Mean Similarity Score")
    ax.set_title("Performance by Project")
    ax.set_xticks(x)
    ax.set_xticklabels(project_means.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table of results."""
    # Summary statistics
    summary = df.groupby("method")["similarity_score"].agg(["mean", "std", "median"])
    
    latex = """
\\begin{table}[h]
\\centering
\\caption{RQ1 Results: Individual vs Collective Analysis}
\\label{tab:rq1-results}
\\begin{tabular}{lccc}
\\toprule
Method & Mean & Std & Median \\\\
\\midrule
"""
    
    for method in summary.index:
        row = summary.loc[method]
        latex += f"{method.capitalize()} & {row['mean']:.2f} & {row['std']:.2f} & {row['median']:.1f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def generate_report(results: dict, df: pd.DataFrame, output_path: Path):
    """Generate a comprehensive analysis report."""
    report = []
    report.append("# RQ1 Experiment Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    # Experiment info
    report.append("## Experiment Information")
    for key, value in results.get("experiment_info", {}).items():
        report.append(f"- {key}: {value}")
    report.append("")
    
    # Dataset statistics
    report.append("## Dataset Statistics")
    for key, value in results.get("dataset_statistics", {}).items():
        if isinstance(value, float):
            report.append(f"- {key}: {value:.2f}")
        else:
            report.append(f"- {key}: {value}")
    report.append("")
    
    # Main results
    report.append("## Main Results")
    metrics = results.get("aggregate_metrics", {})
    
    if "individual_mean_similarity" in metrics:
        report.append(f"- Individual Analysis Mean Similarity: {metrics['individual_mean_similarity']:.3f}")
    if "collective_mean_similarity" in metrics:
        report.append(f"- Collective Analysis Mean Similarity: {metrics['collective_mean_similarity']:.3f}")
    if "similarity_improvement" in metrics:
        report.append(f"- Improvement: {metrics['similarity_improvement']:.3f}")
    report.append("")
    
    # Win/Loss analysis
    if "collective_wins" in metrics:
        report.append("## Win/Loss Analysis")
        report.append(f"- Collective Wins: {metrics['collective_wins']}")
        report.append(f"- Individual Wins: {metrics['individual_wins']}")
        report.append(f"- Ties: {metrics['ties']}")
        report.append("")
    
    # Statistical tests
    if len(df) > 0:
        report.append("## Statistical Tests")
        stat_results = perform_statistical_tests(df)
        
        if "paired_ttest" in stat_results:
            t = stat_results["paired_ttest"]
            report.append(f"- Paired t-test: t={t['statistic']:.3f}, p={t['p_value']:.4f}")
            report.append(f"  - Significant at α=0.05: {t['significant']}")
        
        if "wilcoxon" in stat_results and "error" not in stat_results["wilcoxon"]:
            w = stat_results["wilcoxon"]
            report.append(f"- Wilcoxon signed-rank: W={w['statistic']:.3f}, p={w['p_value']:.4f}")
            report.append(f"  - Significant at α=0.05: {w['significant']}")
        
        if "effect_size" in stat_results:
            e = stat_results["effect_size"]
            report.append(f"- Effect size (Cohen's d): {e['cohens_d']:.3f} ({e['interpretation']})")
        report.append("")
    
    # Category analysis
    report.append("## Performance by Root Cause Category")
    if "actual_category" in df.columns:
        category_stats = df.groupby(["actual_category", "method"])["similarity_score"].mean().unstack()
        report.append(category_stats.to_string())
    report.append("")
    
    # LaTeX table
    report.append("## LaTeX Table")
    report.append("```latex")
    report.append(generate_latex_table(df))
    report.append("```")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved report: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze RQ1 experiment results")
    
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the results JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR / "analysis"),
        help="Directory to save analysis outputs"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_results(Path(args.results_file))
    df = results_to_dataframe(results)
    
    print(f"Loaded {len(df)} evaluation results")
    
    # Generate report
    report_path = output_dir / "analysis_report.md"
    generate_report(results, df, report_path)
    
    # Generate plots
    if not args.no_plots and len(df) > 0:
        print("\nGenerating visualizations...")
        
        plot_similarity_comparison(df, output_dir / "similarity_comparison.png")
        plot_correctness_comparison(df, output_dir / "correctness_comparison.png")
        plot_paired_comparison(df, output_dir / "paired_comparison.png")
        plot_project_comparison(df, output_dir / "project_comparison.png")
        
        if "actual_category" in df.columns and df["actual_category"].notna().any():
            plot_category_heatmap(df, output_dir / "category_heatmap.png")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

