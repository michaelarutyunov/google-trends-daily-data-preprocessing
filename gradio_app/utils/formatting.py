"""
Formatting utilities for Gradio app.
Formats results for display and creates downloadable packages.
"""

import json
import sys
import tempfile
from pathlib import Path
from io import BytesIO
import zipfile

import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import ConfigManager


def format_metrics_markdown(diagnostics: dict) -> str:
    """
    Format metrics as markdown table for display.

    Args:
        diagnostics: Diagnostics dictionary from stitching result

    Returns:
        Formatted markdown string
    """
    md = "### 📊 Key Metrics\n\n"

    # Main metrics table
    md += "| Metric | Monthly | Weekly |\n"
    md += "|--------|---------|--------|\n"

    # MAE
    monthly_mae = diagnostics.get('monthly_mae', 0)
    weekly_mae = diagnostics.get('weekly_mae', 0)
    md += f"| MAE | {monthly_mae:.2f} | {weekly_mae:.2f} |\n"

    # Correlation
    monthly_corr = diagnostics.get('monthly_corr', 0)
    weekly_corr = diagnostics.get('weekly_corr', 0)
    md += f"| Correlation | {monthly_corr:.3f} | {weekly_corr:.3f} |\n"

    # Bias %
    monthly_bias = diagnostics.get('monthly_bias_pct', 0)
    weekly_bias = diagnostics.get('weekly_bias_pct', 0)
    md += f"| Bias % | {monthly_bias:.1f}% | {weekly_bias:.1f}% |\n"

    # NMAE
    monthly_nmae = diagnostics.get('monthly_nmae', 0)
    weekly_nmae = diagnostics.get('weekly_nmae', 0)
    md += f"| NMAE | {monthly_nmae:.3f} | {weekly_nmae:.3f} |\n"

    # Alpha statistics
    md += "\n### 🎯 Alpha Statistics\n\n"
    alpha_mean = diagnostics.get('alpha_mean', 0)
    alpha_std = diagnostics.get('alpha_std', 0)
    alpha_cv = diagnostics.get('alpha_cv', 0)
    alpha_min = diagnostics.get('alpha_min', 0)
    alpha_max = diagnostics.get('alpha_max', 0)

    md += f"- **Mean**: {alpha_mean:.3f}\n"
    md += f"- **Std Dev**: {alpha_std:.3f}\n"
    md += f"- **CV**: {alpha_cv:.1%}\n"
    md += f"- **Range**: [{alpha_min:.3f}, {alpha_max:.3f}]\n"

    # Optimization info
    opt_info = diagnostics.get('optimization', {})
    if opt_info:
        md += "\n### ⚙️ Optimization\n\n"
        md += f"- **Status**: {opt_info.get('status', 'unknown')}\n"
        md += f"- **Converged**: {'✅ Yes' if opt_info.get('converged', False) else '❌ No'}\n"
        md += f"- **Residual Norm**: {opt_info.get('residual_norm', 0):.4f}\n"
        md += f"- **Smoothness λ**: {opt_info.get('lambda_smoothness', 0)}\n"

    return md


def format_detailed_metrics(diagnostics: dict) -> pd.DataFrame:
    """
    Format detailed metrics as DataFrame for gr.Dataframe.

    Args:
        diagnostics: Diagnostics dictionary from stitching result

    Returns:
        DataFrame with detailed metrics
    """
    metrics = []

    # Monthly metrics
    metrics.append({
        "Metric": "MAE",
        "Monthly": f"{diagnostics.get('monthly_mae', 0):.2f}",
        "Weekly": f"{diagnostics.get('weekly_mae', 0):.2f}",
        "Description": "Mean Absolute Error"
    })

    metrics.append({
        "Metric": "Correlation",
        "Monthly": f"{diagnostics.get('monthly_corr', 0):.3f}",
        "Weekly": f"{diagnostics.get('weekly_corr', 0):.3f}",
        "Description": "Pearson Correlation"
    })

    metrics.append({
        "Metric": "Bias %",
        "Monthly": f"{diagnostics.get('monthly_bias_pct', 0):.2f}%",
        "Weekly": f"{diagnostics.get('weekly_bias_pct', 0):.2f}%",
        "Description": "Mean Percentage Bias"
    })

    metrics.append({
        "Metric": "NMAE",
        "Monthly": f"{diagnostics.get('monthly_nmae', 0):.3f}",
        "Weekly": f"{diagnostics.get('weekly_nmae', 0):.3f}",
        "Description": "Normalized MAE"
    })

    # Alpha statistics
    metrics.append({
        "Metric": "Alpha Mean",
        "Monthly": f"{diagnostics.get('alpha_mean', 0):.3f}",
        "Weekly": "-",
        "Description": "Mean scaling factor"
    })

    metrics.append({
        "Metric": "Alpha CV",
        "Monthly": f"{diagnostics.get('alpha_cv', 0):.1%}",
        "Weekly": "-",
        "Description": "Coefficient of Variation"
    })

    return pd.DataFrame(metrics)


def create_download_package(
    stitched_daily: pd.DataFrame,
    diagnostics: dict,
    config: ConfigManager,
    ground_truth: dict,
    timeseries_fig: go.Figure,
    alpha_fig: go.Figure
) -> str:
    """
    Create ZIP file containing all results for download.

    Args:
        stitched_daily: Final stitched time series
        diagnostics: Diagnostics dictionary
        config: Configuration manager
        ground_truth: Dictionary with 'weekly' and 'monthly' DataFrames
        timeseries_fig: Plotly timeseries figure
        alpha_fig: Plotly alpha progression figure

    Returns:
        Path to created ZIP file
    """
    # Create temporary directory for files
    temp_dir = Path(tempfile.mkdtemp())

    # File prefix based on search term and geo
    prefix = f"{config.search_term.replace(' ', '_')}_{config.geo}"

    # 1. Save stitched daily data
    stitched_path = temp_dir / f"{prefix}_stitched_daily.csv"
    stitched_daily.to_csv(stitched_path, index=False)

    # 2. Save metadata JSON
    metadata = {
        "search_term": config.search_term,
        "geo": config.geo,
        "date_range": {
            "start": config.date_range.start,
            "end": config.date_range.end,
            "total_days": config.date_range.total_days
        },
        "configuration": {
            "overlap_days": config.daily.overlap_days,
            "weights": config.stitching.weights,
            "lambda_smoothness": config.stitching.smooth_alpha.lambda_smoothness if config.stitching.smooth_alpha else None,
            "max_iterations": config.stitching.max_iterations,
        },
        "diagnostics": {
            "monthly_mae": diagnostics.get('monthly_mae'),
            "monthly_corr": diagnostics.get('monthly_corr'),
            "monthly_bias_pct": diagnostics.get('monthly_bias_pct'),
            "weekly_mae": diagnostics.get('weekly_mae'),
            "weekly_corr": diagnostics.get('weekly_corr'),
            "weekly_bias_pct": diagnostics.get('weekly_bias_pct'),
            "alpha_mean": diagnostics.get('alpha_mean'),
            "alpha_cv": diagnostics.get('alpha_cv'),
            "num_chunks": diagnostics.get('num_chunks'),
            "optimization_status": diagnostics.get('optimization', {}).get('status'),
        }
    }

    metadata_path = temp_dir / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 3. Save validation report (text)
    report_path = temp_dir / f"{prefix}_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Google Trends Stitching Validation Report\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Search Term: {config.search_term}\n")
        f.write(f"Geographic Location: {config.geo}\n")
        f.write(f"Date Range: {config.date_range.start} to {config.date_range.end}\n")
        f.write(f"Total Days: {config.date_range.total_days}\n\n")

        f.write(f"Method: Smooth Alpha Stitching\n")
        f.write(f"Number of Chunks: {diagnostics.get('num_chunks')}\n\n")

        f.write(f"MONTHLY VALIDATION\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"MAE: {diagnostics.get('monthly_mae', 0):.2f}\n")
        f.write(f"Correlation: {diagnostics.get('monthly_corr', 0):.3f}\n")
        f.write(f"Bias %: {diagnostics.get('monthly_bias_pct', 0):.2f}%\n")
        f.write(f"NMAE: {diagnostics.get('monthly_nmae', 0):.3f}\n\n")

        f.write(f"WEEKLY VALIDATION\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"MAE: {diagnostics.get('weekly_mae', 0):.2f}\n")
        f.write(f"Correlation: {diagnostics.get('weekly_corr', 0):.3f}\n")
        f.write(f"Bias %: {diagnostics.get('weekly_bias_pct', 0):.2f}%\n")
        f.write(f"NMAE: {diagnostics.get('weekly_nmae', 0):.3f}\n\n")

        f.write(f"ALPHA STATISTICS\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"Mean: {diagnostics.get('alpha_mean', 0):.3f}\n")
        f.write(f"Std: {diagnostics.get('alpha_std', 0):.3f}\n")
        f.write(f"CV: {diagnostics.get('alpha_cv', 0):.1%}\n")
        f.write(f"Range: [{diagnostics.get('alpha_min', 0):.3f}, {diagnostics.get('alpha_max', 0):.3f}]\n\n")

        opt_info = diagnostics.get('optimization', {})
        f.write(f"OPTIMIZATION\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"Status: {opt_info.get('status', 'unknown')}\n")
        f.write(f"Converged: {'Yes' if opt_info.get('converged', False) else 'No'}\n")
        f.write(f"Residual Norm: {opt_info.get('residual_norm', 0):.4f}\n")

    # 4. Save plots as PNG
    timeseries_png_path = temp_dir / f"{prefix}_plot_timeseries.png"
    timeseries_fig.write_image(str(timeseries_png_path), width=1200, height=600)

    alpha_png_path = temp_dir / f"{prefix}_plot_alpha.png"
    alpha_fig.write_image(str(alpha_png_path), width=1000, height=500)

    # 5. Save ground truth data
    if ground_truth.get('weekly') is not None:
        weekly_path = temp_dir / f"{prefix}_weekly_ground_truth.csv"
        ground_truth['weekly'].to_csv(weekly_path, index=False)

    if ground_truth.get('monthly') is not None:
        monthly_path = temp_dir / f"{prefix}_monthly_ground_truth.csv"
        ground_truth['monthly'].to_csv(monthly_path, index=False)

    # Create ZIP file
    zip_path = temp_dir / f"{prefix}_results.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in temp_dir.glob('*'):
            if file_path != zip_path:
                zipf.write(file_path, file_path.name)

    return str(zip_path)
