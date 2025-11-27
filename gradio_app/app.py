"""
Gradio Web Application for Google Trends Daily Data Stitching.

This app allows users to enter a search term and receive stitched daily data
using the Smooth Alpha method (hierarchical optimization + smoothness penalty).
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import gradio as gr

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.ui_builder import create_input_components, create_output_components
from processing.pipeline import StitchingPipeline
from processing.config_builder import build_config
from utils.formatting import (
    format_metrics_markdown,
    format_detailed_metrics,
    create_download_package
)
from utils.plotting import plot_timeseries, plot_alpha_progression


def process_pipeline(search_term, start_date, end_date, geo, progress=gr.Progress()):
    """
    Main processing function that Gradio calls.

    Args:
        search_term: Search term input
        start_date: Start date (datetime)
        end_date: End date (datetime)
        geo: Geographic location code
        progress: Gradio progress tracker

    Returns:
        Tuple of (status, metrics_md, timeseries_plot, alpha_plot, detailed_df, download_file)
    """
    try:
        # Validation
        if not search_term or not search_term.strip():
            return (
                "❌ Error: Search term is required",
                "",
                None,
                None,
                None,
                None
            )

        # Calculate date difference
        date_diff = (end_date - start_date).days
        if date_diff < 270:
            return (
                f"❌ Error: Date range must be at least 270 days for stitching.\n"
                f"Current range: {date_diff} days\n\n"
                f"Please adjust your date range and try again.",
                "",
                None,
                None,
                None,
                None
            )

        # Get API key from environment
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            return (
                "❌ Error: SERPAPI_KEY not configured in environment.\n\n"
                "This app requires a valid SerpAPI key to function.",
                "",
                None,
                None,
                None,
                None
            )

        # Build config
        progress(0, desc="🔧 Initializing configuration...")
        config = build_config(
            search_term=search_term.strip(),
            geo=geo,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        # Create pipeline
        pipeline = StitchingPipeline(api_key)

        # Progress callback
        def update_progress(step, total, message):
            progress((step / total), desc=message)

        # Execute pipeline
        progress(0.1, desc="📊 Fetching Google Trends data...")
        result = pipeline.run(config, progress_callback=update_progress)

        # Format outputs
        progress(0.9, desc="📈 Generating visualizations...")

        num_days = len(result.stitched_daily)
        monthly_mae = result.diagnostics.get('monthly_mae', 0)
        weekly_mae = result.diagnostics.get('weekly_mae', 0)
        monthly_corr = result.diagnostics.get('monthly_corr', 0)

        status_msg = (
            f"✅ Pipeline completed successfully!\n\n"
            f"**Processed**: {num_days} days of data\n"
            f"**Search Term**: {search_term}\n"
            f"**Location**: {geo}\n"
            f"**Date Range**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
            f"**Quality Metrics**:\n"
            f"- Monthly MAE: {monthly_mae:.2f}\n"
            f"- Weekly MAE: {weekly_mae:.2f}\n"
            f"- Monthly Correlation: {monthly_corr:.3f}\n"
        )

        metrics_md = format_metrics_markdown(result.diagnostics)

        timeseries_fig = plot_timeseries(
            result.stitched_daily,
            result.ground_truth.get('weekly'),
            result.ground_truth.get('monthly'),
            search_term
        )

        alpha_fig = plot_alpha_progression(result.diagnostics['alpha_values'])

        detailed_df = format_detailed_metrics(result.diagnostics)

        # Create download package
        progress(0.95, desc="📦 Packaging results...")
        download_path = create_download_package(
            result.stitched_daily,
            result.diagnostics,
            config,
            result.ground_truth,
            timeseries_fig,
            alpha_fig
        )

        progress(1.0, desc="✨ Complete!")

        return (
            status_msg,
            metrics_md,
            timeseries_fig,
            alpha_fig,
            detailed_df,
            download_path
        )

    except Exception as e:
        error_msg = (
            f"❌ Pipeline failed with error:\n\n"
            f"```\n{str(e)}\n```\n\n"
            f"**Common issues:**\n"
            f"- Search term has insufficient search interest\n"
            f"- API rate limit exceeded (wait a few minutes)\n"
            f"- Network connectivity issues\n\n"
            f"Please try again or contact support if the issue persists."
        )

        return (
            error_msg,
            "",
            None,
            None,
            None,
            None
        )


# Build Gradio interface
with gr.Blocks(
    title="Google Trends Daily Data Stitcher"
) as app:
    gr.Markdown("""
    # 📈 Google Trends Daily Data Stitcher

    Get high-quality daily Google Trends data for extended time periods using
    **hierarchical constrained optimization** with **smoothness penalty**.

    ### How it works
    1. **Fetches** overlapping daily chunks from Google Trends
    2. **Stitches** them using the Smooth Alpha optimization method
    3. **Validates** against independent weekly/monthly ground truth
    4. **Delivers** downloadable package with data, plots, and metrics

    ---
    """)

    with gr.Row():
        # Left column: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")

            search_term, start_date, end_date, geo, submit_btn = create_input_components()

            gr.Markdown("""
            ---
            **Requirements:**
            - Date range must be ≥ 270 days
            - Search term should have sufficient search interest
            - Valid SerpAPI key required

            **Expected Performance:**
            - MAE < 0.5
            - Correlation > 0.95
            - Processing time: 1-2 minutes
            """)

        # Right column: Outputs
        with gr.Column(scale=2):
            status, metrics_md, timeseries_plot, alpha_plot, detailed_df, download_file = create_output_components()

    # Connect button to processing function
    submit_btn.click(
        fn=process_pipeline,
        inputs=[search_term, start_date, end_date, geo],
        outputs=[status, metrics_md, timeseries_plot, alpha_plot, detailed_df, download_file]
    )

    gr.Markdown("""
    ---
    ### 📚 About This Tool

    This tool uses the **Smooth Alpha** stitching method, which extends hierarchical
    constrained optimization with a smoothness penalty to reduce alpha volatility.

    **Method Details:**
    - Minimize: `||A·α - b||² + λ·||D·α||²`
    - Constraints: Monthly + Weekly + Overlap consistency
    - Solver: Convex optimization (cvxpy)

    **Typical Performance:**
    - MAE: 0.3-0.5
    - Correlation: 0.95-0.97
    - Alpha CV: < 20%

    **Download Package Contains:**
    - `stitched_daily.csv` - Final daily time series
    - `metadata.json` - Configuration and diagnostics
    - `validation_report.txt` - Human-readable metrics
    - `plot_timeseries.png` - Time series visualization
    - `plot_alpha.png` - Alpha progression chart
    - `weekly_ground_truth.csv` - Weekly validation data
    - `monthly_ground_truth.csv` - Monthly validation data

    **GitHub Repository:** [google-trends-daily-data-preprocessing](https://github.com/yourusername/google-trends-daily-data-preprocessing)

    ---
    <small>Powered by SerpAPI | Built with Gradio | Deployed on Hugging Face Spaces</small>
    """)


if __name__ == "__main__":
    app.launch(
        theme=gr.themes.Soft(),
        css="""
            .gradio-container {max-width: 1400px !important}
            #status-box {font-family: monospace; font-size: 14px}
        """
    )
