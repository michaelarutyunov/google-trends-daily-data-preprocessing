"""
Gradio UI component builders.
Creates input and output components for the interface.
"""

from datetime import datetime, timedelta
import gradio as gr


def create_input_components():
    """
    Create Gradio input components for user configuration.

    Returns:
        Tuple of (search_term, start_date, end_date, geo, submit_btn)
    """
    search_term = gr.Textbox(
        label="Search Term",
        placeholder="e.g., chatgpt, bitcoin, climate change",
        info="Enter any Google Trends search term",
        value=""
    )

    # Calculate default dates (2 years ago to today)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=730)

    start_date = gr.DateTime(
        label="Start Date",
        value=default_start,
        info="Must be at least 270 days before end date"
    )

    end_date = gr.DateTime(
        label="End Date",
        value=default_end,
        info="End of date range"
    )

    geo = gr.Dropdown(
        label="Geographic Location",
        choices=[
            ("United States", "US"),
            ("United Kingdom", "GB"),
            ("Canada", "CA"),
            ("Australia", "AU"),
            ("Germany", "DE"),
            ("France", "FR"),
            ("Japan", "JP"),
            ("India", "IN"),
        ],
        value="US",
        info="Country code for Google Trends data"
    )

    submit_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

    return search_term, start_date, end_date, geo, submit_btn


def create_output_components():
    """
    Create Gradio output components for displaying results.

    Returns:
        Tuple of (status_text, metrics_text, timeseries_plot, alpha_plot, detailed_metrics, download_file)
    """
    status_text = gr.Textbox(
        label="Status",
        value="Ready to process... Enter a search term and click 'Run Pipeline'",
        interactive=False,
        lines=3
    )

    metrics_text = gr.Markdown(
        label="Key Metrics",
        value=""
    )

    timeseries_plot = gr.Plot(
        label="Time Series: Stitched vs Ground Truth",
        visible=True
    )

    alpha_plot = gr.Plot(
        label="Alpha Progression (Scaling Factors)",
        visible=True
    )

    detailed_metrics = gr.Dataframe(
        label="Detailed Metrics",
        visible=True
    )

    download_file = gr.File(
        label="Download Results Package",
        visible=True
    )

    return status_text, metrics_text, timeseries_plot, alpha_plot, detailed_metrics, download_file
