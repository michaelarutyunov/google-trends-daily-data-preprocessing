"""
Config builder for Gradio app.
Creates ConfigManager instances programmatically from user inputs.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    ConfigManager,
    DateRange,
    DailyConfig,
    StitchingConfig,
    SmoothAlphaConfig,
    SerpAPIConfig,
    OutputConfig,
    RobustnessConfig,
    TemporalCVConfig,
)


def build_config(
    search_term: str,
    geo: str,
    start_date: str,
    end_date: str
) -> ConfigManager:
    """
    Build ConfigManager programmatically from user inputs.

    Args:
        search_term: Google Trends search term
        geo: Geographic location code (e.g., 'US', 'GB')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        ConfigManager instance with best-practice defaults

    Best-practice defaults:
    - overlap_days: 60
    - weights: {monthly: 1.0, weekly: 0.5, overlap: 0.1}
    - lambda_smoothness: 0.1 (Smooth Alpha method)
    - max_iterations: 1000
    - tolerance: 1e-8
    - zero_threshold: 0.01
    """
    # Create a minimal raw config dictionary
    raw_config = {
        "search_term": search_term,
        "geo": geo,
        "random_seed": 42,
        "date_range": {
            "start": start_date,
            "end": end_date,
        },
        "daily": {
            "overlap_days": 60,
        },
        "stitching": {
            "weights": {
                "monthly": 1.0,
                "weekly": 0.5,
                "overlap": 0.1,
            },
            "zero_threshold": 0.01,
            "structural_zero_months": [],
            "max_iterations": 1000,
            "tolerance": 1.0e-8,
            "smooth_alpha": {
                "lambda_smoothness": 0.1,
            },
        },
        "serpapi": {
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 2,
        },
        "output": {
            "save_plots": True,
            "plot_format": "png",
            "plot_dpi": 300,
        },
        "robustness": {
            "overlap_tests": [60],
            "weight_tests": [[1.0, 0.5]],
            "temporal_cv": {
                "train_months": 24,
                "test_months": 6,
                "gap_months": 3,
            },
        },
    }

    # Create ConfigManager instance
    # We'll need to create a custom instance since ConfigManager expects a file path
    # Instead, we'll manually set the attributes after initialization
    config = _create_config_from_dict(raw_config)

    return config


def _create_config_from_dict(raw_config: dict) -> ConfigManager:
    """
    Create ConfigManager instance from dictionary without requiring a YAML file.

    Args:
        raw_config: Configuration dictionary

    Returns:
        ConfigManager instance
    """
    # Create a mock ConfigManager instance
    # We need to bypass the __init__ method that requires a file path
    config = object.__new__(ConfigManager)

    # Set the raw_config
    config.raw_config = raw_config

    # Parse basic parameters
    config.search_term = raw_config.get("search_term", "")
    if not config.search_term:
        raise ValueError("search_term is required")

    config.geo = raw_config.get("geo", "US")
    config.random_seed = raw_config.get("random_seed", 42)

    # Date range
    date_range_raw = raw_config.get("date_range", {})
    config.date_range = DateRange(
        start=date_range_raw.get("start", ""),
        end=date_range_raw.get("end", ""),
    )

    # Daily chunk config
    daily_raw = raw_config.get("daily", {})
    config.daily = DailyConfig(overlap_days=daily_raw.get("overlap_days", 60))

    # Stitching config
    stitching_raw = raw_config.get("stitching", {})

    # Parse nested smooth_alpha config
    smooth_alpha_raw = stitching_raw.get("smooth_alpha", {})
    smooth_alpha_config = None
    if smooth_alpha_raw:
        smooth_alpha_config = SmoothAlphaConfig(
            lambda_smoothness=smooth_alpha_raw.get("lambda_smoothness", 0.1)
        )

    config.stitching = StitchingConfig(
        weights=stitching_raw.get("weights", {}),
        zero_threshold=stitching_raw.get("zero_threshold", 0.01),
        structural_zero_months=stitching_raw.get("structural_zero_months", []),
        max_iterations=stitching_raw.get("max_iterations", 1000),
        tolerance=stitching_raw.get("tolerance", 1.0e-8),
        smooth_alpha=smooth_alpha_config,
    )

    # SerpAPI config
    serpapi_raw = raw_config.get("serpapi", {})
    config.serpapi = SerpAPIConfig(
        timeout=serpapi_raw.get("timeout", 30),
        max_retries=serpapi_raw.get("max_retries", 3),
        retry_delay=serpapi_raw.get("retry_delay", 2),
    )

    # Output config
    output_raw = raw_config.get("output", {})
    config.output = OutputConfig(
        save_plots=output_raw.get("save_plots", True),
        plot_format=output_raw.get("plot_format", "png"),
        plot_dpi=output_raw.get("plot_dpi", 300),
    )

    # Robustness testing config
    robustness_raw = raw_config.get("robustness", {})
    temporal_cv_raw = robustness_raw.get("temporal_cv", {})
    config.robustness = RobustnessConfig(
        overlap_tests=robustness_raw.get("overlap_tests", [60]),
        weight_tests=robustness_raw.get("weight_tests", [[1.0, 0.5]]),
        temporal_cv=TemporalCVConfig(
            train_months=temporal_cv_raw.get("train_months", 24),
            test_months=temporal_cv_raw.get("test_months", 6),
            gap_months=temporal_cv_raw.get("gap_months", 3),
        ),
    )

    return config
