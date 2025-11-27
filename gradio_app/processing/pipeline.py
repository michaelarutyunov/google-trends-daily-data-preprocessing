"""
Pipeline orchestrator for Gradio app.
Coordinates data fetching, stitching, and validation.
"""

import sys
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import pandas as pd

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api import TrendsAPI, TrendsAPIError
from src.stitching.smooth_alpha import SmoothAlphaStitcher
from src.config import ConfigManager


@dataclass
class PipelineResult:
    """Results from pipeline execution."""
    stitched_daily: pd.DataFrame
    diagnostics: dict
    ground_truth: dict  # {'weekly': df, 'monthly': df}


class StitchingPipeline:
    """
    Orchestrates the complete stitching pipeline.

    Workflow:
    1. Fetch daily chunks from Google Trends
    2. Fetch weekly and monthly ground truth
    3. Run SmoothAlphaStitcher
    4. Return results with diagnostics
    """

    def __init__(self, api_key: str):
        """
        Initialize pipeline.

        Args:
            api_key: SerpAPI key
        """
        self.api = TrendsAPI(
            api_key=api_key,
            timeout=30,
            max_retries=3,
            retry_delay=2
        )

    def run(
        self,
        config: ConfigManager,
        progress_callback: Optional[Callable] = None
    ) -> PipelineResult:
        """
        Execute full stitching pipeline.

        Args:
            config: Configuration manager
            progress_callback: Optional callback(step, total, message)

        Returns:
            PipelineResult with stitched data and diagnostics

        Raises:
            TrendsAPIError: If data fetching fails
            Exception: If stitching or validation fails
        """
        total_steps = 4

        # Step 1: Fetch daily chunks
        if progress_callback:
            progress_callback(1, total_steps, "Fetching daily chunks...")

        daily_chunks = self.api.fetch_daily_chunks(
            search_term=config.search_term,
            start_date=config.date_range.start,
            end_date=config.date_range.end,
            max_chunk_days=266,
            overlap_days=config.daily.overlap_days,
            geo=config.geo
        )

        if not daily_chunks or all(chunk.empty for chunk in daily_chunks):
            raise ValueError(
                f"No daily data found for '{config.search_term}'. "
                f"The search term may have insufficient search interest."
            )

        # Step 2: Fetch ground truth data
        if progress_callback:
            progress_callback(2, total_steps, "Fetching ground truth data...")

        # Fetch monthly data (historical)
        monthly_data = self.api.fetch_historical_monthly(
            search_term=config.search_term,
            start_year=config.date_range.start_date.year,
            end_year=config.date_range.end_date.year,
            geo=config.geo
        )

        # Fetch weekly data (for validation)
        weekly_data = self.api.fetch_weekly(
            search_term=config.search_term,
            start_date=config.date_range.start,
            end_date=config.date_range.end,
            geo=config.geo
        )

        if monthly_data.empty:
            raise ValueError(
                f"No monthly ground truth data found for '{config.search_term}'"
            )

        # Step 3: Run stitching
        if progress_callback:
            progress_callback(3, total_steps, "Running Smooth Alpha optimization...")

        stitcher = SmoothAlphaStitcher()
        result = stitcher.stitch(
            daily_chunks=daily_chunks,
            monthly_data=monthly_data,
            weekly_data=weekly_data if not weekly_data.empty else None,
            config=config
        )

        # Step 4: Package results
        if progress_callback:
            progress_callback(4, total_steps, "Packaging results...")

        return PipelineResult(
            stitched_daily=result.stitched_series,
            diagnostics=result.diagnostics,
            ground_truth={
                'weekly': weekly_data if not weekly_data.empty else None,
                'monthly': monthly_data
            }
        )
