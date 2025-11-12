"""
Baseline stitching method: Simple monthly mean scaling.

This method provides a simple baseline by:
1. Averaging overlapping daily chunks to get raw daily series
2. Computing monthly sums from the raw daily data
3. Calculating scaling factors (alpha) as ratios: ground_truth_monthly / raw_monthly
4. Scaling each day in a month by the corresponding monthly alpha

This is fast but doesn't leverage weekly data or enforce overlap continuity.
"""

from typing import List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from .base import StitchingEngine, StitchingResult


class BaselineStitcher(StitchingEngine):
    """
    Baseline stitching method using simple monthly mean scaling.

    Algorithm:
    1. Average overlapping values from daily chunks
    2. Aggregate averaged daily data to monthly
    3. Calculate alpha = ground_truth_monthly / averaged_monthly
    4. Scale daily values by monthly alpha
    """

    def name(self) -> str:
        """Return method name."""
        return "baseline"

    def stitch(
        self,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> StitchingResult:
        """
        Stitch daily chunks using baseline monthly scaling.

        Args:
            daily_chunks: List of daily chunk DataFrames [date, value]
            monthly_data: Monthly ground truth [date, value]
            weekly_data: Weekly ground truth [date, value] (not used in baseline)
            config: Configuration object

        Returns:
            StitchingResult with stitched series and diagnostics
        """
        logger.info(f"Starting {self.name()} stitching method")

        # Validate inputs
        self._validate_inputs(daily_chunks, monthly_data, weekly_data)

        # Step 1: Average overlapping chunks
        logger.info("Step 1: Averaging overlapping daily chunks")
        raw_daily = self._average_overlaps(daily_chunks)

        # Step 2: Aggregate to monthly
        logger.info("Step 2: Aggregating daily data to monthly")
        raw_monthly = self._aggregate_to_monthly(raw_daily)

        # Step 3: Calculate monthly scaling factors (alpha)
        logger.info("Step 3: Computing monthly scaling factors")
        monthly_alphas = self._compute_monthly_alphas(
            raw_monthly,
            monthly_data,
            config
        )

        # Step 4: Scale daily values by monthly alpha
        logger.info("Step 4: Scaling daily values")
        stitched_daily = self._scale_daily_by_monthly(
            raw_daily,
            monthly_alphas
        )

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            stitched_daily,
            monthly_data,
            weekly_data,
            monthly_alphas,
            raw_daily,
        )

        logger.success(f"{self.name()} stitching completed")

        return StitchingResult(
            stitched_series=stitched_daily,
            alpha_estimates=monthly_alphas["alpha"].values,
            diagnostics=diagnostics,
            method_name=self.name(),
        )

    def _average_overlaps(self, daily_chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Average values in overlap regions across chunks.

        Args:
            daily_chunks: List of daily chunks

        Returns:
            DataFrame with averaged daily values [date, value]
        """
        all_data = []

        for i, chunk in enumerate(daily_chunks):
            if chunk.empty:
                logger.warning(f"Chunk {i} is empty, skipping")
                continue

            chunk_copy = chunk.copy()
            chunk_copy["date"] = pd.to_datetime(chunk_copy["date"])
            all_data.append(chunk_copy[["date", "value"]])

        # Concatenate and group by date to average overlaps
        combined = pd.concat(all_data, ignore_index=True)
        averaged = combined.groupby("date", as_index=False)["value"].mean()
        averaged = averaged.sort_values("date").reset_index(drop=True)

        logger.info(
            f"Averaged {len(combined)} total observations to {len(averaged)} unique dates"
        )

        return averaged

    def _compute_monthly_alphas(
        self,
        raw_monthly: pd.DataFrame,
        ground_truth_monthly: pd.DataFrame,
        config: Any,
    ) -> pd.DataFrame:
        """
        Calculate monthly scaling factors: alpha = ground_truth / raw.

        Args:
            raw_monthly: Raw monthly sums from averaged daily data
            ground_truth_monthly: Ground truth monthly data
            config: Configuration object

        Returns:
            DataFrame with [date, alpha, raw_sum, ground_truth]
        """
        # Merge on month
        raw_monthly = raw_monthly.copy()
        ground_truth_monthly = ground_truth_monthly.copy()

        raw_monthly["date"] = pd.to_datetime(raw_monthly["date"])
        ground_truth_monthly["date"] = pd.to_datetime(ground_truth_monthly["date"])

        # Extract month for matching
        raw_monthly["month"] = raw_monthly["date"].dt.to_period("M")
        ground_truth_monthly["month"] = ground_truth_monthly["date"].dt.to_period("M")

        # Merge
        merged = pd.merge(
            ground_truth_monthly[["month", "value"]],
            raw_monthly[["month", "value"]],
            on="month",
            how="inner",
            suffixes=("_truth", "_raw"),
        )

        # Calculate alpha with zero handling
        zero_threshold = config.stitching.zero_threshold if hasattr(config, "stitching") else 0.01

        # Handle structural zeros
        structural_zero_months = []
        if hasattr(config, "stitching") and hasattr(config.stitching, "structural_zero_months"):
            structural_zero_months = config.stitching.structural_zero_months

        merged["is_structural_zero"] = merged["month"].apply(
            lambda m: m.month in structural_zero_months
        )

        # Calculate alpha
        def compute_alpha(row):
            if row["is_structural_zero"] and row["value_truth"] < zero_threshold:
                # Structural zero: force alpha = 1 to maintain zero
                return 1.0
            elif abs(row["value_raw"]) < zero_threshold:
                # Avoid division by zero
                if abs(row["value_truth"]) < zero_threshold:
                    return 1.0
                else:
                    logger.warning(
                        f"Month {row['month']}: raw value near zero but truth is not. Using alpha=1"
                    )
                    return 1.0
            else:
                return row["value_truth"] / row["value_raw"]

        merged["alpha"] = merged.apply(compute_alpha, axis=1)

        # Convert month back to timestamp for output
        merged["date"] = merged["month"].dt.to_timestamp()

        # Log alpha statistics
        valid_alphas = merged[~merged["is_structural_zero"]]["alpha"]
        logger.info(
            f"Monthly alpha statistics (excluding structural zeros): "
            f"mean={valid_alphas.mean():.3f}, "
            f"std={valid_alphas.std():.3f}, "
            f"min={valid_alphas.min():.3f}, "
            f"max={valid_alphas.max():.3f}"
        )

        # Flag outlier alphas
        outliers = merged[(merged["alpha"] < 0.1) | (merged["alpha"] > 10)]
        if not outliers.empty:
            logger.warning(
                f"Found {len(outliers)} months with extreme alpha values (< 0.1 or > 10). "
                "This may indicate Google Trends rebase or data quality issues."
            )

        return merged[["date", "month", "alpha", "value_raw", "value_truth", "is_structural_zero"]]

    def _scale_daily_by_monthly(
        self,
        daily: pd.DataFrame,
        monthly_alphas: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scale daily values by monthly alpha factors.

        Args:
            daily: Raw daily data [date, value]
            monthly_alphas: Monthly scaling factors [date, month, alpha, ...]

        Returns:
            Scaled daily data [date, value]
        """
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["month"] = daily["date"].dt.to_period("M")

        # Merge alpha values
        alpha_lookup = monthly_alphas.set_index("month")["alpha"].to_dict()
        daily["alpha"] = daily["month"].map(alpha_lookup)

        # Handle missing alphas (should not happen if data is aligned)
        missing_alphas = daily["alpha"].isna()
        if missing_alphas.any():
            logger.warning(
                f"Found {missing_alphas.sum()} days with missing alpha values. "
                "Using alpha=1 for these days."
            )
            daily.loc[missing_alphas, "alpha"] = 1.0

        # Scale values
        daily["value"] = daily["value"] * daily["alpha"]

        # Return clean DataFrame
        return daily[["date", "value"]].reset_index(drop=True)

    def _calculate_diagnostics(
        self,
        stitched_daily: pd.DataFrame,
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        monthly_alphas: pd.DataFrame,
        raw_daily: pd.DataFrame,
    ) -> dict:
        """
        Calculate diagnostic metrics for the stitching result.

        Args:
            stitched_daily: Final stitched daily series
            monthly_data: Ground truth monthly data
            weekly_data: Ground truth weekly data (optional)
            monthly_alphas: Monthly scaling factors
            raw_daily: Raw daily data before scaling

        Returns:
            Dictionary of diagnostic metrics
        """
        diagnostics = {}

        # Monthly error metrics
        # WARNING: These metrics use circular validation (training data = test data)
        # because monthly ground truth is used to compute alpha scaling factors
        stitched_monthly = self._aggregate_to_monthly(stitched_daily)
        monthly_comparison = pd.merge(
            monthly_data.rename(columns={"value": "truth"}),
            stitched_monthly.rename(columns={"value": "stitched"}),
            on="date",
            how="inner",
        )

        if not monthly_comparison.empty:
            monthly_comparison["error"] = monthly_comparison["stitched"] - monthly_comparison["truth"]
            monthly_comparison["abs_error"] = abs(monthly_comparison["error"])

            # Calculate all monthly metrics using centralized method
            monthly_metrics = self._calculate_comparison_metrics(monthly_comparison, "monthly")
            diagnostics.update(monthly_metrics)

            diagnostics["monthly_comparison"] = monthly_comparison

            # Flag circular validation issue
            diagnostics["monthly_validation_warning"] = "CIRCULAR_VALIDATION: Monthly metrics use training data (α = truth/raw)"

            logger.warning(
                "Baseline monthly MAE/RMSE use circular validation (training data = test data). "
                "Use weekly MAE for independent validation."
            )

        # Weekly error metrics (if available)
        logger.debug(f"Checking weekly data: is None? {weekly_data is None}, is empty? {weekly_data.empty if weekly_data is not None else 'N/A'}")
        if weekly_data is not None and not weekly_data.empty:
            logger.info("Calculating weekly validation metrics...")
            stitched_weekly = self._aggregate_to_weekly(stitched_daily)
            logger.debug(f"Stitched weekly: {len(stitched_weekly)} observations")
            weekly_comparison = pd.merge(
                weekly_data.rename(columns={"value": "truth"}),
                stitched_weekly.rename(columns={"value": "stitched"}),
                on="date",
                how="inner",
            )
            logger.debug(f"Weekly comparison after merge: {len(weekly_comparison)} observations")

            if not weekly_comparison.empty:
                logger.debug("Weekly comparison is not empty, calculating metrics...")
                weekly_comparison["error"] = weekly_comparison["stitched"] - weekly_comparison["truth"]
                weekly_comparison["abs_error"] = abs(weekly_comparison["error"])

                # Calculate all weekly metrics using centralized method
                weekly_metrics = self._calculate_comparison_metrics(weekly_comparison, "weekly")
                diagnostics.update(weekly_metrics)

                diagnostics["weekly_comparison"] = weekly_comparison

        # Alpha statistics
        valid_alphas = monthly_alphas[~monthly_alphas["is_structural_zero"]]["alpha"]
        diagnostics["alpha_mean"] = valid_alphas.mean()
        diagnostics["alpha_std"] = valid_alphas.std()
        diagnostics["alpha_min"] = valid_alphas.min()
        diagnostics["alpha_max"] = valid_alphas.max()
        diagnostics["alpha_cv"] = valid_alphas.std() / valid_alphas.mean() if valid_alphas.mean() != 0 else 0

        # Method metadata
        diagnostics["method"] = self.name()
        diagnostics["num_months"] = len(monthly_alphas)

        # Log summary
        log_msg = (
            f"Diagnostics: "
            f"Monthly MAE={diagnostics.get('monthly_mae', 0):.2f} (circular validation ⚠), "
            f"Alpha CV={diagnostics['alpha_cv']:.3f}"
        )

        # Add weekly metrics if available (this is the meaningful metric!)
        if 'weekly_mae' in diagnostics:
            log_msg += (
                f" | Weekly MAE={diagnostics['weekly_mae']:.2f}, "
                f"Corr={diagnostics.get('weekly_corr', 0):.3f}, "
                f"Bias%={diagnostics.get('weekly_bias_pct', 0):.1f}% (independent ✓)"
            )

        logger.info(log_msg)

        # Log R² deprecation notice
        if 'monthly_r2' in diagnostics or 'weekly_r2' in diagnostics:
            logger.debug(
                "Note: R² values are included in diagnostics for backwards compatibility "
                "but are deprecated. Use correlation, nmae, and bias metrics instead."
            )

        return diagnostics
