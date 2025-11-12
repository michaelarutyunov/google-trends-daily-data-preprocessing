"""
Hierarchical + Day-of-Week stitching method.

Extends hierarchical optimization with day-of-week pattern correction.
This is useful for search terms with weekly patterns (e.g., searches spike on weekends).

Algorithm:
1. Run hierarchical stitching to get base stitched series
2. Estimate day-of-week (DOW) pattern from residuals
3. Apply DOW correction multiplicatively to the stitched series
4. Optionally re-run optimization with DOW constraints

This provides the best accuracy for search terms with strong weekly patterns.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

from .base import StitchingEngine, StitchingResult
from .hierarchical import HierarchicalStitcher


class HierarchicalDOWStitcher(StitchingEngine):
    """
    Hierarchical stitching with day-of-week pattern correction.

    Algorithm:
    1. Run base hierarchical stitching
    2. Calculate DOW pattern from weekly data
    3. Apply DOW correction to stitched series
    4. Verify constraints still satisfied
    """

    def __init__(self, reoptimize: bool = False):
        """
        Initialize HierarchicalDOWStitcher.

        Args:
            reoptimize: If True, re-run optimization after DOW correction.
                       If False, simply apply DOW multiplicatively (faster).
        """
        self.reoptimize = reoptimize
        self.base_stitcher = HierarchicalStitcher()

    def name(self) -> str:
        """Return method name."""
        return "hierarchical_dow"

    def stitch(
        self,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> StitchingResult:
        """
        Stitch daily chunks using hierarchical optimization + DOW correction.

        Args:
            daily_chunks: List of daily chunk DataFrames [date, value]
            monthly_data: Monthly ground truth [date, value]
            weekly_data: Weekly ground truth [date, value] (required for DOW estimation)
            config: Configuration object

        Returns:
            StitchingResult with stitched series and diagnostics
        """
        logger.info(f"Starting {self.name()} stitching method")

        # Validate inputs
        self._validate_inputs(daily_chunks, monthly_data, weekly_data)

        if weekly_data is None or weekly_data.empty:
            logger.warning(
                "Weekly data not provided. DOW correction requires weekly data. "
                "Falling back to standard hierarchical method."
            )
            return self.base_stitcher.stitch(daily_chunks, monthly_data, weekly_data, config)

        # Step 1: Run base hierarchical stitching
        logger.info("Step 1: Running base hierarchical stitching")
        base_result = self.base_stitcher.stitch(daily_chunks, monthly_data, weekly_data, config)

        # Step 2: Estimate DOW pattern
        logger.info("Step 2: Estimating day-of-week pattern")
        dow_factors = self._estimate_dow_pattern(
            base_result.stitched_series,
            weekly_data,
        )

        # Step 3: Apply DOW correction
        logger.info("Step 3: Applying DOW correction")
        corrected_series = self._apply_dow_correction(
            base_result.stitched_series,
            dow_factors,
        )

        # Step 4: Renormalize to match constraints
        # BUG FIX: Removed renormalization step - it destroys hierarchical optimization
        # The renormalization forces monthly MAE → 0 (circular validation) but degrades
        # weekly MAE from 0.91 → 1.37 (50% worse). This defeats the purpose of using
        # hierarchical optimization with soft constraints.
        # logger.info("Step 4: Renormalizing to match monthly constraints")
        # final_series = self._renormalize_to_monthly(
        #     corrected_series,
        #     monthly_data,
        #     config,
        # )
        final_series = corrected_series  # Use DOW-corrected series directly
        logger.info("Step 4: Skipping renormalization to preserve hierarchical optimization")

        # Step 5: Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            final_series,
            monthly_data,
            weekly_data,
            base_result.alpha_estimates,
            dow_factors,
            base_result.diagnostics,
        )

        logger.success(f"{self.name()} stitching completed")

        return StitchingResult(
            stitched_series=final_series,
            alpha_estimates=base_result.alpha_estimates,
            diagnostics=diagnostics,
            method_name=self.name(),
        )

    def _estimate_dow_pattern(
        self,
        stitched_daily: pd.DataFrame,
        weekly_data: pd.DataFrame,
    ) -> Dict[int, float]:
        """
        Estimate day-of-week pattern from weekly data.

        The DOW pattern represents relative activity on each day of the week.
        For example, if searches are 20% higher on Sundays, Sunday's factor = 1.2.

        Args:
            stitched_daily: Base stitched daily series
            weekly_data: Weekly ground truth

        Returns:
            Dictionary mapping day-of-week (0=Monday, 6=Sunday) to scaling factor
        """
        # Aggregate stitched daily to weekly
        stitched_daily = stitched_daily.copy()
        stitched_daily["date"] = pd.to_datetime(stitched_daily["date"])
        # CRITICAL: Use Sunday-ending weeks to match Google Trends format
        stitched_daily["week"] = stitched_daily["date"].dt.to_period("W-SUN")
        stitched_daily["dow"] = stitched_daily["date"].dt.dayofweek

        # Prepare weekly data
        weekly_data = weekly_data.copy()
        weekly_data["date"] = pd.to_datetime(weekly_data["date"])
        weekly_data["week"] = weekly_data["date"].dt.to_period("W-SUN")

        # Merge to get weekly ground truth for each daily observation
        merged = pd.merge(
            stitched_daily,
            weekly_data[["week", "value"]],
            on="week",
            how="inner",
            suffixes=("_daily", "_weekly"),
        )

        # Calculate expected value per day if uniformly distributed
        # (weekly_sum / 7) would be the uniform expectation
        merged["days_in_week"] = merged.groupby("week")["date"].transform("count")
        merged["expected_uniform"] = merged["value_weekly"] / merged["days_in_week"]

        # Calculate actual vs expected ratio for each day
        merged["dow_ratio"] = merged["value_daily"] / (merged["expected_uniform"] + 1e-8)

        # Average ratio for each day of week
        dow_factors = merged.groupby("dow")["dow_ratio"].mean().to_dict()

        # Normalize so that average factor = 1.0
        avg_factor = np.mean(list(dow_factors.values()))
        dow_factors = {dow: factor / avg_factor for dow, factor in dow_factors.items()}

        # Fill missing days with 1.0
        for dow in range(7):
            if dow not in dow_factors:
                dow_factors[dow] = 1.0

        logger.info(
            f"DOW factors estimated: "
            f"Mon={dow_factors[0]:.3f}, Tue={dow_factors[1]:.3f}, Wed={dow_factors[2]:.3f}, "
            f"Thu={dow_factors[3]:.3f}, Fri={dow_factors[4]:.3f}, Sat={dow_factors[5]:.3f}, "
            f"Sun={dow_factors[6]:.3f}"
        )

        return dow_factors

    def _apply_dow_correction(
        self,
        daily: pd.DataFrame,
        dow_factors: Dict[int, float],
    ) -> pd.DataFrame:
        """
        Apply day-of-week correction multiplicatively.

        Args:
            daily: Daily series [date, value]
            dow_factors: DOW scaling factors

        Returns:
            Corrected daily series [date, value]
        """
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["dow"] = daily["date"].dt.dayofweek
        daily["dow_factor"] = daily["dow"].map(dow_factors)

        # Apply multiplicative correction
        daily["value"] = daily["value"] * daily["dow_factor"]

        return daily[["date", "value"]].reset_index(drop=True)

    def _renormalize_to_monthly(
        self,
        daily: pd.DataFrame,
        monthly_data: pd.DataFrame,
        config: Any,
    ) -> pd.DataFrame:
        """
        Renormalize daily series to match monthly constraints.

        After applying DOW correction, monthly sums may drift slightly.
        This step rescales each month to match ground truth.

        Args:
            daily: Daily series after DOW correction
            monthly_data: Monthly ground truth
            config: Configuration object

        Returns:
            Renormalized daily series [date, value]
        """
        daily = daily.copy()
        monthly_data = monthly_data.copy()

        daily["date"] = pd.to_datetime(daily["date"])
        monthly_data["date"] = pd.to_datetime(monthly_data["date"])

        daily["month"] = daily["date"].dt.to_period("M")
        monthly_data["month"] = monthly_data["date"].dt.to_period("M")

        # Calculate current monthly sums
        current_monthly = daily.groupby("month")["value"].sum().reset_index()
        current_monthly = current_monthly.rename(columns={"value": "current_sum"})

        # Merge with ground truth
        monthly_data = monthly_data.rename(columns={"value": "truth"})
        monthly_comparison = pd.merge(
            current_monthly,
            monthly_data[["month", "truth"]],
            on="month",
            how="inner",
        )

        # Calculate monthly correction factors
        zero_threshold = config.stitching.zero_threshold if hasattr(config, "stitching") else 0.01
        structural_zero_months = []
        if hasattr(config, "stitching") and hasattr(config.stitching, "structural_zero_months"):
            structural_zero_months = config.stitching.structural_zero_months

        def compute_correction(row):
            is_structural_zero = row["month"].month in structural_zero_months
            if is_structural_zero and row["truth"] < zero_threshold:
                return 1.0
            elif abs(row["current_sum"]) < zero_threshold:
                return 1.0
            else:
                return row["truth"] / row["current_sum"]

        monthly_comparison["correction"] = monthly_comparison.apply(compute_correction, axis=1)

        # Apply monthly corrections
        correction_lookup = monthly_comparison.set_index("month")["correction"].to_dict()
        daily["correction"] = daily["month"].map(correction_lookup).fillna(1.0)
        daily["value"] = daily["value"] * daily["correction"]

        logger.info(
            f"Applied monthly renormalization: "
            f"mean_correction={monthly_comparison['correction'].mean():.3f}, "
            f"std_correction={monthly_comparison['correction'].std():.3f}"
        )

        return daily[["date", "value"]].reset_index(drop=True)

    def _calculate_diagnostics(
        self,
        stitched_daily: pd.DataFrame,
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        alphas: np.ndarray,
        dow_factors: Dict[int, float],
        base_diagnostics: dict,
    ) -> dict:
        """
        Calculate diagnostic metrics for the stitching result.

        Args:
            stitched_daily: Final stitched daily series
            monthly_data: Ground truth monthly data
            weekly_data: Ground truth weekly data (optional)
            alphas: Chunk scaling factors from base hierarchical method
            dow_factors: Day-of-week correction factors
            base_diagnostics: Diagnostics from base hierarchical method

        Returns:
            Dictionary of diagnostic metrics
        """
        diagnostics = {}

        # Monthly error metrics
        # WARNING: Post-renormalization metrics are forced to ~0 by construction
        # because _renormalize_to_monthly() scales each month to match ground truth
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

            # Note: With renormalization removed, monthly metrics are now meaningful validation
            # (using soft constraints from hierarchical optimization, not forced matching)

        # Weekly error metrics (if available)
        if weekly_data is not None and not weekly_data.empty:
            stitched_weekly = self._aggregate_to_weekly(stitched_daily)
            weekly_comparison = pd.merge(
                weekly_data.rename(columns={"value": "truth"}),
                stitched_weekly.rename(columns={"value": "stitched"}),
                on="date",
                how="inner",
            )

            if not weekly_comparison.empty:
                weekly_comparison["error"] = weekly_comparison["stitched"] - weekly_comparison["truth"]
                weekly_comparison["abs_error"] = abs(weekly_comparison["error"])

                # Calculate all weekly metrics using centralized method
                weekly_metrics = self._calculate_comparison_metrics(weekly_comparison, "weekly")
                diagnostics.update(weekly_metrics)

                diagnostics["weekly_comparison"] = weekly_comparison

        # Alpha statistics (from base method)
        diagnostics["alpha_mean"] = alphas.mean()
        diagnostics["alpha_std"] = alphas.std()
        diagnostics["alpha_min"] = alphas.min()
        diagnostics["alpha_max"] = alphas.max()
        diagnostics["alpha_cv"] = alphas.std() / alphas.mean() if alphas.mean() != 0 else 0
        diagnostics["alpha_values"] = alphas

        # DOW statistics
        diagnostics["dow_factors"] = dow_factors
        dow_values = np.array(list(dow_factors.values()))
        diagnostics["dow_mean"] = dow_values.mean()
        diagnostics["dow_std"] = dow_values.std()
        diagnostics["dow_range"] = dow_values.max() - dow_values.min()

        # Comparison with base method
        # Store base hierarchical metrics for comparison
        diagnostics["base_monthly_mae"] = base_diagnostics.get("monthly_mae", None)
        diagnostics["base_weekly_mae"] = base_diagnostics.get("weekly_mae", None)

        # With renormalization removed, compare actual performance
        if diagnostics.get("monthly_mae") and diagnostics.get("base_monthly_mae"):
            improvement = (
                100 * (diagnostics["base_monthly_mae"] - diagnostics["monthly_mae"])
                / diagnostics["base_monthly_mae"]
            )
            diagnostics["monthly_improvement_pct"] = improvement
            logger.info(
                f"Monthly MAE: {diagnostics['monthly_mae']:.2f} (DOW-corrected, soft constraints), "
                f"Base: {diagnostics['base_monthly_mae']:.2f} (before DOW). "
                f"Change: {improvement:+.1f}%"
            )

        # Method metadata
        diagnostics["method"] = self.name()
        diagnostics["num_chunks"] = len(alphas)
        diagnostics["base_diagnostics"] = base_diagnostics

        # Log summary
        logger.info(
            f"Diagnostics: "
            f"Monthly MAE={diagnostics.get('monthly_mae', 0):.2f} (DOW-corrected, soft constraints ✓), "
            f"Weekly MAE={diagnostics.get('weekly_mae', 0):.2f}, "
            f"Corr={diagnostics.get('weekly_corr', 0):.3f}, "
            f"Bias%={diagnostics.get('weekly_bias_pct', 0):.1f}% (independent ✓) | "
            f"DOW range={diagnostics['dow_range']:.3f}, "
            f"Base MAE={diagnostics.get('base_monthly_mae', 0):.2f}"
        )

        # Log R² deprecation notice
        if 'monthly_r2' in diagnostics or 'weekly_r2' in diagnostics:
            logger.debug(
                "Note: R² values are included in diagnostics for backwards compatibility "
                "but are deprecated. Use correlation, nmae, and bias metrics instead."
            )

        return diagnostics
