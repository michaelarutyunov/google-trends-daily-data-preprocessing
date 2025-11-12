"""
Rung 5: State-Space Stitching Method (Kalman Filter) [INCOMPLETE IMPLEMENTATION]

⚠️  WARNING: This implementation is NOT a true state-space Kalman filter.

CURRENT IMPLEMENTATION:
- Uses monthly scaling + exponential temporal smoothing
- Provides heuristic confidence bands (not true Kalman covariance)
- Results similar to baseline method (Weekly MAE ~1.37)

PLANNED IMPLEMENTATION (not yet done):
1. Define state-space model:
   - State: x_t = log(S_t) where S_t is true search index
   - Observations: y_{k,t} = α_k * exp(x_t) + noise
2. Use statsmodels MLEModel to estimate parameters
3. Run Kalman filter/smoother for optimal state estimates
4. Calculate confidence bands from state covariance

RECOMMENDATION:
For best results, use HierarchicalStitcher instead (Weekly MAE ~0.91).
Only use this method if you understand its limitations and need confidence bands
for exploratory analysis (not production).
"""

from typing import List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger

try:
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    from statsmodels.tsa.statespace import tools
except ImportError:
    raise ImportError(
        "statsmodels is required for StateSpaceStitcher. "
        "Install with: uv pip install statsmodels"
    )

from .base import StitchingEngine, StitchingResult


class StateSpaceStitcher(StitchingEngine):
    """
    State-space stitching using Kalman filter.

    State-space model:
    - State equation: x_t = x_{t-1} + w_t, where w_t ~ N(0, Q)
    - Observation equation: y_{k,t} = α_k * exp(x_t) + v_{k,t}, where v_{k,t} ~ N(0, R_k)

    The Kalman filter provides:
    1. Optimal state estimates given all observations
    2. State covariance (uncertainty quantification)
    3. Confidence bands for predictions
    """

    def name(self) -> str:
        """Return method name."""
        return "state_space"

    def stitch(
        self,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> StitchingResult:
        """
        Stitch daily chunks using state-space Kalman filter.

        Args:
            daily_chunks: List of daily chunk DataFrames [date, value]
            monthly_data: Monthly ground truth [date, value]
            weekly_data: Weekly ground truth [date, value] (optional but recommended)
            config: Configuration object

        Returns:
            StitchingResult with stitched series, alphas, and confidence bands
        """
        logger.info(f"Starting {self.name()} stitching method")

        # WARNING: This is NOT a true state-space Kalman filter implementation
        logger.warning(
            "⚠️  STATE-SPACE METHOD LIMITATION: This implementation uses temporal smoothing "
            "heuristics, not a true Kalman filter. Results are similar to baseline method "
            "(Weekly MAE ~1.37). For best results, use HierarchicalStitcher (Weekly MAE ~0.91). "
            "True state-space Kalman filter implementation is planned for future release."
        )

        # Validate inputs
        self._validate_inputs(daily_chunks, monthly_data, weekly_data)

        # Step 1: Prepare data structures
        logger.info("Step 1: Building daily data index")
        daily_index = self._build_daily_index(daily_chunks)

        # Step 2: Initialize alpha estimates using hierarchical method
        logger.info("Step 2: Initializing alpha estimates")
        initial_alphas = self._initialize_alphas(daily_index, monthly_data)

        # Step 3: Run simplified Kalman-inspired smoothing
        logger.info("Step 3: Running state-space estimation")
        stitched_daily, refined_alphas, confidence_bands, solve_info = self._run_state_space_estimation(
            daily_index,
            initial_alphas,
            monthly_data,
            weekly_data,
            config,
        )

        # Step 4: Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            stitched_daily,
            monthly_data,
            weekly_data,
            refined_alphas,
            daily_index,
            solve_info,
            confidence_bands,
        )

        logger.success(f"{self.name()} stitching completed")

        return StitchingResult(
            stitched_series=stitched_daily,
            alpha_estimates=refined_alphas,
            diagnostics=diagnostics,
            method_name=self.name(),
        )

    def _build_daily_index(self, daily_chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Build indexed daily data structure.

        Args:
            daily_chunks: List of daily chunks

        Returns:
            DataFrame with [date, value, chunk_id, day_index]
        """
        all_data = []

        for chunk_id, chunk in enumerate(daily_chunks):
            if chunk.empty:
                logger.warning(f"Chunk {chunk_id} is empty, skipping")
                continue

            chunk_copy = chunk.copy()
            chunk_copy["date"] = pd.to_datetime(chunk_copy["date"])
            chunk_copy["chunk_id"] = chunk_id
            all_data.append(chunk_copy[["date", "value", "chunk_id"]])

        # Concatenate all chunks
        daily_index = pd.concat(all_data, ignore_index=True)
        daily_index = daily_index.sort_values(["date", "chunk_id"]).reset_index(drop=True)
        daily_index["day_index"] = daily_index.index

        logger.info(
            f"Built daily index: {len(daily_index)} observations, "
            f"{len(daily_chunks)} chunks, "
            f"{daily_index['date'].nunique()} unique dates"
        )

        return daily_index

    def _initialize_alphas(
        self,
        daily_index: pd.DataFrame,
        monthly_data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Initialize alpha estimates using simple monthly scaling.

        This provides a good starting point for the state-space estimation.

        Args:
            daily_index: Indexed daily data
            monthly_data: Monthly ground truth

        Returns:
            Array of initial alpha estimates (one per chunk)
        """
        num_chunks = daily_index["chunk_id"].max() + 1
        initial_alphas = np.ones(num_chunks)

        # Calculate monthly aggregates for each chunk
        monthly_data_copy = monthly_data.copy()
        monthly_data_copy["date"] = pd.to_datetime(monthly_data_copy["date"])
        monthly_data_copy["month"] = monthly_data_copy["date"].dt.to_period("M")

        daily_index_copy = daily_index.copy()
        daily_index_copy["month"] = daily_index_copy["date"].dt.to_period("M")

        for chunk_id in range(num_chunks):
            chunk_data = daily_index_copy[daily_index_copy["chunk_id"] == chunk_id]

            if chunk_data.empty:
                continue

            # Aggregate chunk to monthly
            chunk_monthly = chunk_data.groupby("month")["value"].sum().reset_index()
            chunk_monthly.columns = ["month", "chunk_sum"]

            # Merge with ground truth
            merged = pd.merge(
                monthly_data_copy[["month", "value"]],
                chunk_monthly,
                on="month",
                how="inner",
            )

            if not merged.empty and merged["chunk_sum"].sum() > 0:
                # Calculate alpha as mean ratio
                alpha = merged["value"].sum() / merged["chunk_sum"].sum()
                initial_alphas[chunk_id] = alpha

        logger.info(
            f"Initialized alphas: mean={initial_alphas.mean():.3f}, "
            f"CV={initial_alphas.std()/initial_alphas.mean():.3f}"
        )

        return initial_alphas

    def _run_state_space_estimation(
        self,
        daily_index: pd.DataFrame,
        initial_alphas: np.ndarray,
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        """
        Run state-space estimation using iterative refinement.

        This is a simplified approach that:
        1. Starts with initial alphas
        2. Applies local smoothing with uncertainty estimates
        3. Refines alphas based on monthly/weekly constraints
        4. Generates confidence bands

        Args:
            daily_index: Indexed daily data
            initial_alphas: Initial alpha estimates
            monthly_data: Monthly ground truth
            weekly_data: Weekly ground truth (optional)
            config: Configuration object

        Returns:
            Tuple of (stitched_daily, refined_alphas, confidence_bands, solve_info)
        """
        # Get state-space parameters from config
        process_noise = config.stitching.state_space.process_noise
        obs_noise = config.stitching.state_space.observation_noise
        confidence_level = config.stitching.state_space.confidence_level

        logger.info(
            f"State-space parameters: Q={process_noise}, R={obs_noise}, "
            f"confidence={confidence_level}"
        )

        # Step 1: Apply initial alphas to get raw estimates
        daily_scaled = daily_index.copy()
        daily_scaled["alpha"] = daily_scaled["chunk_id"].map(
            lambda chunk_id: initial_alphas[chunk_id]
        )
        daily_scaled["scaled_value"] = daily_scaled["value"] * daily_scaled["alpha"]

        # Step 2: Average overlapping dates and compute variance
        grouped = daily_scaled.groupby("date").agg(
            value_mean=("scaled_value", "mean"),
            value_std=("scaled_value", "std"),
            count=("scaled_value", "count"),
        ).reset_index()

        # Handle single observations (no std)
        grouped["value_std"] = grouped["value_std"].fillna(obs_noise * grouped["value_mean"])
        grouped.loc[grouped["value_std"] == 0, "value_std"] = obs_noise * grouped.loc[grouped["value_std"] == 0, "value_mean"]

        # Step 3: Apply exponential smoothing (simplified Kalman filter)
        # This is a lightweight alternative to full state-space modeling
        smoothed_values = []
        smoothed_std = []

        alpha_smooth = 0.3  # Smoothing parameter (similar to Kalman gain)
        prev_value = grouped["value_mean"].iloc[0]
        prev_std = grouped["value_std"].iloc[0]

        for i, row in grouped.iterrows():
            # Prediction step (use previous value)
            predicted_value = prev_value
            predicted_std = np.sqrt(prev_std**2 + process_noise)

            # Update step (incorporate observation)
            obs_value = row["value_mean"]
            obs_std = row["value_std"] / np.sqrt(row["count"])

            # Kalman-like update
            kalman_gain = predicted_std**2 / (predicted_std**2 + obs_std**2)
            updated_value = predicted_value + kalman_gain * (obs_value - predicted_value)
            updated_std = np.sqrt((1 - kalman_gain) * predicted_std**2)

            smoothed_values.append(updated_value)
            smoothed_std.append(updated_std)

            prev_value = updated_value
            prev_std = updated_std

        grouped["smoothed_value"] = smoothed_values
        grouped["smoothed_std"] = smoothed_std

        # Step 4: Calculate confidence bands
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        confidence_bands = grouped[["date", "smoothed_value", "smoothed_std"]].copy()
        confidence_bands["lower"] = confidence_bands["smoothed_value"] - z_score * confidence_bands["smoothed_std"]
        confidence_bands["upper"] = confidence_bands["smoothed_value"] + z_score * confidence_bands["smoothed_std"]
        # Ensure non-negative
        confidence_bands["lower"] = confidence_bands["lower"].clip(lower=0)

        # Step 5: Refine alphas based on monthly constraints
        refined_alphas = self._refine_alphas(
            daily_index,
            grouped,
            monthly_data,
            initial_alphas,
        )

        # Step 6: Apply refined alphas
        daily_scaled["alpha_refined"] = daily_scaled["chunk_id"].map(
            lambda chunk_id: refined_alphas[chunk_id]
        )
        daily_scaled["final_value"] = daily_scaled["value"] * daily_scaled["alpha_refined"]

        # Final aggregation
        stitched_daily = daily_scaled.groupby("date", as_index=False)["final_value"].mean()
        stitched_daily = stitched_daily.rename(columns={"final_value": "value"})
        stitched_daily = stitched_daily.sort_values("date").reset_index(drop=True)

        # Update confidence bands to match stitched values
        confidence_bands = pd.merge(
            stitched_daily[["date", "value"]],
            confidence_bands[["date", "lower", "upper"]],
            on="date",
            how="left",
        )

        solve_info = {
            "process_noise": process_noise,
            "observation_noise": obs_noise,
            "confidence_level": confidence_level,
            "smoothing_alpha": alpha_smooth,
            "converged": True,
            "method": "exponential_smoothing_with_kalman_gain",
        }

        logger.info(
            f"State-space estimation complete: "
            f"{len(stitched_daily)} dates, "
            f"mean_std={grouped['smoothed_std'].mean():.3f}"
        )

        return stitched_daily, refined_alphas, confidence_bands, solve_info

    def _refine_alphas(
        self,
        daily_index: pd.DataFrame,
        smoothed_daily: pd.DataFrame,
        monthly_data: pd.DataFrame,
        initial_alphas: np.ndarray,
    ) -> np.ndarray:
        """
        Refine alpha estimates to match monthly constraints.

        Args:
            daily_index: Original daily index
            smoothed_daily: Smoothed daily estimates
            monthly_data: Monthly ground truth
            initial_alphas: Initial alpha estimates

        Returns:
            Refined alpha array
        """
        num_chunks = len(initial_alphas)
        refined_alphas = initial_alphas.copy()

        # Aggregate smoothed daily to monthly
        smoothed_copy = smoothed_daily.copy()
        smoothed_copy["date"] = pd.to_datetime(smoothed_copy["date"])
        smoothed_copy["month"] = smoothed_copy["date"].dt.to_period("M")
        smoothed_monthly = smoothed_copy.groupby("month")["smoothed_value"].sum().reset_index()

        # Compare with ground truth
        monthly_copy = monthly_data.copy()
        monthly_copy["date"] = pd.to_datetime(monthly_copy["date"])
        monthly_copy["month"] = monthly_copy["date"].dt.to_period("M")

        merged = pd.merge(
            monthly_copy[["month", "value"]],
            smoothed_monthly,
            on="month",
            how="inner",
        )

        if not merged.empty:
            # Calculate overall correction factor
            total_truth = merged["value"].sum()
            total_smoothed = merged["smoothed_value"].sum()

            if total_smoothed > 0:
                correction = total_truth / total_smoothed
                refined_alphas = initial_alphas * correction

                logger.info(f"Applied monthly correction factor: {correction:.4f}")

        return refined_alphas

    def _calculate_diagnostics(
        self,
        stitched_daily: pd.DataFrame,
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        alphas: np.ndarray,
        daily_index: pd.DataFrame,
        solve_info: dict,
        confidence_bands: pd.DataFrame,
    ) -> dict:
        """
        Calculate diagnostic metrics for the stitching result.

        Args:
            stitched_daily: Final stitched daily series
            monthly_data: Ground truth monthly data
            weekly_data: Ground truth weekly data (optional)
            alphas: Chunk scaling factors
            daily_index: Original daily index
            solve_info: State-space estimation information
            confidence_bands: Confidence band DataFrame

        Returns:
            Dictionary of diagnostic metrics
        """
        diagnostics = {}

        # Monthly error metrics
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

        # Alpha statistics
        diagnostics["alpha_mean"] = alphas.mean()
        diagnostics["alpha_std"] = alphas.std()
        diagnostics["alpha_min"] = alphas.min()
        diagnostics["alpha_max"] = alphas.max()
        diagnostics["alpha_cv"] = alphas.std() / alphas.mean() if alphas.mean() != 0 else 0
        diagnostics["alpha_values"] = alphas

        # Confidence band statistics
        diagnostics["confidence_bands"] = confidence_bands
        diagnostics["mean_confidence_width"] = (
            confidence_bands["upper"] - confidence_bands["lower"]
        ).mean()

        # State-space diagnostics
        diagnostics["state_space"] = solve_info

        # Method metadata
        diagnostics["method"] = self.name()
        diagnostics["num_chunks"] = len(alphas)

        # Log summary
        log_msg = (
            f"Diagnostics: "
            f"Monthly MAE={diagnostics.get('monthly_mae', 0):.2f}, "
            f"Corr={diagnostics.get('monthly_corr', 0):.3f}, "
            f"Bias%={diagnostics.get('monthly_bias_pct', 0):.1f}%, "
            f"Alpha CV={diagnostics['alpha_cv']:.3f}, "
            f"Mean CI width={diagnostics['mean_confidence_width']:.2f}, "
            f"Converged={solve_info['converged']}"
        )

        # Add weekly metrics if available
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
