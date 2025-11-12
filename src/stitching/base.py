"""
Abstract base class for stitching engines.
Provides uniform interface for all stitching methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

# R² Deprecation Notice
R2_DEPRECATION_NOTE = """
[DEPRECATED] R² (coefficient of determination) is included for backwards compatibility
but should NOT be used as a primary validation metric for Google Trends stitching.

Why R² is misleading:
- Google Trends uses different normalization for different time ranges (monthly 0-100,
  daily/weekly 0.05-0.8), creating systematic scale mismatches
- R² penalizes predictions on different scales even when patterns match correctly
- Negative R² values indicate the model is worse than predicting the mean, but this
  reflects aggregation bucket mismatches, not necessarily poor pattern matching

Recommended metrics instead:
- Pearson correlation (corr): Measures pattern similarity independent of scale
- Normalized MAE (nmae): Scale-invariant accuracy measure
- Bias percentage (bias_pct): Identifies systematic over/under-prediction
- MAE/RMSE: Direct error magnitude for same-scale comparisons
"""


@dataclass
class StitchingResult:
    """
    Result from a stitching method.
    """

    stitched_series: pd.DataFrame  # Daily time series [date, value]
    alpha_estimates: np.ndarray  # Scaling factors for each chunk
    diagnostics: Dict[str, Any] = field(default_factory=dict)  # Method-specific diagnostics
    method_name: str = ""

    def save(self, file_manager, method_name: str):
        """
        Save stitching results to disk.

        Args:
            file_manager: FileManager instance
            method_name: Name of the method (for directory structure)
        """
        import pickle

        # Save stitched series as parquet
        series_path = file_manager.get_interim_path(method_name, "stitched_series.parquet")
        self.stitched_series.to_parquet(series_path, index=False)
        logger.info(f"Saved stitched series to {series_path}")

        # Save alpha estimates as pickle
        alpha_path = file_manager.get_interim_path(method_name, "alpha_estimates.pkl")
        with open(alpha_path, "wb") as f:
            pickle.dump(self.alpha_estimates, f)
        logger.info(f"Saved alpha estimates to {alpha_path}")

        # Save diagnostics as pickle
        diagnostics_path = file_manager.get_interim_path(method_name, "diagnostics.pkl")
        with open(diagnostics_path, "wb") as f:
            pickle.dump(self.diagnostics, f)
        logger.info(f"Saved diagnostics to {diagnostics_path}")

    @classmethod
    def load(cls, file_manager, method_name: str) -> "StitchingResult":
        """
        Load stitching results from disk.

        Args:
            file_manager: FileManager instance
            method_name: Name of the method

        Returns:
            StitchingResult instance
        """
        import pickle

        # Load stitched series
        series_path = file_manager.get_interim_path(method_name, "stitched_series.parquet")
        stitched_series = pd.read_parquet(series_path)

        # Load alpha estimates
        alpha_path = file_manager.get_interim_path(method_name, "alpha_estimates.pkl")
        with open(alpha_path, "rb") as f:
            alpha_estimates = pickle.load(f)

        # Load diagnostics
        diagnostics_path = file_manager.get_interim_path(method_name, "diagnostics.pkl")
        with open(diagnostics_path, "rb") as f:
            diagnostics = pickle.load(f)

        logger.info(f"Loaded stitching results for {method_name}")

        return cls(
            stitched_series=stitched_series,
            alpha_estimates=alpha_estimates,
            diagnostics=diagnostics,
            method_name=method_name,
        )


class StitchingEngine(ABC):
    """
    Abstract base class for all stitching methods.
    Provides uniform interface for comparison.
    """

    @abstractmethod
    def stitch(
        self,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> StitchingResult:
        """
        Stitch overlapping daily chunks into a continuous time series.

        Args:
            daily_chunks: List of daily chunk DataFrames [date, value]
            monthly_data: Monthly ground truth [date, value]
            weekly_data: Weekly ground truth [date, value] (optional)
            config: Configuration object

        Returns:
            StitchingResult with stitched series, alpha estimates, and diagnostics
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the stitching method."""
        pass

    def _validate_inputs(
        self,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
    ):
        """
        Validate input data.

        Args:
            daily_chunks: List of daily chunks
            monthly_data: Monthly data
            weekly_data: Weekly data (optional)

        Raises:
            ValueError: If validation fails
        """
        if not daily_chunks:
            raise ValueError("daily_chunks cannot be empty")

        if monthly_data.empty:
            raise ValueError("monthly_data cannot be empty")

        required_columns = ["date", "value"]

        # Check monthly data
        for col in required_columns:
            if col not in monthly_data.columns:
                raise ValueError(f"monthly_data missing column: {col}")

        # Check weekly data
        if weekly_data is not None and not weekly_data.empty:
            for col in required_columns:
                if col not in weekly_data.columns:
                    raise ValueError(f"weekly_data missing column: {col}")

        # Check daily chunks
        for i, chunk in enumerate(daily_chunks):
            if chunk.empty:
                logger.warning(f"Chunk {i} is empty")
                continue

            for col in required_columns:
                if col not in chunk.columns:
                    raise ValueError(f"Chunk {i} missing column: {col}")

        logger.debug("Input validation passed")

    def _concatenate_chunks(self, daily_chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate all daily chunks (with duplicates in overlap regions).

        Args:
            daily_chunks: List of daily chunks

        Returns:
            Concatenated DataFrame with all daily data
        """
        all_chunks = []
        for i, chunk in enumerate(daily_chunks):
            if chunk.empty:
                continue

            chunk_copy = chunk.copy()
            chunk_copy["chunk_id"] = i
            all_chunks.append(chunk_copy)

        concatenated = pd.concat(all_chunks, ignore_index=True)
        concatenated["date"] = pd.to_datetime(concatenated["date"])
        concatenated = concatenated.sort_values("date").reset_index(drop=True)

        logger.debug(f"Concatenated {len(daily_chunks)} chunks into {len(concatenated)} rows")
        return concatenated

    def _aggregate_to_monthly(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily data to monthly.

        Args:
            daily: Daily data [date, value]

        Returns:
            Monthly aggregated data [date, value]
        """
        daily_copy = daily.copy()
        daily_copy["date"] = pd.to_datetime(daily_copy["date"])
        daily_copy["month"] = daily_copy["date"].dt.to_period("M")

        monthly = daily_copy.groupby("month")["value"].sum().reset_index()
        monthly["date"] = monthly["month"].dt.to_timestamp()
        monthly = monthly[["date", "value"]]

        return monthly

    def _aggregate_to_weekly(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily data to weekly (Sunday-ending weeks to match Google Trends).

        Args:
            daily: Daily data [date, value]

        Returns:
            Weekly aggregated data [date, value] with dates on Sundays
        """
        daily_copy = daily.copy()
        daily_copy["date"] = pd.to_datetime(daily_copy["date"])

        # Use Sunday-ending weeks to match Google Trends format
        # resample('W-SUN') groups weeks ending on Sunday
        daily_copy = daily_copy.set_index("date")
        weekly = daily_copy.resample("W-SUN")["value"].sum().reset_index()

        return weekly

    def _calculate_overlap_indices(self, daily_chunks: List[pd.DataFrame]) -> List[List[int]]:
        """
        Calculate indices of overlap regions between consecutive chunks.

        Args:
            daily_chunks: List of daily chunks

        Returns:
            List of lists, where each inner list contains global indices in overlap region
        """
        overlaps = []

        # Build a mapping from date to global index
        all_data = self._concatenate_chunks(daily_chunks)
        date_to_indices = {}

        for idx, row in all_data.iterrows():
            date = row["date"]
            if date not in date_to_indices:
                date_to_indices[date] = []
            date_to_indices[date].append(idx)

        # Find overlaps (dates that appear in multiple chunks)
        for date, indices in date_to_indices.items():
            if len(indices) > 1:
                overlaps.append(indices)

        logger.debug(f"Found {len(overlaps)} overlap dates")
        return overlaps

    def _calculate_comparison_metrics(
        self, comparison_df: pd.DataFrame, prefix: str = "monthly"
    ) -> Dict[str, float]:
        """
        Calculate standardized validation metrics for monthly/weekly comparisons.

        This method centralizes all metric calculation to ensure consistency across
        all stitching methods.

        Args:
            comparison_df: DataFrame with columns [date, stitched, truth, error, abs_error]
            prefix: Prefix for metric names ("monthly" or "weekly")

        Returns:
            Dictionary with the following metrics:
                - {prefix}_mae: Mean Absolute Error
                - {prefix}_rmse: Root Mean Squared Error
                - {prefix}_corr: Pearson correlation coefficient (pattern similarity)
                - {prefix}_nmae: Normalized MAE (MAE / mean(truth)) - scale-invariant
                - {prefix}_bias: Mean bias (mean(predicted - actual))
                - {prefix}_bias_pct: Bias as percentage of mean(truth)
                - {prefix}_r2: [DEPRECATED] Coefficient of determination

        Note:
            R² is deprecated for Google Trends validation. Use correlation, nmae,
            and bias metrics instead. See R2_DEPRECATION_NOTE for details.
        """
        metrics = {}

        # Extract data
        stitched = comparison_df["stitched"].values
        truth = comparison_df["truth"].values
        error = comparison_df["error"].values
        abs_error = comparison_df["abs_error"].values

        # Calculate means for normalization
        mean_truth = truth.mean()
        mean_stitched = stitched.mean()

        # 1. Mean Absolute Error (MAE)
        metrics[f"{prefix}_mae"] = abs_error.mean()

        # 2. Root Mean Squared Error (RMSE)
        metrics[f"{prefix}_rmse"] = np.sqrt((error**2).mean())

        # 3. Pearson Correlation Coefficient (pattern similarity, scale-independent)
        # This is the PRIMARY metric for pattern matching
        if len(stitched) > 1 and np.std(stitched) > 0 and np.std(truth) > 0:
            correlation_matrix = np.corrcoef(stitched, truth)
            metrics[f"{prefix}_corr"] = correlation_matrix[0, 1]
        else:
            metrics[f"{prefix}_corr"] = 0.0

        # 4. Normalized MAE (scale-invariant accuracy)
        # Enables comparison across different search terms
        if mean_truth != 0:
            metrics[f"{prefix}_nmae"] = metrics[f"{prefix}_mae"] / mean_truth
        else:
            metrics[f"{prefix}_nmae"] = np.inf

        # 5. Mean Bias (systematic directional error)
        metrics[f"{prefix}_bias"] = error.mean()

        # 6. Bias Percentage (systematic error as % of mean)
        if mean_truth != 0:
            metrics[f"{prefix}_bias_pct"] = 100 * metrics[f"{prefix}_bias"] / mean_truth
        else:
            metrics[f"{prefix}_bias_pct"] = np.inf

        # 7. R² [DEPRECATED - kept for backwards compatibility]
        # This should NOT be used as primary metric - see R2_DEPRECATION_NOTE
        ss_res = (error**2).sum()
        ss_tot = ((truth - mean_truth) ** 2).sum()
        if ss_tot > 0:
            metrics[f"{prefix}_r2"] = 1 - (ss_res / ss_tot)
        else:
            metrics[f"{prefix}_r2"] = 0.0

        return metrics

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"
