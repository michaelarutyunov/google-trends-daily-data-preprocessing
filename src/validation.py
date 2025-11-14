"""
Data validation and quality checks for Google Trends data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import warnings
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class ValidationReport:
    """
    Report containing validation results.
    """

    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)

    def add_warning(self, message: str):
        """Add a warning to the report."""
        self.warnings.append(message)
        logger.warning(f"Validation warning: {message}")

    def add_error(self, message: str):
        """Add an error to the report."""
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"Validation error: {message}")

    def summary(self) -> str:
        """Generate validation summary."""
        status = "VALID" if self.is_valid else "INVALID"
        summary = f"\nValidation Report - Status: {status}\n"
        summary += "=" * 50 + "\n\n"

        if self.errors:
            summary += "ERRORS:\n"
            for error in self.errors:
                summary += f"  - {error}\n"
            summary += "\n"

        if self.warnings:
            summary += "WARNINGS:\n"
            for warning in self.warnings:
                summary += f"  - {warning}\n"
            summary += "\n"

        if self.statistics:
            summary += "STATISTICS:\n"
            for key, value in self.statistics.items():
                summary += f"  {key}: {value}\n"

        return summary


class DataValidator:
    """
    Validates Google Trends data quality and consistency.
    """

    def __init__(self, zero_threshold: float = 0.01):
        """
        Initialize DataValidator.

        Args:
            zero_threshold: Values below this are considered zero
        """
        self.zero_threshold = zero_threshold

    def validate(
        self,
        monthly: pd.DataFrame,
        weekly: Optional[pd.DataFrame] = None,
        daily_chunks: Optional[List[pd.DataFrame]] = None,
        structural_zero_months: Optional[List[int]] = None,
    ) -> ValidationReport:
        """
        Validate data quality and consistency.

        Args:
            monthly: Monthly data
            weekly: Weekly data (optional)
            daily_chunks: List of daily chunk DataFrames (optional)
            structural_zero_months: Months expected to have structural zeros

        Returns:
            ValidationReport
        """
        report = ValidationReport(is_valid=True)
        logger.info("Starting validation...")

        # Validate monthly data
        logger.info("Validating monthly data...")
        self._validate_dataframe(monthly, "monthly", report)
        self._check_completeness(monthly, "monthly", report)
        self._check_zero_pattern(monthly, "monthly", structural_zero_months, report)

        # Validate weekly data
        if weekly is not None:
            logger.info("Validating weekly data...")
            self._validate_dataframe(weekly, "weekly", report)
            self._check_completeness(weekly, "weekly", report)

        # Validate daily chunks
        if daily_chunks is not None:
            logger.info(f"Validating {len(daily_chunks)} daily chunks...")
            self._validate_daily_chunks(daily_chunks, report)

        # Cross-validation
        if weekly is not None and not report.errors:
            logger.info("Running cross-validation...")
            self._cross_validate_monthly_weekly(monthly, weekly, report)

        # Calculate statistics
        logger.info("Calculating statistics...")
        self._calculate_statistics(monthly, weekly, daily_chunks, report)

        logger.info(f"Validation complete: {'VALID' if report.is_valid else 'INVALID'}")
        return report

    def _validate_dataframe(self, df: pd.DataFrame, name: str, report: ValidationReport):
        """Validate basic DataFrame structure."""
        required_columns = ["date", "value"]

        for col in required_columns:
            if col not in df.columns:
                report.add_error(f"{name}: Missing required column '{col}'")

        if df.empty:
            report.add_error(f"{name}: DataFrame is empty")
            return

        # Check for NaN values
        if df["value"].isna().any():
            nan_count = df["value"].isna().sum()
            report.add_warning(f"{name}: Contains {nan_count} NaN values")

        # Check for negative values
        if (df["value"] < 0).any():
            neg_count = (df["value"] < 0).sum()
            report.add_error(f"{name}: Contains {neg_count} negative values")

    def _check_completeness(self, df: pd.DataFrame, name: str, report: ValidationReport):
        """Check for missing dates."""
        if df.empty:
            return

        df = df.sort_values("date")
        dates = pd.to_datetime(df["date"])

        # Infer expected frequency
        if len(dates) < 2:
            return

        date_diffs = dates.diff().dropna()
        median_diff = date_diffs.median()

        # Check for gaps larger than expected
        large_gaps = date_diffs[date_diffs > median_diff * 2]
        if len(large_gaps) > 0:
            report.add_warning(
                f"{name}: Found {len(large_gaps)} gaps larger than expected frequency"
            )

    def _check_zero_pattern(
        self,
        df: pd.DataFrame,
        name: str,
        structural_zero_months: Optional[List[int]],
        report: ValidationReport,
    ):
        """Analyze zero patterns."""
        if df.empty:
            return

        zero_mask = df["value"] <= self.zero_threshold
        zero_count = zero_mask.sum()
        zero_pct = 100 * zero_count / len(df)

        report.statistics[f"{name}_zero_count"] = int(zero_count)
        report.statistics[f"{name}_zero_percentage"] = f"{zero_pct:.1f}%"

        if zero_pct > 50:
            report.add_warning(f"{name}: High zero percentage ({zero_pct:.1f}%)")

        # Check structural zeros
        if structural_zero_months:
            df = df.copy()
            df["month"] = pd.to_datetime(df["date"]).dt.month
            structural_mask = df["month"].isin(structural_zero_months)

            structural_zeros = (structural_mask & zero_mask).sum()
            non_structural_zeros = (~structural_mask & zero_mask).sum()

            report.statistics[f"{name}_structural_zeros"] = int(structural_zeros)
            report.statistics[f"{name}_sampling_zeros"] = int(non_structural_zeros)

            logger.info(
                f"{name}: {structural_zeros} structural zeros, {non_structural_zeros} sampling zeros"
            )

    def _validate_daily_chunks(self, daily_chunks: List[pd.DataFrame], report: ValidationReport):
        """Validate daily chunk structure and overlaps."""
        if not daily_chunks:
            report.add_error("daily_chunks: No chunks provided")
            return

        logger.info(f"Validating {len(daily_chunks)} daily chunks...")
        for i, chunk in enumerate(daily_chunks):
            logger.debug(f"Validating chunk {i} ({len(chunk)} rows)...")
            self._validate_dataframe(chunk, f"chunk_{i}", report)

        # Check overlaps between consecutive chunks
        logger.info("Checking overlaps between consecutive chunks...")
        for i in range(len(daily_chunks) - 1):
            logger.debug(f"Checking overlap {i}-{i+1}...")
            chunk1 = daily_chunks[i]
            chunk2 = daily_chunks[i + 1]

            if chunk1.empty or chunk2.empty:
                continue

            dates1 = set(pd.to_datetime(chunk1["date"]))
            dates2 = set(pd.to_datetime(chunk2["date"]))

            overlap = dates1 & dates2
            overlap_days = len(overlap)

            if overlap_days == 0:
                report.add_warning(f"No overlap between chunk_{i} and chunk_{i+1}")
            else:
                logger.info(f"Chunk {i}-{i+1} overlap: {overlap_days} days")

            report.statistics[f"chunk_{i}_{i+1}_overlap_days"] = overlap_days

        logger.info("Daily chunk validation complete")

    def _cross_validate_monthly_weekly(
        self, monthly: pd.DataFrame, weekly: pd.DataFrame, report: ValidationReport
    ):
        """Cross-validate monthly and weekly data consistency."""
        if monthly.empty or weekly.empty:
            return

        # Aggregate weekly to monthly
        weekly_copy = weekly.copy()
        weekly_copy["date"] = pd.to_datetime(weekly_copy["date"])
        weekly_copy["month"] = weekly_copy["date"].dt.to_period("M")

        weekly_monthly = weekly_copy.groupby("month")["value"].sum().reset_index()
        weekly_monthly["date"] = weekly_monthly["month"].dt.to_timestamp()

        # Merge with monthly data
        monthly_copy = monthly.copy()
        monthly_copy["date"] = pd.to_datetime(monthly_copy["date"])

        merged = pd.merge(
            monthly_copy,
            weekly_monthly[["date", "value"]],
            on="date",
            suffixes=("_monthly", "_weekly"),
            how="inner",
        )

        if merged.empty:
            report.add_warning("No overlapping months between monthly and weekly data")
            return

        # Calculate correlation and MAE
        correlation = merged["value_monthly"].corr(merged["value_weekly"])
        mae = np.abs(merged["value_monthly"] - merged["value_weekly"]).mean()
        mae_pct = 100 * mae / merged["value_monthly"].mean()

        report.statistics["monthly_weekly_correlation"] = f"{correlation:.3f}"
        report.statistics["monthly_weekly_mae"] = f"{mae:.2f}"
        report.statistics["monthly_weekly_mae_pct"] = f"{mae_pct:.1f}%"

        if correlation < 0.8:
            report.add_warning(
                f"Low correlation between monthly and weekly data (r={correlation:.3f})"
            )

        if mae_pct > 20:
            report.add_warning(
                f"High discrepancy between monthly and weekly data (MAE={mae_pct:.1f}%)"
            )

    def _calculate_statistics(
        self,
        monthly: pd.DataFrame,
        weekly: Optional[pd.DataFrame],
        daily_chunks: Optional[List[pd.DataFrame]],
        report: ValidationReport,
    ):
        """Calculate summary statistics."""
        # Monthly stats
        if not monthly.empty:
            report.statistics["monthly_count"] = len(monthly)
            report.statistics["monthly_mean"] = f"{monthly['value'].mean():.2f}"
            report.statistics["monthly_std"] = f"{monthly['value'].std():.2f}"
            report.statistics["monthly_min"] = int(monthly["value"].min())
            report.statistics["monthly_max"] = int(monthly["value"].max())

        # Weekly stats
        if weekly is not None and not weekly.empty:
            report.statistics["weekly_count"] = len(weekly)
            report.statistics["weekly_mean"] = f"{weekly['value'].mean():.2f}"

        # Daily chunks stats
        if daily_chunks is not None:
            total_daily_points = sum(len(chunk) for chunk in daily_chunks)
            report.statistics["num_daily_chunks"] = len(daily_chunks)
            report.statistics["total_daily_points"] = total_daily_points


class Validator:
    """
    Validates stitched results against ground truth.
    """

    @staticmethod
    def calculate_mae(
        stitched: pd.DataFrame,
        ground_truth: pd.DataFrame,
        resolution: str = "monthly",
        return_nmae: bool = False,
    ) -> float:
        """
        Calculate Mean Absolute Error (and optionally NMAE).

        .. deprecated:: This function is orphaned and not used by stitching methods.
            All metrics are calculated by StitchingEngine._calculate_comparison_metrics()
            in base.py. This function is retained for standalone validation only.

        Args:
            stitched: Stitched daily series
            ground_truth: Ground truth (monthly or weekly)
            resolution: Aggregation resolution ('monthly' or 'weekly')
            return_nmae: If True, return (MAE, NMAE) tuple instead of MAE scalar

        Returns:
            MAE value, or (MAE, NMAE) tuple if return_nmae=True
        """
        warnings.warn(
            "Validator.calculate_mae() is orphaned. "
            "Use StitchingEngine._calculate_comparison_metrics() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Aggregate stitched to resolution
        stitched_copy = stitched.copy()
        stitched_copy["date"] = pd.to_datetime(stitched_copy["date"])

        if resolution == "monthly":
            stitched_copy["period"] = stitched_copy["date"].dt.to_period("M")
            stitched_agg = stitched_copy.groupby("period")["value"].sum().reset_index()
            stitched_agg["date"] = stitched_agg["period"].dt.to_timestamp()
        elif resolution == "weekly":
            # FIXED: Use resample('W-SUN') to produce Sunday dates (not Monday)
            # Previous to_period('W-SUN').to_timestamp() produced Mondays and failed to merge
            stitched_copy = stitched_copy.set_index("date")
            stitched_agg = stitched_copy.resample("W-SUN")["value"].sum().reset_index()

            # Verify Sunday alignment
            if not stitched_agg.empty:
                assert (stitched_agg["date"].dt.dayofweek == 6).all(), \
                    f"Weekly aggregation produced non-Sunday dates: {stitched_agg['date'].dt.day_name().unique()}"
        else:
            raise ValueError(f"Invalid resolution: {resolution}")

        # Merge with ground truth
        gt_copy = ground_truth.copy()
        gt_copy["date"] = pd.to_datetime(gt_copy["date"])

        # Verify ground truth frequency for weekly data
        if resolution == "weekly" and len(gt_copy) > 1:
            freq_mode = gt_copy["date"].diff().dropna().mode()
            if len(freq_mode) > 0 and freq_mode[0] != pd.Timedelta(7, 'D'):
                logger.warning(
                    f"Ground truth may not be weekly: mode frequency = {freq_mode[0]} "
                    f"(expected 7 days)"
                )

        merged = pd.merge(
            stitched_agg[["date", "value"]],
            gt_copy[["date", "value"]],
            on="date",
            suffixes=("_stitched", "_gt"),
        )

        if merged.empty:
            logger.warning(
                f"No overlapping {resolution} periods for MAE calculation. "
                f"Stitched has {len(stitched_agg)} periods, GT has {len(gt_copy)} periods. "
                f"This usually indicates date misalignment."
            )
            return (float("inf"), float("inf")) if return_nmae else float("inf")

        # Calculate MAE
        mae = np.abs(merged["value_stitched"] - merged["value_gt"]).mean()

        # Calculate NMAE if requested
        if return_nmae:
            mean_gt = merged["value_gt"].mean()
            nmae = mae / mean_gt if mean_gt > 1e-6 else float("inf")
            return (mae, nmae)

        return mae

    @staticmethod
    def calculate_mape(
        stitched: pd.DataFrame,
        ground_truth: pd.DataFrame,
        resolution: str = "monthly",
        epsilon: float = 1.0,
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        .. deprecated::
            MAPE is deprecated for this use case as it is undefined for zero values
            and not meaningful for normalized index data. Use NMAE (Normalized MAE)
            or Correlation instead for validation metrics.

        Args:
            stitched: Stitched daily series
            ground_truth: Ground truth (monthly or weekly)
            resolution: Aggregation resolution
            epsilon: Small constant to avoid division by zero

        Returns:
            MAPE value in percentage
        """
        warnings.warn(
            "calculate_mape() is deprecated. MAPE is not appropriate for normalized "
            "index data and is undefined for zero values. Use NMAE (Normalized MAE) "
            "or Correlation for validation metrics.",
            DeprecationWarning,
            stacklevel=2
        )

        # Aggregate stitched to resolution
        stitched_copy = stitched.copy()
        stitched_copy["date"] = pd.to_datetime(stitched_copy["date"])

        if resolution == "monthly":
            stitched_copy["period"] = stitched_copy["date"].dt.to_period("M")
            stitched_agg = stitched_copy.groupby("period")["value"].sum().reset_index()
            stitched_agg["date"] = stitched_agg["period"].dt.to_timestamp()
        elif resolution == "weekly":
            # FIXED: Use resample('W-SUN') to produce Sunday dates (not Monday)
            stitched_copy = stitched_copy.set_index("date")
            stitched_agg = stitched_copy.resample("W-SUN")["value"].sum().reset_index()

            # Verify Sunday alignment
            if not stitched_agg.empty:
                assert (stitched_agg["date"].dt.dayofweek == 6).all(), \
                    f"Weekly aggregation produced non-Sunday dates: {stitched_agg['date'].dt.day_name().unique()}"
        else:
            raise ValueError(f"Invalid resolution: {resolution}")

        # Merge with ground truth
        gt_copy = ground_truth.copy()
        gt_copy["date"] = pd.to_datetime(gt_copy["date"])

        merged = pd.merge(
            stitched_agg[["date", "value"]],
            gt_copy[["date", "value"]],
            on="date",
            suffixes=("_stitched", "_gt"),
        )

        if merged.empty:
            logger.warning(
                f"No overlapping {resolution} periods for MAPE calculation. "
                f"This usually indicates date misalignment."
            )
            return float("inf")

        mape = 100 * np.mean(
            np.abs(merged["value_stitched"] - merged["value_gt"]) / (merged["value_gt"] + epsilon)
        )
        return mape


class CVTester:
    """
    Temporal cross-validation tester for stitching methods.

    Implements time series cross-validation with train/test/gap periods
    to assess temporal generalization performance.
    """

    def __init__(self, train_months: int = 24, test_months: int = 6, gap_months: int = 3):
        """
        Initialize CVTester.

        Args:
            train_months: Number of months in training period
            test_months: Number of months in test period
            gap_months: Gap between train and test to prevent leakage
        """
        self.train_months = train_months
        self.test_months = test_months
        self.gap_months = gap_months

    def split_temporal(
        self,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Split data into train/test periods for temporal CV.

        Args:
            daily_chunks: List of daily chunk DataFrames
            monthly_data: Monthly ground truth
            weekly_data: Weekly ground truth (optional)

        Returns:
            Dictionary with train/test splits and metadata
        """
        # Get date range (optimized to avoid materializing all dates)
        min_date = None
        max_date = None

        for chunk in daily_chunks:
            if not chunk.empty:
                chunk_dates = pd.to_datetime(chunk["date"])
                chunk_min = chunk_dates.min()
                chunk_max = chunk_dates.max()

                min_date = chunk_min if min_date is None else min(min_date, chunk_min)
                max_date = chunk_max if max_date is None else max(max_date, chunk_max)

        if min_date is None or max_date is None:
            raise ValueError("No dates found in daily chunks")

        logger.info(f"Data range: {min_date.date()} to {max_date.date()}")

        # Calculate split points
        train_end = min_date + pd.DateOffset(months=self.train_months)
        gap_end = train_end + pd.DateOffset(months=self.gap_months)
        test_end = gap_end + pd.DateOffset(months=self.test_months)

        logger.info(
            f"CV splits: Train={min_date.date()} to {train_end.date()}, "
            f"Gap={train_end.date()} to {gap_end.date()}, "
            f"Test={gap_end.date()} to {test_end.date()}"
        )

        # Check if we have enough data
        if test_end > max_date:
            logger.warning(
                f"Test period extends beyond available data. "
                f"Test end={test_end.date()}, Data end={max_date.date()}"
            )
            test_end = max_date

        # Split daily chunks
        train_chunks = []
        test_chunks = []

        for chunk in daily_chunks:
            if chunk.empty:
                continue

            chunk_copy = chunk.copy()
            chunk_copy["date"] = pd.to_datetime(chunk_copy["date"])

            # Train period
            train_mask = chunk_copy["date"] < train_end
            if train_mask.any():
                train_chunks.append(chunk_copy[train_mask].reset_index(drop=True))

            # Test period (after gap)
            test_mask = (chunk_copy["date"] >= gap_end) & (chunk_copy["date"] <= test_end)
            if test_mask.any():
                test_chunks.append(chunk_copy[test_mask].reset_index(drop=True))

        # Split monthly data
        monthly_data = monthly_data.copy()
        monthly_data["date"] = pd.to_datetime(monthly_data["date"])

        train_monthly = monthly_data[monthly_data["date"] < train_end].reset_index(drop=True)
        test_monthly = monthly_data[
            (monthly_data["date"] >= gap_end) & (monthly_data["date"] <= test_end)
        ].reset_index(drop=True)

        # Split weekly data (if provided)
        train_weekly = None
        test_weekly = None
        if weekly_data is not None and not weekly_data.empty:
            weekly_data = weekly_data.copy()
            weekly_data["date"] = pd.to_datetime(weekly_data["date"])

            train_weekly = weekly_data[weekly_data["date"] < train_end].reset_index(drop=True)
            test_weekly = weekly_data[
                (weekly_data["date"] >= gap_end) & (weekly_data["date"] <= test_end)
            ].reset_index(drop=True)

        logger.info(
            f"Split results: "
            f"Train chunks={len(train_chunks)}, Test chunks={len(test_chunks)}, "
            f"Train monthly={len(train_monthly)}, Test monthly={len(test_monthly)}"
        )

        # Validate no train/test leakage
        if train_chunks and test_chunks:
            train_dates = set(pd.concat(train_chunks)["date"])
            test_dates = set(pd.concat(test_chunks)["date"])
            overlap = train_dates & test_dates

            if overlap:
                logger.error(
                    f"Train/test leakage detected: {len(overlap)} overlapping dates found. "
                    f"This should not happen with gap_months={self.gap_months}."
                )
                raise ValueError(f"Train/test split has {len(overlap)} overlapping dates")

        return {
            "train_chunks": train_chunks,
            "test_chunks": test_chunks,
            "train_monthly": train_monthly,
            "test_monthly": test_monthly,
            "train_weekly": train_weekly,
            "test_weekly": test_weekly,
            "split_dates": {
                "train_start": min_date,
                "train_end": train_end,
                "gap_end": gap_end,
                "test_end": test_end,
            },
        }

    def evaluate_method(
        self,
        stitcher,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> dict:
        """
        Evaluate a stitching method using temporal cross-validation.

        Args:
            stitcher: StitchingEngine instance
            daily_chunks: List of daily chunks
            monthly_data: Monthly ground truth
            weekly_data: Weekly ground truth (optional)
            config: Configuration object

        Returns:
            Dictionary with CV results and metrics
        """
        logger.info(f"Running temporal CV for {stitcher.name()}")

        # Split data
        splits = self.split_temporal(daily_chunks, monthly_data, weekly_data)

        # Validate splits have sufficient data
        if not splits["train_chunks"] or not splits["test_chunks"]:
            logger.error(
                f"Insufficient data for temporal CV: "
                f"train_chunks={len(splits['train_chunks'])}, "
                f"test_chunks={len(splits['test_chunks'])}"
            )
            return {
                "method": stitcher.name(),
                "error": "Insufficient data for temporal CV",
                "train_metrics": {},
                "test_metrics": {},
                "generalization_gap_nmae": float('nan'),
                "generalization_gap_mae": float('nan'),
                "split_info": {
                    "train_months": self.train_months,
                    "test_months": self.test_months,
                    "gap_months": self.gap_months,
                    "train_chunks": len(splits["train_chunks"]),
                    "test_chunks": len(splits["test_chunks"]),
                },
            }

        # Train on train period
        logger.info("Training on train period...")
        train_result = stitcher.stitch(
            splits["train_chunks"],
            splits["train_monthly"],
            splits["train_weekly"],
            config,
        )

        # Evaluate on test period
        logger.info("Evaluating on test period...")
        test_result = stitcher.stitch(
            splits["test_chunks"],
            splits["test_monthly"],
            splits["test_weekly"],
            config,
        )

        # Calculate metrics
        cv_metrics = {
            "method": stitcher.name(),
            "train_metrics": {
                "monthly_mae": train_result.diagnostics.get("monthly_mae"),
                "monthly_nmae": train_result.diagnostics.get("monthly_nmae"),
                "weekly_mae": train_result.diagnostics.get("weekly_mae"),
                "weekly_nmae": train_result.diagnostics.get("weekly_nmae"),
                "alpha_cv": train_result.diagnostics.get("alpha_cv"),
            },
            "test_metrics": {
                "monthly_mae": test_result.diagnostics.get("monthly_mae"),
                "monthly_nmae": test_result.diagnostics.get("monthly_nmae"),
                "weekly_mae": test_result.diagnostics.get("weekly_mae"),
                "weekly_nmae": test_result.diagnostics.get("weekly_nmae"),
                "alpha_cv": test_result.diagnostics.get("alpha_cv"),
            },
            "split_info": {
                "train_months": self.train_months,
                "test_months": self.test_months,
                "gap_months": self.gap_months,
                "train_chunks": len(splits["train_chunks"]),
                "test_chunks": len(splits["test_chunks"]),
            },
        }

        # Calculate generalization metrics (MAE)
        train_mae = cv_metrics["train_metrics"].get("weekly_mae") or cv_metrics["train_metrics"].get("monthly_mae")
        test_mae = cv_metrics["test_metrics"].get("weekly_mae") or cv_metrics["test_metrics"].get("monthly_mae")

        if train_mae and test_mae:
            cv_metrics["generalization_gap_mae"] = test_mae - train_mae
            # Avoid division by very small values (guard against near-zero denominators)
            if abs(train_mae) < 1e-6:
                cv_metrics["generalization_gap_mae_pct"] = 0.0 if abs(test_mae) < 1e-6 else float('inf')
            else:
                cv_metrics["generalization_gap_mae_pct"] = 100 * (test_mae - train_mae) / train_mae

        # Calculate generalization metrics (NMAE - scale-invariant)
        train_nmae = cv_metrics["train_metrics"].get("weekly_nmae") or cv_metrics["train_metrics"].get("monthly_nmae")
        test_nmae = cv_metrics["test_metrics"].get("weekly_nmae") or cv_metrics["test_metrics"].get("monthly_nmae")

        if train_nmae and test_nmae:
            cv_metrics["generalization_gap_nmae"] = test_nmae - train_nmae
            # NMAE is already normalized, so percentage change is more stable
            if abs(train_nmae) < 1e-6:
                cv_metrics["generalization_gap_nmae_pct"] = 0.0 if abs(test_nmae) < 1e-6 else float('inf')
            else:
                cv_metrics["generalization_gap_nmae_pct"] = 100 * (test_nmae - train_nmae) / train_nmae

            # Check if performance degrades significantly (using NMAE as primary)
            if cv_metrics["generalization_gap_nmae"] > 0.10:  # More than 10% of mean degradation
                logger.warning(
                    f"Significant generalization gap detected: "
                    f"Train NMAE={train_nmae:.3f}, Test NMAE={test_nmae:.3f}, "
                    f"Gap={cv_metrics['generalization_gap_nmae']:.3f} ({cv_metrics['generalization_gap_nmae_pct']:.1f}%)"
                )
                if train_mae and test_mae:
                    logger.info(f"  [MAE context: Train={train_mae:.2f}, Test={test_mae:.2f}]")
            else:
                logger.success(
                    f"Good generalization: "
                    f"Train NMAE={train_nmae:.3f}, Test NMAE={test_nmae:.3f}, "
                    f"Gap={cv_metrics['generalization_gap_nmae']:.3f} ({cv_metrics['generalization_gap_nmae_pct']:.1f}%)"
                )
                if train_mae and test_mae:
                    logger.info(f"  [MAE context: Train={train_mae:.2f}, Test={test_mae:.2f}]")
        elif train_mae and test_mae:
            # Fallback to MAE if NMAE not available
            if cv_metrics["generalization_gap_mae"] > train_mae * 0.02:  # More than 2% relative degradation
                logger.warning(
                    f"Significant generalization gap detected: "
                    f"Train MAE={train_mae:.2f}, Test MAE={test_mae:.2f}, "
                    f"Gap={cv_metrics['generalization_gap_mae']:.2f} ({cv_metrics['generalization_gap_mae_pct']:.1f}%)"
                )
            else:
                logger.success(
                    f"Good generalization: "
                    f"Train MAE={train_mae:.2f}, Test MAE={test_mae:.2f}, "
                    f"Gap={cv_metrics['generalization_gap_mae']:.2f} ({cv_metrics['generalization_gap_mae_pct']:.1f}%)"
                )

        logger.info(f"Temporal CV complete for {stitcher.name()}")

        return cv_metrics
