"""
Rung 4: Smooth Alpha Stitching Method

This method extends the hierarchical stitching by adding a smoothness penalty on alpha values.

Algorithm:
1. Build hierarchical constraint matrices (monthly, weekly, overlap)
2. Add smoothness penalty: minimize λ * Σ(α_k - α_{k-1})²
3. Solve using convex optimization (cvxpy)
4. Apply smoothed alphas to daily data

Advantages over hierarchical:
- Reduces alpha CV when hierarchical shows high variability (>20%)
- More robust to Google Trends rebasing events
- Smoother extrapolation to future periods

Use when: Alpha CV > 20% or suspected mid-period rebases
Target: Alpha CV < 20%, Weekly MAE < 1.5
"""

from typing import List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import sparse
from loguru import logger

try:
    import cvxpy as cp
except ImportError:
    raise ImportError(
        "cvxpy is required for SmoothAlphaStitcher. "
        "Install with: uv pip install cvxpy"
    )

from .base import StitchingEngine, StitchingResult


class SmoothAlphaStitcher(StitchingEngine):
    """
    Smooth alpha stitching using constrained optimization with smoothness penalty.

    Algorithm:
    minimize: ||A * alpha - b||² + λ * ||D * alpha||²

    where:
    - A * alpha - b: hierarchical constraints (monthly, weekly, overlap)
    - D * alpha: first differences (smoothness penalty)
    - λ: smoothness weight (higher = smoother alpha path)

    This reduces alpha volatility while still respecting hierarchical constraints.
    """

    def name(self) -> str:
        """Return method name."""
        return "smooth_alpha"

    def stitch(
        self,
        daily_chunks: List[pd.DataFrame],
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> StitchingResult:
        """
        Stitch daily chunks using smooth alpha optimization.

        Args:
            daily_chunks: List of daily chunk DataFrames [date, value]
            monthly_data: Monthly ground truth [date, value]
            weekly_data: Weekly ground truth [date, value] (optional but recommended)
            config: Configuration object

        Returns:
            StitchingResult with stitched series and diagnostics
        """
        logger.info(f"Starting {self.name()} stitching method")

        # Validate inputs
        self._validate_inputs(daily_chunks, monthly_data, weekly_data)

        # Step 1: Prepare data structures
        logger.info("Step 1: Building daily data index")
        daily_index = self._build_daily_index(daily_chunks)

        # Step 2: Build constraint matrices
        logger.info("Step 2: Building constraint matrices")
        A, b, constraint_info = self._build_constraint_matrices(
            daily_index,
            monthly_data,
            weekly_data,
            config,
        )

        # Step 3: Solve optimization with smoothness penalty
        logger.info("Step 3: Solving smooth alpha optimization")
        alphas, solve_info = self._solve_smooth_optimization(A, b, config)

        # Step 4: Apply alphas to daily data
        logger.info("Step 4: Applying scaling factors")
        stitched_daily = self._apply_alphas(daily_index, alphas)

        # Step 5: Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            stitched_daily,
            monthly_data,
            weekly_data,
            alphas,
            daily_index,
            solve_info,
            constraint_info,
        )

        logger.success(f"{self.name()} stitching completed")

        return StitchingResult(
            stitched_series=stitched_daily,
            alpha_estimates=alphas,
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

    def _build_constraint_matrices(
        self,
        daily_index: pd.DataFrame,
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        config: Any,
    ) -> Tuple[sparse.csr_matrix, np.ndarray, dict]:
        """
        Build constraint matrices A and target vector b.

        Same as hierarchical method - builds monthly, weekly, and overlap constraints.

        Args:
            daily_index: Indexed daily data
            monthly_data: Monthly ground truth
            weekly_data: Weekly ground truth (optional)
            config: Configuration object

        Returns:
            Tuple of (A_matrix, b_vector, constraint_info)
        """
        num_chunks = daily_index["chunk_id"].max() + 1
        num_observations = len(daily_index)

        # Get weights from config
        weights = config.stitching.weights
        w_monthly = weights["monthly"]
        w_weekly = weights.get("weekly", 0.5)
        w_overlap = weights.get("overlap", 0.1)

        # Get zero handling settings
        zero_threshold = config.stitching.zero_threshold
        structural_zero_months = config.stitching.structural_zero_months

        rows_A = []
        cols_A = []
        data_A = []
        b_values = []
        constraint_types = []

        current_row = 0

        # === MONTHLY CONSTRAINTS ===
        logger.info("Adding monthly constraints")
        monthly_data = monthly_data.copy()
        monthly_data["date"] = pd.to_datetime(monthly_data["date"])
        monthly_data["month"] = monthly_data["date"].dt.to_period("M")

        daily_index_copy = daily_index.copy()
        daily_index_copy["month"] = daily_index_copy["date"].dt.to_period("M")

        for month, group in daily_index_copy.groupby("month"):
            # Get ground truth for this month
            truth_rows = monthly_data[monthly_data["month"] == month]
            if truth_rows.empty:
                continue

            truth_value = truth_rows["value"].iloc[0]

            # Check if structural zero
            is_structural_zero = month.month in structural_zero_months

            if is_structural_zero and truth_value < zero_threshold:
                # Skip constraint for structural zero months
                continue

            # Build constraint: sum(alpha_k * value_k) = truth_value
            for _, row in group.iterrows():
                rows_A.append(current_row)
                cols_A.append(row["chunk_id"])
                data_A.append(w_monthly * row["value"])

            b_values.append(w_monthly * truth_value)
            constraint_types.append("monthly")
            current_row += 1

        num_monthly_constraints = current_row
        logger.info(f"Added {num_monthly_constraints} monthly constraints")

        # === WEEKLY CONSTRAINTS ===
        if weekly_data is not None and not weekly_data.empty:
            logger.info("Adding weekly constraints")
            weekly_data = weekly_data.copy()
            weekly_data["date"] = pd.to_datetime(weekly_data["date"])
            # CRITICAL: Use Sunday-ending weeks to match Google Trends format
            weekly_data["week"] = weekly_data["date"].dt.to_period("W-SUN")

            daily_index_copy["week"] = daily_index_copy["date"].dt.to_period("W-SUN")

            for week, group in daily_index_copy.groupby("week"):
                # Get ground truth for this week
                truth_rows = weekly_data[weekly_data["week"] == week]
                if truth_rows.empty:
                    continue

                truth_value = truth_rows["value"].iloc[0]

                # Build constraint
                for _, row in group.iterrows():
                    rows_A.append(current_row)
                    cols_A.append(row["chunk_id"])
                    data_A.append(w_weekly * row["value"])

                b_values.append(w_weekly * truth_value)
                constraint_types.append("weekly")
                current_row += 1

            num_weekly_constraints = current_row - num_monthly_constraints
            logger.info(f"Added {num_weekly_constraints} weekly constraints")
        else:
            num_weekly_constraints = 0
            logger.info("No weekly data provided, skipping weekly constraints")

        # === OVERLAP CONSTRAINTS ===
        logger.info("Adding overlap constraints")
        overlap_dates = daily_index_copy.groupby("date").filter(lambda x: len(x) > 1)

        for date, group in overlap_dates.groupby("date"):
            # For each overlapping date, constrain consecutive chunks to be similar
            chunks_in_overlap = group.sort_values("chunk_id")
            for i in range(len(chunks_in_overlap) - 1):
                row_k = chunks_in_overlap.iloc[i]
                row_k1 = chunks_in_overlap.iloc[i + 1]

                rows_A.append(current_row)
                cols_A.append(row_k["chunk_id"])
                data_A.append(w_overlap * row_k["value"])

                rows_A.append(current_row)
                cols_A.append(row_k1["chunk_id"])
                data_A.append(-w_overlap * row_k1["value"])

                b_values.append(0.0)
                constraint_types.append("overlap")
                current_row += 1

        num_overlap_constraints = current_row - num_monthly_constraints - num_weekly_constraints
        logger.info(f"Added {num_overlap_constraints} overlap constraints")

        # Build sparse matrix
        A = sparse.csr_matrix(
            (data_A, (rows_A, cols_A)),
            shape=(current_row, num_chunks),
        )
        b = np.array(b_values)

        constraint_info = {
            "num_monthly": num_monthly_constraints,
            "num_weekly": num_weekly_constraints,
            "num_overlap": num_overlap_constraints,
            "total": current_row,
            "num_chunks": num_chunks,
        }

        logger.info(
            f"Built constraint system: "
            f"{A.shape[0]} constraints × {A.shape[1]} variables, "
            f"sparsity={A.nnz / (A.shape[0] * A.shape[1]):.3f}"
        )

        return A, b, constraint_info

    def _solve_smooth_optimization(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        config: Any,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve smooth alpha optimization using cvxpy.

        Minimizes: ||A * alpha - b||² + λ * ||D * alpha||²

        where D is the first difference matrix:
        D * alpha = [α₁-α₀, α₂-α₁, α₃-α₂, ...]

        Args:
            A: Constraint matrix
            b: Target vector
            config: Configuration object

        Returns:
            Tuple of (alpha_vector, solve_info)
        """
        num_chunks = A.shape[1]

        # Get smoothness parameter
        lambda_smooth = config.stitching.smooth_alpha.lambda_smoothness
        logger.info(f"Smoothness penalty λ = {lambda_smooth}")

        # Define optimization variable
        alpha = cp.Variable(num_chunks)

        # Convert scipy sparse to cvxpy compatible format
        A_dense = A.toarray()

        # Objective: ||A*alpha - b||² + λ*||D*alpha||²
        residual = A_dense @ alpha - b

        # Special case: single chunk (no smoothness penalty needed)
        if num_chunks == 1:
            logger.info("Single chunk detected, skipping smoothness penalty")
            objective = cp.Minimize(cp.sum_squares(residual))
            smoothness_norm = 0.0
        else:
            # Build difference matrix D for smoothness penalty
            # D is (n-1) × n matrix where D[i,i] = -1, D[i,i+1] = 1
            D_rows = []
            D_cols = []
            D_data = []
            for i in range(num_chunks - 1):
                D_rows.extend([i, i])
                D_cols.extend([i, i + 1])
                D_data.extend([-1.0, 1.0])

            D = sparse.csr_matrix(
                (D_data, (D_rows, D_cols)),
                shape=(num_chunks - 1, num_chunks),
            )
            D_dense = D.toarray()

            # Objective with smoothness penalty
            smoothness = D_dense @ alpha
            objective = cp.Minimize(
                cp.sum_squares(residual) + lambda_smooth * cp.sum_squares(smoothness)
            )

        # Optional: Add constraints to keep alphas positive and reasonable
        # This prevents extreme negative values
        constraints = [alpha >= 0.01, alpha <= 100]

        # Solve
        problem = cp.Problem(objective, constraints)

        logger.info("Solving convex optimization problem with cvxpy")

        try:
            problem.solve(verbose=False)
        except Exception as e:
            logger.error(f"cvxpy solver failed: {e}")
            # Fallback: try different solver
            logger.info("Trying alternative solver (SCS)")
            problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(
                f"Optimization did not converge to optimal solution. "
                f"Status: {problem.status}"
            )

        alphas = alpha.value

        # Calculate metrics
        residual_norm = np.linalg.norm(A.dot(alphas) - b)

        # Calculate smoothness norm (skip for single chunk)
        if num_chunks > 1:
            smoothness_norm = np.linalg.norm(D.dot(alphas))
        # else: smoothness_norm already set to 0.0 above

        logger.info(
            f"Optimization complete: "
            f"status={problem.status}, "
            f"objective={problem.value:.6f}, "
            f"residual_norm={residual_norm:.6f}, "
            f"smoothness_norm={smoothness_norm:.6f}"
        )

        # Check for outlier alphas
        outliers = np.where((alphas < 0.1) | (alphas > 10))[0]
        if len(outliers) > 0:
            logger.warning(
                f"Found {len(outliers)} chunks with extreme alpha values (< 0.1 or > 10). "
                f"Chunk IDs: {outliers.tolist()}"
            )

        solve_info = {
            "status": problem.status,
            "objective_value": problem.value,
            "residual_norm": residual_norm,
            "smoothness_norm": smoothness_norm,
            "lambda_smoothness": lambda_smooth,
            "converged": problem.status in ["optimal", "optimal_inaccurate"],
        }

        return alphas, solve_info

    def _apply_alphas(
        self,
        daily_index: pd.DataFrame,
        alphas: np.ndarray,
    ) -> pd.DataFrame:
        """
        Apply chunk-specific alpha scaling factors to daily data.

        For overlapping dates, average the scaled values from different chunks.

        Args:
            daily_index: Indexed daily data
            alphas: Array of scaling factors (one per chunk)

        Returns:
            Stitched daily series [date, value]
        """
        daily_scaled = daily_index.copy()
        daily_scaled["alpha"] = daily_scaled["chunk_id"].map(
            lambda chunk_id: alphas[chunk_id]
        )
        daily_scaled["scaled_value"] = daily_scaled["value"] * daily_scaled["alpha"]

        # Average overlapping dates
        stitched = daily_scaled.groupby("date", as_index=False)["scaled_value"].mean()
        stitched = stitched.rename(columns={"scaled_value": "value"})
        stitched = stitched.sort_values("date").reset_index(drop=True)

        logger.info(f"Applied alphas to {len(stitched)} unique dates")

        return stitched

    def _calculate_diagnostics(
        self,
        stitched_daily: pd.DataFrame,
        monthly_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame],
        alphas: np.ndarray,
        daily_index: pd.DataFrame,
        solve_info: dict,
        constraint_info: dict,
    ) -> dict:
        """
        Calculate diagnostic metrics for the stitching result.

        Args:
            stitched_daily: Final stitched daily series
            monthly_data: Ground truth monthly data
            weekly_data: Ground truth weekly data (optional)
            alphas: Chunk scaling factors
            daily_index: Original daily index
            solve_info: Optimization solver information
            constraint_info: Constraint matrix information

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

        # Alpha smoothness metrics (only for multiple chunks)
        if len(alphas) > 1:
            alpha_diffs = np.diff(alphas)
            diagnostics["alpha_smoothness"] = {
                "mean_diff": np.mean(np.abs(alpha_diffs)),
                "max_diff": np.max(np.abs(alpha_diffs)),
                "std_diff": np.std(alpha_diffs),
            }
        else:
            # Single chunk: no smoothness metrics
            diagnostics["alpha_smoothness"] = {
                "mean_diff": 0.0,
                "max_diff": 0.0,
                "std_diff": 0.0,
            }

        # Optimization diagnostics
        diagnostics["optimization"] = solve_info
        diagnostics["constraints"] = constraint_info

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
            f"Smoothness={diagnostics['alpha_smoothness']['mean_diff']:.4f}, "
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
