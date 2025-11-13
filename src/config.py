"""
Configuration management for Google Trends stitching system.
Loads and validates config.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import yaml
from dotenv import load_dotenv
from loguru import logger


@dataclass
class DateRange:
    """Date range configuration."""

    start: str
    end: str

    def __post_init__(self):
        """Validate date format."""
        try:
            datetime.strptime(self.start, "%Y-%m-%d")
            datetime.strptime(self.end, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    @property
    def start_date(self) -> datetime:
        """Get start date as datetime object."""
        return datetime.strptime(self.start, "%Y-%m-%d")

    @property
    def end_date(self) -> datetime:
        """Get end date as datetime object."""
        return datetime.strptime(self.end, "%Y-%m-%d")

    @property
    def total_days(self) -> int:
        """Calculate total days in the range."""
        return (self.end_date - self.start_date).days + 1


@dataclass
class DailyConfig:
    """Daily chunk configuration."""

    overlap_days: int = 60

    def __post_init__(self):
        """Validate overlap days."""
        if not 30 <= self.overlap_days <= 180:
            raise ValueError(f"overlap_days must be in [30, 180], got {self.overlap_days}")


@dataclass
class SmoothAlphaConfig:
    """Smooth alpha stitcher configuration (Rung 4)."""

    lambda_smoothness: float = 0.1

    def __post_init__(self):
        """Validate smooth alpha parameters."""
        if self.lambda_smoothness <= 0 or self.lambda_smoothness > 100:
            raise ValueError(
                f"lambda_smoothness must be in (0, 100], got {self.lambda_smoothness}"
            )


@dataclass
class StateSpaceConfig:
    """State-space stitcher configuration (Rung 5)."""

    process_noise: float = 0.01
    observation_noise: float = 0.05
    confidence_level: float = 0.95

    def __post_init__(self):
        """Validate state-space parameters."""
        if self.process_noise <= 0:
            raise ValueError(f"process_noise must be positive, got {self.process_noise}")
        if self.observation_noise <= 0:
            raise ValueError(
                f"observation_noise must be positive, got {self.observation_noise}"
            )
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )


@dataclass
class StitchingConfig:
    """Stitching method configuration."""

    weights: Dict[str, float]
    zero_threshold: float
    structural_zero_months: List[int]
    max_iterations: int
    tolerance: float
    smooth_alpha: Optional[SmoothAlphaConfig] = None
    state_space: Optional[StateSpaceConfig] = None

    def __post_init__(self):
        """Validate stitching parameters."""
        required_weights = ["monthly", "weekly", "overlap"]
        for weight in required_weights:
            if weight not in self.weights:
                raise ValueError(f"Missing required weight: {weight}")

        if self.zero_threshold <= 0 or self.zero_threshold > 0.1:
            raise ValueError(f"zero_threshold must be in (0, 0.1], got {self.zero_threshold}")

        if self.max_iterations < 100:
            logger.warning(
                f"max_iterations={self.max_iterations} is low. "
                "Recommendation: use 1000+ for 9+ chunks"
            )


@dataclass
class SerpAPIConfig:
    """SerpAPI configuration."""

    timeout: int
    max_retries: int
    retry_delay: int

    def __post_init__(self):
        """Validate SerpAPI parameters."""
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")


@dataclass
class OutputConfig:
    """Output configuration."""

    save_plots: bool
    plot_format: str
    plot_dpi: int

    def __post_init__(self):
        """Validate output parameters."""
        valid_formats = ["png", "jpg", "svg", "pdf"]
        if self.plot_format not in valid_formats:
            raise ValueError(f"plot_format must be one of {valid_formats}, got {self.plot_format}")


@dataclass
class TemporalCVConfig:
    """Temporal cross-validation configuration."""

    train_months: int
    test_months: int
    gap_months: int

    def __post_init__(self):
        """Validate temporal CV parameters."""
        if self.train_months <= 0:
            raise ValueError(f"train_months must be positive, got {self.train_months}")
        if self.test_months <= 0:
            raise ValueError(f"test_months must be positive, got {self.test_months}")
        if self.gap_months < 0:
            raise ValueError(f"gap_months must be non-negative, got {self.gap_months}")


@dataclass
class RobustnessConfig:
    """Robustness testing configuration for Phase 3."""

    overlap_tests: List[int]
    weight_tests: List[List[float]]
    temporal_cv: TemporalCVConfig

    def __post_init__(self):
        """Validate robustness testing parameters."""
        if not self.overlap_tests:
            raise ValueError("overlap_tests must have at least one value")

        for overlap in self.overlap_tests:
            if overlap < 30 or overlap > 180:
                logger.warning(f"overlap_tests value {overlap} outside recommended range [30, 180]")

        if not self.weight_tests:
            raise ValueError("weight_tests must have at least one combination")

        for weights in self.weight_tests:
            if len(weights) != 2:
                raise ValueError(f"Each weight_test must have 2 values [monthly, weekly], got {weights}")
            monthly_w, weekly_w = weights
            if monthly_w <= 0 or weekly_w <= 0:
                raise ValueError(f"Weights must be positive, got {weights}")


class ConfigManager:
    """
    Manages configuration from config.yaml and environment variables.
    """

    def __init__(self, config_path: Optional[Path] = None, env_path: Optional[Path] = None):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to config.yaml. Defaults to ./config.yaml
            env_path: Path to .env file. Defaults to ./.env
        """
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self.env_path = Path(env_path) if env_path else Path(".env")

        # Load environment variables
        if self.env_path.exists():
            load_dotenv(self.env_path)
            logger.info(f"Loaded environment variables from {self.env_path}")
        else:
            logger.warning(f"Environment file not found: {self.env_path}")

        # Load config.yaml
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.raw_config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")

        # Parse configuration
        self._parse_config()

    def _parse_config(self):
        """Parse and validate configuration."""
        # Basic parameters
        self.search_term: str = self.raw_config.get("search_term", "")
        if not self.search_term:
            raise ValueError("search_term is required in config.yaml")

        self.geo: str = self.raw_config.get("geo", "US")

        self.random_seed: int = self.raw_config.get("random_seed", 42)

        # Date range
        date_range_raw = self.raw_config.get("date_range", {})
        self.date_range = DateRange(
            start=date_range_raw.get("start", ""),
            end=date_range_raw.get("end", ""),
        )

        # Daily chunk config
        daily_raw = self.raw_config.get("daily", {})
        self.daily = DailyConfig(overlap_days=daily_raw.get("overlap_days", 60))

        # Stitching config
        stitching_raw = self.raw_config.get("stitching", {})

        # Parse nested smooth_alpha config
        smooth_alpha_raw = stitching_raw.get("smooth_alpha", {})
        smooth_alpha_config = None
        if smooth_alpha_raw:
            smooth_alpha_config = SmoothAlphaConfig(
                lambda_smoothness=smooth_alpha_raw.get("lambda_smoothness", 0.1)
            )

        # Parse nested state_space config
        state_space_raw = stitching_raw.get("state_space", {})
        state_space_config = None
        if state_space_raw:
            state_space_config = StateSpaceConfig(
                process_noise=state_space_raw.get("process_noise", 0.01),
                observation_noise=state_space_raw.get("observation_noise", 0.05),
                confidence_level=state_space_raw.get("confidence_level", 0.95),
            )

        self.stitching = StitchingConfig(
            weights=stitching_raw.get("weights", {}),
            zero_threshold=stitching_raw.get("zero_threshold", 0.01),
            structural_zero_months=stitching_raw.get("structural_zero_months", []),
            max_iterations=stitching_raw.get("max_iterations", 1000),
            tolerance=stitching_raw.get("tolerance", 1.0e-8),
            smooth_alpha=smooth_alpha_config,
            state_space=state_space_config,
        )

        # SerpAPI config
        serpapi_raw = self.raw_config.get("serpapi", {})
        self.serpapi = SerpAPIConfig(
            timeout=serpapi_raw.get("timeout", 30),
            max_retries=serpapi_raw.get("max_retries", 3),
            retry_delay=serpapi_raw.get("retry_delay", 2),
        )

        # Output config
        output_raw = self.raw_config.get("output", {})
        self.output = OutputConfig(
            save_plots=output_raw.get("save_plots", True),
            plot_format=output_raw.get("plot_format", "png"),
            plot_dpi=output_raw.get("plot_dpi", 300),
        )

        # Robustness testing config
        robustness_raw = self.raw_config.get("robustness", {})
        temporal_cv_raw = robustness_raw.get("temporal_cv", {})
        self.robustness = RobustnessConfig(
            overlap_tests=robustness_raw.get("overlap_tests", [30, 60, 90]),
            weight_tests=robustness_raw.get("weight_tests", [[1.0, 0.3], [1.0, 0.5], [1.0, 0.7]]),
            temporal_cv=TemporalCVConfig(
                train_months=temporal_cv_raw.get("train_months", 24),
                test_months=temporal_cv_raw.get("test_months", 6),
                gap_months=temporal_cv_raw.get("gap_months", 3),
            ),
        )

        logger.success("Configuration validated successfully")

    def get_api_key(self, key_name: str = "SERPAPI_KEY") -> str:
        """
        Get API key from environment variables.

        Args:
            key_name: Name of the environment variable

        Returns:
            API key value

        Raises:
            ValueError: If API key is not found
        """
        key_value = os.getenv(key_name)
        if not key_value:
            raise ValueError(
                f"{key_name} not found in environment variables. "
                f"Please add it to {self.env_path}"
            )
        return key_value

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration from environment variables.

        Returns:
            Dictionary with LLM configuration
        """
        return {
            "api_key": os.getenv("LLM_API_KEY"),
            "model": os.getenv("LLM_MODEL", "moonshot-v1-128k"),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "retry_attempts": int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
            "base_url": os.getenv("LLM_BASE_URL", "https://api.moonshot.ai/v1"),
        }

    def summary(self) -> str:
        """
        Generate configuration summary.

        Returns:
            Formatted configuration summary string
        """
        summary = f"""
Configuration Summary
=====================
Search Term: {self.search_term}
Geographic Location: {self.geo}
Date Range: {self.date_range.start} to {self.date_range.end} ({self.date_range.total_days} days)
Overlap: {self.daily.overlap_days} days
Random Seed: {self.random_seed}

Stitching Weights:
  Monthly: {self.stitching.weights['monthly']}
  Weekly: {self.stitching.weights['weekly']}
  Overlap: {self.stitching.weights['overlap']}

Zero Handling:
  Threshold: {self.stitching.zero_threshold}
  Structural Zero Months: {self.stitching.structural_zero_months}

Optimization:
  Max Iterations: {self.stitching.max_iterations}
  Tolerance: {self.stitching.tolerance}

SerpAPI:
  Timeout: {self.serpapi.timeout}s
  Max Retries: {self.serpapi.max_retries}
  Retry Delay: {self.serpapi.retry_delay}s
        """
        return summary.strip()
