"""
Utility functions and classes for Google Trends stitching system.
Includes logging setup and file management.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger


class FileManager:
    """
    Centralized file path management for the stitching system.
    Ensures consistent directory structure and handles path creation.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize FileManager with base directory.

        Args:
            base_dir: Root directory of the project. Defaults to current working directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        # Define standard directories
        self.data_raw = self.base_dir / "data" / "raw"
        self.data_daily_chunks = self.data_raw / "daily_chunks"
        self.interim = self.base_dir / "interim"
        self.results = self.base_dir / "results"
        self.reports = self.base_dir / "reports"
        self.logs = self.base_dir / "logs"

    def ensure_directories(self):
        """Create all standard directories if they don't exist."""
        directories = [
            self.data_raw,
            self.data_daily_chunks,
            self.interim,
            self.results,
            self.reports,
            self.logs,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Ensured all directories exist under {self.base_dir}")

    def get_raw_data_path(self, resolution: str) -> Path:
        """
        Get path for raw data file.

        Args:
            resolution: One of 'yearly', 'monthly', 'weekly', 'daily'

        Returns:
            Path to the raw data parquet file
        """
        return self.data_raw / f"{resolution}.parquet"

    def get_daily_chunk_path(self, chunk_index: int) -> Path:
        """
        Get path for a daily chunk file.

        Args:
            chunk_index: Index of the chunk (0-based)

        Returns:
            Path to the chunk parquet file
        """
        return self.data_daily_chunks / f"chunk_{chunk_index:02d}.parquet"

    def get_interim_path(self, method: str, filename: str) -> Path:
        """
        Get path for interim results from a stitching method.

        Args:
            method: Name of the stitching method (e.g., 'baseline', 'hierarchical')
            filename: Name of the file to save

        Returns:
            Path to the interim file
        """
        method_dir = self.interim / method
        method_dir.mkdir(parents=True, exist_ok=True)
        return method_dir / filename

    def get_results_path(self, filename: str) -> Path:
        """
        Get path for final results.

        Args:
            filename: Name of the results file

        Returns:
            Path to the results file
        """
        return self.results / filename

    def get_report_path(self, filename: str) -> Path:
        """
        Get path for reports.

        Args:
            filename: Name of the report file

        Returns:
            Path to the report file
        """
        return self.reports / filename

    def get_log_path(self, name: Optional[str] = None) -> Path:
        """
        Get path for log file.

        Args:
            name: Optional name for the log file. If None, uses timestamp.

        Returns:
            Path to the log file
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"stitching_{timestamp}.log"
        return self.logs / name


def setup_logger(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    console_level: str = "INFO",
):
    """
    Configure loguru logger with console and file outputs.

    Args:
        log_file: Path to log file. If None, only console logging is enabled.
        level: Logging level for file output (DEBUG, INFO, WARNING, ERROR)
        console_level: Logging level for console output
    """
    # Remove default logger
    logger.remove()

    # Add console logger with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=console_level,
        colorize=True,
    )

    # Add file logger if path provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
        )
        logger.info(f"Logging to file: {log_file}")

    return logger


def format_bytes(size: int) -> str:
    """
    Format byte size to human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string (e.g., '1.5 MB')
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def calculate_chunk_parameters(
    total_days: int,
    max_chunk_size: int = 266,
    overlap_days: int = 60,
) -> tuple[int, list[tuple[int, int]]]:
    """
    Calculate optimal chunking parameters for daily data.

    Args:
        total_days: Total number of days to cover
        max_chunk_size: Maximum days per chunk (Google Trends limit ~270)
        overlap_days: Overlap between consecutive chunks

    Returns:
        Tuple of (num_chunks, list of (start_day, end_day) tuples)
    """
    if overlap_days >= max_chunk_size:
        raise ValueError("Overlap must be less than max chunk size")

    effective_chunk_size = max_chunk_size - overlap_days
    num_chunks = (total_days + effective_chunk_size - 1) // effective_chunk_size

    chunks = []
    start = 0
    for i in range(num_chunks):
        end = min(start + max_chunk_size, total_days)
        chunks.append((start, end))
        start += effective_chunk_size

    logger.info(
        f"Calculated {num_chunks} chunks for {total_days} days "
        f"(chunk_size={max_chunk_size}, overlap={overlap_days})"
    )

    return num_chunks, chunks
