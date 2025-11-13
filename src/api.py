"""
Google Trends API wrapper using SerpAPI.
Handles data fetching with retry logic and rate limiting.
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd
from loguru import logger


class TrendsAPIError(Exception):
    """Custom exception for Trends API errors."""

    pass


class TrendsAPI:
    """
    Wrapper for Google Trends data via SerpAPI.
    Handles retries, rate limiting, and data formatting.
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        """
        Initialize TrendsAPI.

        Args:
            api_key: SerpAPI key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = "https://serpapi.com/search"

        logger.info("TrendsAPI initialized")

    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with retry logic.

        Args:
            params: Request parameters

        Returns:
            API response as dictionary

        Raises:
            TrendsAPIError: If request fails after all retries
        """
        params["api_key"] = self.api_key

        # Log request parameters for debugging (hide API key)
        debug_params = {k: v for k, v in params.items() if k != "api_key"}
        debug_params["api_key"] = "***" if "api_key" in params else "MISSING"
        logger.debug(f"API request parameters: {debug_params}")

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limit hit (429). Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue

                # Handle authentication errors
                if response.status_code == 401:
                    raise TrendsAPIError(
                        "Authentication failed (401). Check your SERPAPI_KEY in .env"
                    )

                # Raise for other HTTP errors
                response.raise_for_status()

                data = response.json()

                # Check for API errors in response
                if "error" in data:
                    raise TrendsAPIError(f"API error: {data['error']}")

                logger.debug(f"API request successful: {params.get('q')}")
                return data

            except requests.exceptions.Timeout:
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request timeout. Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)

            except TrendsAPIError:
                # Re-raise our custom errors immediately
                raise

            except requests.exceptions.RequestException as e:
                # Log detailed error information
                error_details = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_json = e.response.json()
                        error_details = f"{e} | Response: {error_json}"
                    except:
                        error_details = f"{e} | Status: {e.response.status_code} | Text: {e.response.text[:200]}"

                logger.error(f"Request error details: {error_details}")

                if attempt == self.max_retries - 1:
                    raise TrendsAPIError(f"Request failed after {self.max_retries} attempts: {error_details}")

                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed: {error_details}. Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)

        raise TrendsAPIError(f"Request failed after {self.max_retries} attempts")

    def fetch(
        self,
        search_term: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        geo: str = "US",
    ) -> pd.DataFrame:
        """
        Fetch Google Trends data.

        NOTE: Google Trends automatically determines resolution based on date range:
        - Long range (2004-present) → monthly frequency
        - Medium range (e.g., 3 years) → weekly frequency
        - Short range (≤270 days) → daily frequency

        Args:
            search_term: Search query
            start_date: Start date in YYYY-MM-DD format (optional, defaults to 2004-01-01)
            end_date: End date in YYYY-MM-DD format (optional, defaults to today)
            geo: Geographic location code (default: US)

        Returns:
            DataFrame with columns: date, value, is_partial

        Raises:
            TrendsAPIError: If fetch fails
        """
        params = {
            "engine": "google_trends",
            "q": search_term,
            "data_type": "TIMESERIES",
            "geo": geo,
        }

        # Add date range if specified
        if start_date and end_date:
            params["date"] = f"{start_date} {end_date}"

        date_range_str = f"{start_date or '2004-01-01'} to {end_date or 'today'}"
        logger.info(f"Fetching data for '{search_term}' ({date_range_str})")

        response = self._make_request(params)

        # Parse response
        if "interest_over_time" not in response:
            raise TrendsAPIError("No interest_over_time data in response")

        timeline_data = response["interest_over_time"]["timeline_data"]

        if not timeline_data:
            logger.warning(f"No data returned for {search_term}")
            return pd.DataFrame(columns=["date", "value", "is_partial"])

        # Convert to DataFrame
        records = []
        for item in timeline_data:
            date_str = item.get("date")
            values = item.get("values", [])

            if not values:
                continue

            # Get first value (assumes single search term)
            value_data = values[0]
            value = value_data.get("value", 0)
            extracted_value = value_data.get("extracted_value", value)

            # Parse date - try multiple formats
            date = None
            original_date_str = date_str

            # First, try parsing as ISO format (YYYY-MM-DD) directly
            # This prevents confusion with hyphens in date ranges
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass

            # If not ISO format, handle date ranges
            if date is None:
                # Handle date ranges (e.g., "Jan 15 – 21, 2024" or "Dec 26, 2021 – Jan 1, 2022")
                # Try multiple dash characters (en-dash, em-dash, hyphen) with various whitespace
                # Note: SerpAPI uses thin spaces (\u2009) around dashes
                is_range = False
                for separator in ["\u2009\u2013\u2009", " \u2013 ", "\u2009\u2014\u2009", " \u2014 ", "\u2009-\u2009", " - ", "–", "—"]:
                    if separator in date_str:
                        is_range = True
                        # Split and get first part, then strip ALL Unicode whitespace
                        date_str = date_str.split(separator)[0]
                        # Strip regular spaces, thin spaces, and other Unicode whitespace
                        date_str = date_str.strip().strip('\u2009').strip('\u200a').strip()
                        break

                # If it's a range and the extracted date doesn't have a year,
                # extract year from the original string and append it
                if is_range and "," not in date_str:
                    # Extract year from original string (last 4 digits)
                    year_match = re.search(r'\d{4}', original_date_str)
                    if year_match:
                        year = year_match.group()
                        date_str = f"{date_str}, {year}"

                date_formats = [
                    "%Y",                    # Yearly: "2024"
                    "%b %Y",                 # Monthly: "Jan 2024"
                    "%b %d, %Y",             # Daily/Weekly: "Jan 15, 2024"
                    "%B %d, %Y",             # Full month name: "January 15, 2024"
                ]

                for fmt in date_formats:
                    try:
                        date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue

            if date is None:
                logger.warning(f"Could not parse date '{original_date_str}' (extracted: '{date_str}' repr: {repr(date_str)})")
                continue

            records.append(
                {
                    "date": date,
                    "value": int(extracted_value) if extracted_value else 0,
                    "is_partial": False,  # SerpAPI doesn't provide this flag
                }
            )

        df = pd.DataFrame(records)

        if df.empty:
            logger.warning(f"No valid data parsed for {search_term}")
            return pd.DataFrame(columns=["date", "value", "is_partial"])

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        logger.success(
            f"Fetched {len(df)} data points for '{search_term}' ({date_range_str})"
        )

        return df

    def fetch_historical_monthly(
        self, search_term: str, start_year: int = 2004, end_year: Optional[int] = None, geo: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data (returns monthly frequency).

        For long date ranges (2004-present), Google Trends returns monthly data.

        Args:
            search_term: Search query
            start_year: Start year (default: 2004, when Google Trends started)
            end_year: End year (default: current year)
            geo: Geographic location code (default: US)

        Returns:
            DataFrame with monthly frequency data
        """
        if end_year is None:
            end_year = datetime.now().year

        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        return self.fetch(search_term, start_date, end_date, geo=geo or "US")

    def fetch_weekly(self, search_term: str, start_date: str, end_date: str, geo: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for medium date range (returns weekly frequency).

        For medium date ranges (e.g., 1-3 years), Google Trends returns weekly data.

        Args:
            search_term: Search query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            geo: Geographic location code (default: US)

        Returns:
            DataFrame with weekly frequency data
        """
        return self.fetch(search_term, start_date, end_date, geo=geo or "US")

    def fetch_daily(self, search_term: str, start_date: str, end_date: str, geo: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for short date range (returns daily frequency).

        For short date ranges (≤270 days), Google Trends returns daily data.

        Args:
            search_term: Search query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            geo: Geographic location code (default: US)

        Returns:
            DataFrame with daily frequency data

        Note:
            For longer periods, use fetch_daily_chunks() to split into multiple requests.
        """
        return self.fetch(search_term, start_date, end_date, geo=geo or "US")

    def fetch_daily_chunks(
        self,
        search_term: str,
        start_date: str,
        end_date: str,
        max_chunk_days: int = 266,
        overlap_days: int = 60,
        geo: Optional[str] = None,
    ) -> List[pd.DataFrame]:
        """
        Fetch daily data in overlapping chunks.

        Args:
            search_term: Search query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_chunk_days: Maximum days per chunk (Google Trends limit ~270)
            overlap_days: Overlap between consecutive chunks
            geo: Geographic location code (default: US)

        Returns:
            List of DataFrames, one per chunk
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1

        effective_chunk_size = max_chunk_days - overlap_days
        num_chunks = (total_days + effective_chunk_size - 1) // effective_chunk_size

        logger.info(
            f"Fetching {num_chunks} daily chunks "
            f"(chunk_size={max_chunk_days}, overlap={overlap_days})"
        )

        chunks = []
        current_start = start

        for i in range(num_chunks):
            chunk_end = min(current_start + timedelta(days=max_chunk_days - 1), end)

            chunk_start_str = current_start.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            logger.info(f"Fetching chunk {i}/{num_chunks}: {chunk_start_str} to {chunk_end_str}")

            chunk_df = self.fetch_daily(search_term, chunk_start_str, chunk_end_str, geo=geo)
            chunks.append(chunk_df)

            # Move to next chunk (with overlap)
            current_start += timedelta(days=effective_chunk_size)

            # Stop if we've reached the end
            if current_start > end:
                break

            # Rate limiting: delay between requests to avoid SerpAPI/Google Trends errors
            # Google Trends can be slow to process requests, especially for daily data
            delay = 10  # 10 seconds between chunks
            logger.info(f"Waiting {delay}s before next request (rate limiting)...")
            time.sleep(delay)

        logger.success(f"Fetched {len(chunks)} daily chunks successfully")
        return chunks
