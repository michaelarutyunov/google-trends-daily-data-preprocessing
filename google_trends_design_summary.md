# Implementation Design Document: Google Trends Stitching System

**Version:** 1.0 | **Date:** November 11, 2025 | **Python:** 3.13 | **Package Manager:** UV  
**Target:** Claude Code | **Environment:** WSL (Ubuntu), 16GB RAM, No GPU

---

## Executive Summary

Complete implementation design for stitching Google Trends daily data across 3 years using hierarchical constrained optimization. System organized into 3 development phases with 6 Jupyter notebooks, modular architecture, and comprehensive error handling.

**Key Outcomes:**
- ✅ **Input:** Search term + date range → **Output:** Valid daily time series (MAE < 3% vs monthly ground truth)
- ✅ **Modularity:** Config-driven (change search term/overlap in YAML, rerun)
- ✅ **Reproducibility:** All interim results saved (Parquet + Pickle)
- ✅ **Extensibility:** Abstract base class for stitchers, placeholders for Rung 4-5

---

## Design Decisions Summary

| Decision | Rationale | Alternative Rejected |
|----------|-----------|---------------------|
| **SerpAPI** for data | Stable, documented API | PyTrends (fragile, breaks often) |
| **Jupyter notebooks** (6) | Interactive, checkpoint-driven | Single script (harder to debug) |
| **Parquet for data** | Fast, columnar, interoperable | CSV (slow, large files) |
| **Pickle for models** | Preserves Python objects | JSON (can't serialize numpy) |
| **Config.yaml** | Single source of truth | Hardcoded params (not reusable) |
| **Abstract base for stitchers** | Uniform interface, extensible | Separate functions (code duplication) |
| **UV package manager** | Modern, fast, Python 3.13 | pip+venv (slower, manual) |

---

## System Architecture

```
CONFIG.YAML
     ↓
[Phase 1] DATA ACQUISITION
├─ TrendsAPI → SerpAPI
├─ DataValidator
└─ Output: data/raw/*.parquet

[Phase 2] STITCHING METHODS  
├─ BaselineStitcher
├─ HierarchicalStitcher (RECOMMENDED)
├─ HierarchicalDOWStitcher
└─ Output: interim/{method}/*.parquet/.pkl

[Phase 3] VALIDATION & REPORTING
├─ Validator (MAE computation)
├─ CVTester (robustness battery)
├─ Visualizer (plots)
├─ ReportGenerator
└─ Output: results/*.md/*.csv/*.png
```

**7 Major Components:** ConfigManager, TrendsAPI, DataValidator, StitchingEngine, Validator, Visualizer, ReportGenerator

---

## Phased Development

### Phase 1: Data Collection & Audit (2-3 days)

**Notebook:** `nb/01_data_collection.ipynb`

**What it does:**
1. Load config.yaml (search term, date range, overlap)
2. Fetch yearly data (2004-present) via SerpAPI
3. Fetch monthly, weekly data for specified range
4. Calculate daily chunk allocation:
   ```
   3 years (1,095 days) ÷ (266-day chunks - 60-day overlap) = 9 chunks
   ```
5. Download all 9 daily chunks with progress bar
6. Run data validation:
   - Check completeness, quality
   - Classify zeros (structural vs sampling)
   - Generate audit report

**Outputs:**
- `data/raw/yearly.parquet`
- `data/raw/monthly.parquet`
- `data/raw/weekly.parquet`
- `data/raw/daily_chunks/chunk_00.parquet` ... `chunk_08.parquet`
- `reports/stage0_audit.txt`

**Success Criteria:**
- All data downloaded without errors
- Audit report shows quality acceptable (< 50% zeros, no missing data)
- Daily chunks have correct overlap (verify in audit)

---

### Phase 2: Stitching Methods (4-5 days)

#### Notebook 02: Baseline
- Scale each chunk to its monthly mean
- Average overlap regions
- **Use:** Comparison baseline only (expected MAE ~5%)

#### Notebook 03: Hierarchical (RECOMMENDED)
- Constrained optimization: `minimize ||H@α - y||²`
- H includes monthly + weekly + overlap constraints
- Weights: (1.0, 0.5, 0.1)
- Features:
  - Zero masking (exclude structural zero months 6-8)
  - Intra-chunk rebase detection (overlap residual regression)
  - Convergence diagnostics
- **Target:** MAE < 3%

#### Notebook 04: Hierarchical + DOW
- Remove day-of-week pattern before optimization
- Add pattern back after
- **Decision:** Keep if improvement > 0.5%, else drop

**Outputs per Method:**
```
interim/baseline/stitched_series.parquet
interim/baseline/alpha_estimates.pkl
interim/baseline/diagnostics.pkl

interim/hierarchical/[same structure]
interim/hierarchical_dow/[same structure]
```

**Success Criteria:**
- All 3 methods complete without errors
- PRIMARY METRICS: Weekly Correlation ≥ 0.90, Weekly NMAE < 0.50 (scale-invariant validation)
- SECONDARY METRICS: Weekly MAE < 0.50, Monthly MAE < 3.0 (context)
- DOW improvement evaluated (keep/drop decision made)

---

### Phase 3: Validation & Reporting (3-4 days)

#### Notebook 05: Robustness Testing
**Tests:**
1. **Overlap sensitivity:** Rerun best method with overlaps = [30, 60, 90, 133] days
2. **Weight sensitivity:** Weights = [(1, 0.3), (1, 0.5), (1, 0.7)]
3. **Temporal CV:** Train on months 1-24 (gap 3 months) → test months 31-36

**Success Criteria:**
- Temporal CV NMAE gap < 0.10 (10% of mean - scale-invariant generalization)
- Alpha CV acceptable (chunk quality, ideal < 10% but higher is OK if other metrics pass)

#### Notebook 06: Comparison & Reporting
**Generates:**
- Comparison table (MAE by method)
- Time series plots (all methods overlaid, chunk boundaries marked)
- Residual diagnostics (ACF, QQ, scatter)
- Final markdown report
- Export best method → `results/stitched_series.csv` + `.parquet`

**Success Criteria:**
- Report includes all required sections
- Best method recommended with justification
- Config snapshot saved with results

---

## Configuration (config.yaml)

```yaml
# Search parameters
search_term: "flu vaccine"

# Date range
date_range:
  start: "2022-01-01"
  end: "2024-12-31"

# Daily chunk parameters
daily:
  overlap_days: 60  # [30, 180] - lower = fewer API calls

# Reproducibility (Google samples, so not fully reproducible)
random_seed: 42

# Stitching weights (hierarchical optimization)
stitching:
  weights:
    monthly: 1.0   # Fixed - most reliable
    weekly: 0.5    # [0.3, 0.7]
    overlap: 0.1   # [0.05, 0.2] - soft glue
  
  # Zero handling
  zero_threshold: 0.01  # 1%
  structural_zero_months: [6, 7, 8]  # June-Aug for flu vaccine
  
  # Convergence
  max_iterations: 1000  # IMPORTANT: default 100 too low
  tolerance: 1.0e-8

# SerpAPI config
serpapi:
  timeout: 30
  max_retries: 3
  retry_delay: 2

# Output
output:
  save_plots: true
  plot_format: "png"
  plot_dpi: 300

# Robustness testing (Phase 3)
robustness:
  overlap_tests: [30, 60, 90, 133]
  weight_tests: [[1.0, 0.3], [1.0, 0.5], [1.0, 0.7]]
  temporal_cv:
    train_months: 24
    test_months: 6
    gap_months: 3
```

**To change search term:** Edit `search_term`, rerun all notebooks.  
**To change overlap:** Edit `overlap_days`, rerun from Notebook 01.

---

## Folder Structure

```
google-trends-stitching/
├── .env                    # SERPAPI_KEY=your_key
├── config.yaml
├── pyproject.toml
├── README.md
│
├── nb/                     # Notebooks (run 01 → 06)
│   ├── 01_data_collection.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_hierarchical.ipynb
│   ├── 04_hierarchical_dow.ipynb
│   ├── 05_robustness.ipynb
│   └── 06_comparison.ipynb
│
├── src/                    # Python modules (imported by notebooks)
│   ├── config.py           # ConfigManager
│   ├── api.py              # TrendsAPI
│   ├── validation.py       # DataValidator, Validator, CVTester
│   ├── stitching/
│   │   ├── base.py         # StitchingEngine (abstract)
│   │   ├── baseline.py
│   │   ├── hierarchical.py
│   │   └── hierarchical_dow.py
│   ├── visualization.py
│   ├── reporting.py
│   └── utils.py            # Logger, FileManager
│
├── data/raw/               # API responses (gitignored)
├── interim/                # Method outputs (gitignored)
├── results/                # Final outputs (committed)
├── reports/                # Audit reports
└── logs/                   # Execution logs (gitignored)
```

---

## Dependencies (pyproject.toml)

```toml
[project]
name = "google-trends-stitching"
requires-python = ">=3.13"

dependencies = [
    "pandas>=2.1.0",
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "pyarrow>=14.0.0",
    "requests>=2.31.0",
    "tqdm>=4.66.0",
    "loguru>=0.7.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]

[project.optional-dependencies]
stats = ["statsmodels>=0.14.0"]
viz = ["plotly>=5.18.0"]
dev = ["pytest>=7.4.0", "jupyter>=1.0.0"]
```

**Installation:**
```bash
uv init google-trends-stitching
cd google-trends-stitching
uv pip install -e ".[stats,viz,dev]"
```

---

## Component Interfaces

### StitchingEngine (Abstract Base)

```python
from abc import ABC, abstractmethod

class StitchingEngine(ABC):
    @abstractmethod
    def stitch(self, daily_chunks, monthly_data, weekly_data, config):
        """
        Returns: StitchingResult with fields:
          - stitched_series: pd.DataFrame [date, value]
          - alpha_estimates: np.ndarray
          - diagnostics: Dict (mae_monthly, mae_weekly, etc.)
        """
        pass
    
    @abstractmethod  
    def name(self) -> str:
        pass
```

All stitchers inherit this → uniform interface for comparison.

### TrendsAPI

```python
class TrendsAPI:
    def fetch(self, search_term, resolution, start_date, end_date):
        """
        resolution: 'yearly' | 'monthly' | 'weekly' | 'daily'
        Returns: pd.DataFrame [date, value, is_partial]
        """
```

Handles retries, exponential backoff, rate limiting.

### DataValidator

```python
class DataValidator:
    def validate(self, monthly, weekly, daily_chunks, config):
        """
        Returns: ValidationReport(
          is_valid: bool,
          warnings: List[str],
          errors: List[str],
          statistics: Dict
        )
        """
```

---

## Error Handling

| Error | Detection | Recovery |
|-------|-----------|----------|
| API 429 (rate limit) | HTTP status | Exponential backoff (2s, 4s, 8s), max 3 retries |
| API 401 (auth) | HTTP status | Abort immediately, check .env |
| Validation failure | DataValidator | Abort if critical, warn if minor |
| Optimization non-convergence | istop ≠ 1 | Fall back to monthly-only, or add ridge |
| Alpha out of bounds (< 0.1 or > 10) | After optimization | Flag chunk, continue |
| Memory exhaustion | MemoryError | Load chunks iteratively, use Parquet partitions |

**All errors logged:** `logs/stitching_{timestamp}.log` (DEBUG) + console (INFO)

---

## Future Extensions (Placeholders)

### Rung 4: Smooth Alpha Stitcher (Phase 4)
**Component:** `src/stitching/smooth_alpha.py` (not implemented)

**Method:** Add penalty λ Σ(α_k - α_{k-1})² to smooth alpha path

**Dependency:** `cvxpy` (install when needed)

**Interface:**
```python
class SmoothAlphaStitcher(StitchingEngine):
    def stitch(self, ...):
        raise NotImplementedError("Phase 4 - install cvxpy")
    def name(self):
        return "Hierarchical + Smooth Alpha"
```

**When to implement:** If Hierarchical produces alpha CV > 20% or suspected rebases

---

## Performance Targets

| Task | Target | Hardware |
|------|--------|----------|
| Data download (9 chunks) | < 15 min | Network-dependent |
| Baseline stitching | < 1 min | CPU |
| Hierarchical optimization | < 5 min | CPU |
| Robustness battery | < 15 min | CPU |
| **Full pipeline** | **< 30 min** | 16GB RAM, WSL |
| Peak RAM usage | < 4GB | Comfortable on 16GB |

---

## Testing Strategy

### Integration Tests (in Notebook 06)
1. Full pipeline on 1-year subset (smoke test)
2. Method comparison produces expected ranking
3. Config change propagates correctly

### Validation Tests (Analytical Plan Criteria)
1. MAE (monthly) < 3% for Hierarchical
2. Alpha CV < 10% across overlap tests
3. Temporal CV: Test MAE < Train + 2%
4. Seasonal sanity check: Oct-Nov peak present

### Unit Tests (Future)
- ConfigLoader validation
- TrendsAPI retry logic
- DataValidator zero classification

---

## Success Criteria by Phase

**Phase 1 Complete:**
- ✅ All data downloaded (yearly, monthly, weekly, daily chunks)
- ✅ Audit report generated, quality acceptable
- ✅ Chunk overlap verified correct

**Phase 2 Complete:**
- ✅ 3 methods implemented and compared
- ✅ PRIMARY METRICS: Weekly Correlation ≥ 0.90, Weekly NMAE < 0.50 (scale-invariant)
- ✅ SECONDARY METRICS: Weekly MAE < 0.50, Monthly MAE < 3.0 (context)
- ✅ All interim results saved

**Phase 3 Complete:**
- ✅ Robustness tests pass (Temporal CV NMAE gap < 0.10, Alpha CV acceptable)
- ✅ Final report generated with method recommendation
- ✅ Best method exported (Parquet + CSV)

**Production Ready:**
- ✅ All above + config snapshot saved

---

## Known Pitfalls

1. **Google Trends sampling:** Data varies ±0.5% across runs (not reproducible)
   - Mitigation: Document expected variation

2. **LSQR iteration limit:** Default 100 too low for 9+ chunks
   - Mitigation: Always set `iter_lim=1000`

3. **Zero-division in baseline:** Chunk mean = 0
   - Mitigation: Check, set alpha=1.0 if zero

4. **High overlap residuals:** Google's sampling causes 10-20% variation
   - Mitigation: Use low overlap weight (0.1), not hard constraint

5. **Memory with many chunks:** Loading all at once
   - Mitigation: Load iteratively if needed

---

## Next Steps

1. **Environment setup:**
   ```bash
   uv init google-trends-stitching
   cd google-trends-stitching
   # Create .env with SERPAPI_KEY=your_key
   # Create config.yaml (use template above)
   ```

2. **Implement Phase 1:**
   - `src/utils.py` (logger, file helpers)
   - `src/config.py` (ConfigLoader)
   - `src/api.py` (TrendsAPI)
   - `src/validation.py` (DataValidator)
   - `nb/01_data_collection.ipynb`

3. **Validate Phase 1:**
   - Run notebook, review audit report
   - Confirm: data quality acceptable?

4. **Proceed to Phase 2** (stitching methods)

---

## Related Documents

- **Analytical Plan:** `google_trends_stitching_analytical_plan.md`
  - Focuses on: Which methods, validation strategy, decision criteria
  - This design doc focuses on: How to implement, code structure, error handling

---

**Timeline:** 9-12 days for Phase 1-3 implementation  
**Tools:** Python 3.13, UV, Jupyter, Claude Code

**END OF DESIGN SUMMARY**
