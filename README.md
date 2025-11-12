# Google Trends Daily Data Preprocessing

A production-ready system for stitching overlapping Google Trends daily chunks into valid time series using hierarchical constrained optimization.

## Overview

Google Trends provides different data resolutions based on the date range requested:
- **Short ranges** (≤270 days) → Daily frequency
- **Medium ranges** (1-3 years) → Weekly frequency
- **Long ranges** (2004-present) → Monthly frequency

This project solves the challenge of creating **reliable daily time series** for arbitrary date ranges by:
1. Fetching overlapping daily chunks (≤266 days each)
2. Stitching them together using hierarchical optimization with monthly/weekly constraints
3. Validating results against independent weekly ground truth

**Status**: Phase 2 complete. Hierarchical method achieves **Weekly MAE 0.91** (target: ≤1.5).

## Quick Start

```bash
# Setup environment
uv venv && source .venv/bin/activate
uv pip install -e ".[stats,viz,dev]"

# Configure API keys
cp .env.example .env
# Edit .env and add your SERPAPI_KEY

# Run Phase 1: Data Collection
jupyter notebook nb/01_data_collection.ipynb

# Run Phase 2: Stitching (Hierarchical method recommended)
jupyter notebook nb/03_hierarchical_stitching.ipynb
```

## Methods Implemented

| Method | Weekly MAE | Status | Notebook | Recommendation |
|--------|------------|--------|----------|----------------|
| **Hierarchical** | **0.91** ⭐ | ✅ Working | nb/03 | **USE THIS** - Best accuracy |
| Smooth Alpha | 0.91 | ✅ Working | nb/04 | Use if Alpha CV > 50% |
| Baseline | 1.37 | ✅ Working | nb/02 | Reference only |
| Hierarchical+DOW | ~0.85-0.95 | ⚠️ Needs retest | nb/05 | Day-of-week patterns |
| State-Space | 1.37 | ⚠️ Heuristic | nb/08 | Exploratory reference |

## Key Features

- **Hierarchical Optimization**: Soft constraints for monthly/weekly targets using LSQR
- **Independent Validation**: Weekly MAE against Google Trends weekly ground truth (not used in fitting)
- **Structural Zero Handling**: Configurable masking for seasonal terms (e.g., "flu vaccine")
- **Reproducible Pipeline**: Config-driven workflow with automated logging and validation reports
- **Production-Ready**: Parquet storage, comprehensive error handling, detailed diagnostics

## Validation Results

**Hierarchical Method** (Recommended):
- ✅ **Weekly MAE: 0.91** (independent validation, target ≤1.5)
- ✅ **Monthly MAE: 2.04** (soft constraints, target <3.0)
- ✅ Converges in ~6 iterations
- ⚠️ Alpha CV: 56% (indicates chunk quality variation, but results remain accurate)

## Project Structure

```
├── nb/                         # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_baseline_stitching.ipynb
│   ├── 03_hierarchical_stitching.ipynb      ⭐ Recommended
│   ├── 04_smooth_alpha_stitching.ipynb
│   ├── 05_hierarchical_dow_stitching.ipynb
│   └── 08_state_space_stitching.ipynb       (heuristic reference)
├── src/
│   ├── stitching/              # Stitching methods
│   │   ├── hierarchical.py     # ⭐ Production method
│   │   ├── baseline.py
│   │   ├── smooth_alpha.py
│   │   ├── hierarchical_dow.py
│   │   └── state_space.py
│   ├── config.py               # ConfigManager
│   ├── api.py                  # SerpAPI wrapper
│   ├── validation.py           # Validation metrics
│   └── utils.py                # Logging, file management
├── config.yaml                 # Stitching configuration
├── data/raw/                   # API responses (gitignored)
├── interim/                    # Stitched results (gitignored)
├── results/                    # Final outputs
└── reports/                    # Validation reports
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete implementation guide for Claude Code
- **[google_trends_design_summary.md](google_trends_design_summary.md)** - System design
- **[google_trends_stitching_analytical_plan.md](google_trends_stitching_analytical_plan.md)** - Methods and validation strategy
- **[reports/validation_comparison.md](reports/validation_comparison.md)** - Method performance comparison

## Requirements

- Python 3.13+
- SerpAPI key (for Google Trends data)
- Key packages: pandas, numpy, scipy, statsmodels, cvxpy, jupyter

## License

[Add your license here]

## Citation

[Add citation info if publishing results]