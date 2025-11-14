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

# Run Phase 2: Stitching (Smooth Alpha method recommended)
jupyter notebook nb/04_smooth_alpha_stitching.ipynb
```

## Methods Implemented

| Method | Status | Notebook | Recommendation |
|--------|--------|----------|----------------|
| **Smooth Alpha** | ✅ Working | nb/04 | Leading |
| **Hierarchical** | ✅ Working | nb/03 | Leading |
| Hierarchical+DOW | ✅ Working | nb/05 | Day-of-week patterns |
| Baseline | ✅ Working | nb/02 | Reference only |

## Key Features

- **Hierarchical Optimization**: Soft constraints for monthly/weekly targets using LSQR
- **Independent Validation**: Weekly MAE against Google Trends weekly ground truth (not used in fitting)
- **Structural Zero Handling**: Configurable masking for seasonal terms (e.g., "flu vaccine")
- **Reproducible Pipeline**: Config-driven workflow with automated logging and validation reports
- **Production-Ready**: Parquet storage, comprehensive error handling, detailed diagnostics


## Project Structure

```
├── nb/                         # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_baseline_stitching.ipynb
│   ├── 03_hierarchical_stitching.ipynb
│   ├── 04_smooth_alpha_stitching.ipynb      
│   └── 05_hierarchical_dow_stitching.ipynb
├── src/
│   ├── stitching/              # Stitching methods
│   │   ├── smooth_alpha.py     
│   │   ├── hierarchical.py
│   │   ├── baseline.py
│   │   └── hierarchical_dow.py
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

- **[google_trends_design_summary.md](google_trends_design_summary.md)** - System design
- **[google_trends_stitching_analytical_plan.md](google_trends_stitching_analytical_plan.md)** - Methods and validation strategy

## Requirements

- Python 3.13+
- SerpAPI key (for Google Trends data)
- Key packages: pandas, numpy, scipy, statsmodels, cvxpy, jupyter

## License

[Add your license here]

## Citation

[Add citation info if publishing results]


