# AGENTS.md

This file provides guidance to Kimi CLI when working with code in this repository.

## Project Snapshot for Kimi

**Current Status**: Phase 3 complete ✅ | **Recommended Method**: Smooth Alpha (Weekly MAE 0.32) | **Primary Goal**: Stitch Google Trends daily chunks into reliable time series

## Kimi CLI Quick Start

### Essential Context (Read First)
1. **Google Trends has NO resolution parameter** - date range determines frequency automatically
2. **Weekly data uses Sunday-ending weeks** - critical for validation (`resample('W-SUN')`)
3. **Smooth Alpha method is production-ready** - 54% better than baseline, 11% better than Hierarchical
4. **Weekly MAE is primary validation metric** - target < 1.5, achieved 0.32

### Kimi's Optimal Workflow
1. **Understand constraints first** - Google Trends API limitations drive entire design
2. **Follow the three-phase structure** - Data Collection → Stitching Methods → Validation
3. **Use systematic validation** - Always check both weekly and monthly metrics where meaningful
4. **Preserve modular architecture** - All stitchers inherit from `StitchingEngine` base class

## Critical Technical Details

### Google Trends API Behavior (Must Know)
```
Long ranges (2004-present) → Monthly frequency
Medium ranges (1-3 years) → Weekly frequency  
Short ranges (≤270 days) → Daily frequency
```

**Key Implications:**
- `fetch_historical_monthly()` gets monthly data (not yearly despite name)
- `fetch_daily_chunks()` must use ≤266 day chunks with 60-day overlap
- Weekly validation requires Sunday-based weeks (`W-SUN` not `W`)

### Validation Methodology for Kimi
```
Primary: Weekly MAE (independent validation) - Target < 1.5
Secondary: Monthly MAE (where meaningful) - Target < 3.0
Avoid: R² (deprecated due to scale mismatches)
Use: Pearson correlation, NMAE, Bias % instead
```

**Method Validation Status:**
- ✅ **Smooth Alpha**: Both metrics meaningful (Weekly 0.32, Monthly 2.96)
- ✅ **Hierarchical**: Both metrics meaningful (Weekly 0.36, Monthly 2.92)
- ⚠️ **Baseline**: Only Weekly MAE valid (Monthly is circular → 0.00)
- ⚠️ **State-Space**: Neither metric reliable (heuristic, not true Kalman)

## Architecture for Systematic Analysis

### Directory Structure (Kimi's Map)
```
nb/                    # Jupyter notebooks (01-08, sequential workflow)
src/                   # Python modules (modular, testable)
├── stitching/         # All stitchers inherit from StitchingEngine
├── validation.py      # Validation framework
├── api.py            # SerpAPI wrapper
└── config.py         # Configuration management

data/raw/             # API responses (gitignored)
interim/              # Method outputs (gitignored)  
results/              # Final outputs (committed)
reports/              # Validation reports
```

### Stitching Engine Pattern (Kimi's Focus)
```python
class StitchingEngine(ABC):
    @abstractmethod
    def stitch(self, daily_chunks, monthly_data, weekly_data, config):
        """Returns: StitchingResult with stitched_series, alpha_estimates, diagnostics"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass
```

**Kimi's Analysis Approach:**
1. Check inheritance hierarchy first
2. Understand constraint matrices and optimization setup
3. Validate alpha estimation methodology
4. Verify diagnostic output completeness

## Common Kimi Tasks

### When Debugging Stitching Issues
1. **Check optimization convergence** - LSQR needs `iter_lim=1000` for multiple chunks
2. **Verify date alignment** - Weekly must use Sunday weeks, overlap regions should align
3. **Validate constraint matrices** - Monthly constraints should be weighted, not forced
4. **Examine alpha estimates** - CV >10% indicates chunk quality issues

### When Adding New Methods
1. **Inherit from StitchingEngine** - Maintain uniform interface
2. **Implement both constraints** - Monthly aggregation + some form of regularization
3. **Include comprehensive diagnostics** - alpha_estimates, convergence info, validation metrics
4. **Test with existing validation framework** - Ensure weekly/monthly MAE calculation works

### When Optimizing Performance
1. **Profile constraint matrix construction** - Often the bottleneck
2. **Check memory usage** - All chunks loaded simultaneously (~4GB+ peak)
3. **Validate API rate limiting** - Exponential backoff for 429 errors
4. **Consider chunk size tradeoffs** - Smaller chunks = more API calls but better overlap

## Kimi-Specific Insights

### Systematic Problem-Solving Pattern
1. **Isolate the layer** - API vs stitching vs validation issue?
2. **Check constraints first** - Are monthly/weekly constraints properly formulated?
3. **Validate data flow** - Are daily chunks, monthly data, weekly data aligned?
4. **Examine optimization** - Convergence, alpha bounds, residual patterns

### Code Quality Focus Areas
- **Modularity**: Each stitcher should be independently testable
- **Validation**: Always include both weekly and monthly diagnostics
- **Error handling**: Graceful degradation for API issues, optimization failures
- **Reproducibility**: Fixed random seeds, deterministic optimization

### Performance Considerations
- **Memory**: Stream large chunk processing if needed
- **API calls**: Batch requests, respect rate limits
- **Optimization**: Warm-start similar problems, use appropriate solvers
- **Validation**: Cache intermediate results for iterative development

## Quick Reference for Kimi

### Essential Files to Understand First
1. `src/stitching/base.py` - Core abstraction and validation framework
2. `src/stitching/smooth_alpha.py` - Recommended method implementation
3. `src/validation.py` - Validation metrics and cross-validation
4. `config.yaml` - All tunable parameters

### Key Configuration Parameters
```yaml
search_term: "your search term"  # Change this for different analyses
date_range: ["2022-01-01", "2024-12-31"]
overlap_days: 60  # Critical for chunk stitching
lambda_smooth: 99  # For Smooth Alpha method
structural_zero_months: [6, 7, 8]  # For seasonal terms
```

### Validation Targets (Achieved)
- Weekly MAE: 0.32 (target < 1.5) ✅
- Monthly MAE: 2.96 (target < 3.0) ✅  
- Alpha CV: 0.70 (high but acceptable) ⚠️
- Temporal CV: -39% (better on test!) ✅

## Kimi CLI Advantages Here

- **Systematic analysis**: Perfect for understanding constraint optimization
- **Code structure insight**: Excellent at navigating modular architecture  
- **Validation rigor**: Strong at implementing comprehensive testing frameworks
- **Debugging methodology**: Methodical approach to complex stitching issues
- **Performance optimization**: Good at identifying bottlenecks in optimization pipelines

This project rewards Kimi's systematic, thorough approach to understanding complex constraint satisfaction problems.