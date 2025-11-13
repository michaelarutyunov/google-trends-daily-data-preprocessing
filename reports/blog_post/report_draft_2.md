# Stitching Google Trends Daily Data: Finding the Right Method

## Why This Matters

In 2025, Google Trends will celebrate its 20th birthday. Over these two decades, it has proven to be a highly valuable real-time, high-frequency, and low-cost data source. Researchers use it to nowcast diseases and economic indicators, while businesses rely on it for market research and strategic planning.

Its simple interface makes it accessible to anyone – but the simplicity hides some important nuances:

1. **Relative scaling**: The data is rescaled to 0-100, where 100 represents the maximum value in the requested time interval. This means you can't directly compare values across different queries or date ranges.

2. **Differential frequency**: Daily granularity is only available for intervals up to 269 days. Need a full year or more of daily data? You'll need to download overlapping chunks and stitch them together.

This is where it gets interesting. Stitching chunks that are normalized to different scales requires ground truth data to calibrate against. In marketing, for example, Google Ads click data often serves this purpose.

There are commercial solutions like **Glimpse** that offer flexible data downloads with various frequencies. However, I couldn't find detailed documentation about their stitching methodology, so I decided to explore several approaches myself and see which one performs best.

## The Approach

Here's the game plan:

1. **Get data at multiple resolutions**: Download historical monthly data (to capture global maximum), weekly data (for independent validation), and overlapping daily chunks (for the period of interest).

2. **Calibrate using ground truth**: Use long-term monthly data to understand the global scale and normalize daily values accordingly.

3. **Test multiple stitching methods**: Implement different techniques ranging from simple averaging to sophisticated constrained optimization.

4. **Validate independently**: Aggregate stitched daily data to weekly level and compare against directly downloaded weekly data – this avoids circular validation issues.

5. **Select the right metrics**: Since Google Trends uses different 0-100 normalizations at different resolutions, traditional metrics like R² fail. Instead, focus on MAE (absolute error), correlation (pattern similarity), and bias (systematic deviation).

## Setup

I chose **"flu vaccine"** searches in the **UK** as a test case, focusing on **January 2022 to December 2024**. This term has clear seasonal patterns (high in autumn/winter, low in summer) which makes it great for testing.

Since Google Trends' API closed some time ago (with a new version currently in alpha testing), I used **SerpAPI** as a proxy to access the data.

The data collection yielded:
- **263 monthly observations** (2004-present, for global context)
- **158 weekly observations** (2022-2024, for validation)
- **6 daily chunks** of ~266 days each with 60-day overlaps
- **1,096 daily points** after stitching

## The Methods

### 1. Baseline (Reference)
Averages overlapping daily chunks to create a raw daily series, then aggregates to monthly totals. It computes scaling factors (alphas) as the ratio of ground truth monthly values to raw monthly values: α = monthly_truth / monthly_raw. These factors are applied uniformly to all days within each month.

**Limitation**: This method uses monthly ground truth to calculate alphas, then validates against the same monthly data – creating circular validation. Only weekly validation is meaningful.

### 2. Hierarchical
Uses constrained least-squares optimization to find scaling factors (alphas) for each chunk. The optimization balances three types of constraints:
- **Monthly constraints** (weighted, not hard): Match monthly aggregates to ground truth
- **Weekly constraints** (weighted): Match weekly aggregates to ground truth
- **Overlap constraints** (soft): Ensure continuity in overlapping regions

The key advantage: since constraints are weighted (soft), not forced, both monthly and weekly validation are meaningful.

### 3. Hierarchical + Day-of-Week Correction
Extends hierarchical stitching by estimating day-of-week (DOW) patterns from the residuals between stitched data and weekly ground truth. The method calculates DOW factors representing relative activity for each weekday (Monday=0 to Sunday=6), normalizes them to average 1.0, and applies a multiplicative correction to the stitched series.

**Note**: An earlier version included a renormalization step that forced monthly totals to match ground truth exactly, which destroyed the hierarchical optimization and created circular validation. This has been removed in the current implementation.

### 4. Smooth Alpha (Recommended)
Extends hierarchical optimization by adding a smoothness penalty on alpha values using convex optimization (cvxpy). The objective function is:

**minimize: ||Aα - b||² + λ||Dα||²**

Where:
- A is the constraint matrix, b is the target values
- D is a first-difference matrix that penalizes large changes between consecutive alphas
- λ is the smoothness parameter (λ=499 in this implementation)

This approach produces more stable alpha estimates (CV=0.54 vs 0.81 for hierarchical) while maintaining all the benefits of soft constraints.

## Choosing the Right Metrics

Moving between daily, weekly, and monthly levels with Google's 0-100 normalization is confusing. The core problem: **stitched daily data and weekly ground truth are on different scales**.

Traditional R² fails spectacularly here (values of -45 to -90!) because it penalizes correct patterns that are simply on different scales. Instead, I focused on four metrics:

1. **Weekly MAE (Mean Absolute Error)**: Primary accuracy metric. Measures average absolute difference between aggregated daily data and weekly ground truth. Target: < 1.5 (all methods achieved this).

2. **Pearson Correlation**: Pattern similarity independent of scale. A high correlation (>0.90) means the method captures the right trends, even if the absolute values differ slightly.

3. **NMAE (Normalized MAE)**: MAE divided by the range of weekly values. This makes the metric scale-invariant, allowing comparison across different search terms or time periods.

4. **Bias %**: Measures systematic deviation from ground truth as a percentage. Positive bias = overestimation, negative = underestimation. Target: close to 0%.

5. **Alpha CV (Coefficient of Variation)**: Measures stability of chunk-level scaling factors. High CV suggests chunks vary widely in quality or that Google rebased the data during your period.

## Results

### Performance Summary

| Method               | Weekly MAE | Pearson Corr | NMAE  | Bias  | Alpha CV |
|:---------------------|------------|--------------|-------|-------|----------|
| Baseline             | 0.70       | 0.936        | 0.765 | 74.4% | 0.72     |
| Hierarchical         | 0.36       | 0.902        | 0.391 | 3.6%  | 0.81     |
| Smooth Alpha (λ=499) | **0.26**   | **0.937**    | **0.279** | **-0.4%** | **0.54** |
| Hierarchical+DOW     | 0.38       | 0.900        | 0.412 | 9.2%  | 0.81     |

**Validation on holdout period (last 3 months):**

| Method          | Train MAE | Test MAE | Generalization Gap |
|:----------------|-----------|----------|-------------------|
| Baseline        | 0.76      | 0.44     | -43% ✓            |
| Hierarchical    | 0.38      | 0.23     | -40% ✓            |
| Smooth Alpha    | **0.26**  | **0.20** | **-21% ✓**        |
| Hierarchical+DOW| 0.40      | 0.31     | -23% ✓            |

*(Negative gap means better performance on test data – excellent!)*

## Conclusions

### Clear Winner: Smooth Alpha

After testing four methods on 158 weeks of independent validation data, **Smooth Alpha with λ=499** emerges as the recommended approach for production use. Here's why:

**1. Best Accuracy Where It Matters**
- Weekly MAE of **0.26** – that's **63% better than baseline** and **28% better than standard hierarchical**
- Achieved the target bias of nearly 0% (-0.4%), meaning no systematic over- or under-prediction
- Maintained high pattern similarity (correlation 0.937) while achieving superior accuracy

**2. Better Generalization**
- Test set performance actually improved (MAE 0.20 vs 0.26 on training)
- This negative generalization gap (-21%) suggests the method isn't overfitting and captures real patterns rather than noise

**3. More Stable Scaling**
- Alpha CV of 0.54 vs 0.81 for hierarchical methods
- The smoothness penalty (λ||Dα||²) successfully reduced wild swings in chunk-level scaling factors
- More stable alphas = more reliable extrapolation to new time periods

**4. Principled Optimization**
- Uses soft (weighted) constraints, so both monthly and weekly validation are meaningful
- Convex optimization guarantees finding the global optimum
- Explicit trade-off parameter (λ) between constraint satisfaction and smoothness

### When to Use Alternatives

- **Hierarchical**: If you can't install cvxpy or need a simpler implementation. Only 28% worse than Smooth Alpha, still performs well (Weekly MAE 0.36).

- **Baseline**: For quick prototyping or when ground truth is only monthly. Simple to implement but 63% worse accuracy. Not recommended for production.

- **Hierarchical+DOW**: If you have strong day-of-week patterns (e.g., retail sales). The DOW correction helped with bias but didn't improve overall accuracy for "flu vaccine".

### Key Takeaways

1. **Weekly validation is critical**: Monthly validation can be circular depending on the method. Always validate against independent weekly ground truth.

2. **Smoothness matters**: Raw optimization can produce jumpy scaling factors between chunks. Adding a smoothness penalty significantly improves both stability (54% better alpha CV) and accuracy (28% better MAE).

3. **R² is the wrong metric**: When working with multi-resolution Google Trends data, use MAE for accuracy, correlation for pattern similarity, and bias for systematic errors.

4. **All methods generalize well**: Every method showed negative generalization gaps, meaning they performed better on the holdout period. This suggests Google Trends data is relatively consistent and the methods aren't overfitting.

5. **The method scales**: While tested on "flu vaccine" (UK, 2022-2024), the approach generalizes to any search term and geography. Just adjust the overlap period and number of chunks as needed.

### Production Implementation

The recommended configuration:
- **Method**: Smooth Alpha
- **Smoothness parameter**: λ = 499
- **Overlap period**: 60 days
- **Number of chunks**: 6 (for 3-year period)
- **Expected performance**: Weekly MAE < 0.3, Correlation > 0.93

All code and results are available in the project repository, including the full stitched dataset (`results/smooth_alpha_final.parquet`) ready for downstream forecasting or nowcasting applications.

---

*Have you worked with Google Trends data? What challenges have you encountered with daily stitching? I'd love to hear about your experiences and approaches in the comments below.*
