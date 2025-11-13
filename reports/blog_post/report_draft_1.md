Next year Google Trends will celebrate its 20th birthday.
Since then it has proven to be a highly valuable real-time, high-frequency and low-cost data source.
It is used in healthcare to now-cast diseases and economic indicators, used in market research and business planning.

Its simple inferface makes 

But it is not without the nuances.
1. the data is rescaled to 0-100 where 100 is the maximum value in a given interval. Hence…
2. differencial frequency: the daily granularity is only available for intervals up to 269 days, meaning that..

If the business needs a daily tracking for a year or more, it needs to get the chunks of data and stitch them together. But stitching together the chunks at different scales requires ground truth data. 
E.g. in marketing Google Ad Words is a common source of ground truth data.

There are commercial solutions like Gimple that offer flexiblity with downloading the data with various frequencies. However, i could not find the details about the method they use to stitch and wanted to explore few options myself.


The idea is:
- get the data on monthly, weekly and daily levels
- get long term monthly data that captures the global maximum for the search term and useto calibrate the daily values
- get overlapping chunks of daily data covering the period of interest
- get a few stitched variants using different techniques 
- resample the stitched daily data to monthly level and compare it with directly downloaded weekly data for the focus period
- select the right metrics to see which stitching method works best

Setup

I chose UK’s “flu vaccine” for a test, focusing on Jan 2022-2025
Google Trends API has closed some time ago and a new version is in alpha testing now, I used SerpAPI.

(plot with data downloaded from google trends)


1.Baseline: averages overlapping daily chunks to create a raw daily series, then aggregates this to monthly totals. It computes scaling factors (alphas) as the ratio of ground truth monthly values to raw monthly values, and applies these factors uniformly to all days within each month.

2.Hierarchical: Uses constrained optimization to solve a least-squares problem with three constraint types: monthly constraints (hard), weekly constraints (weighted), and overlap constraints (soft). Each chunk gets an alpha scaling factor that minimizes the weighted error across all constraints.

3.Hierarchical with Day of Week: Extends hierarchical stitching by estimating day-of-week (DOW) patterns from weekly data residuals. Calculates DOW factors representing relative activity for each weekday (Monday=0 to Sunday=6), normalizes them to average 1.0, and applies multiplicative correction to the stitched series.

4.Alpha Smoothing: Extends hierarchical optimization by adding a smoothness penalty on alpha values using convex optimization (cvxpy). Minimizes the objective function: ||Aα - b||² + λ||Dα||², where D is a first-difference matrix penalizing large changes between consecutive alphas. The smoothness parameter λ controls the trade-off between constraint satisfaction and alpha smoothness.

Metrics:
Moving between the daily, weekly and monthly levels with Google’s 0-100 normalization is fairly confusing.
The problem is that the scales of the re-sampled daily data and weekly ground truth are not comparable.
So I chose to compare the methods in terms of:
- Correlation - for pattern similarity
- Weekly NMAE - MAE normalised by the interval between…
- Bias - to monitor systematic deviation vs ground truth
- Alpha CV - generalisation gap derived from validating the method on the last 3 month of the data

Results:

Metrics Summary Table:

| Method               |   Weekly MAE |   Pearson Corr |   NMAE | Bias %   |   Alpha CV |
|:---------------------|-------------:|---------------:|-------:|:---------|-----------:|
| Baseline             |         0.7  |          0.936 |  0.765 | 74.4%    |       0.72 |
| Hierarchical         |         0.36 |          0.902 |  0.391 | 3.6%     |       0.81 |
| Smooth Alpha (λ=499) |         0.26 |          0.937 |  0.279 | -0.4%    |       0.54 |
| Hierarchical+DOW     |         0.38 |          0.9   |  0.412 | 9.2%     |       0.81 |

(plot with four panels showing each method vs ground truth)