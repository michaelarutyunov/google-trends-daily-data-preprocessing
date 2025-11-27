---
title: Google Trends Daily Stitcher
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.0
app_file: gradio_app/app.py
pinned: false
license: mit
---

# Google Trends Daily Data Stitcher

Get high-quality daily Google Trends data for extended time periods using hierarchical constrained optimization.

## 🚀 Features

- **Enter any search term** - Track interest over time for any topic
- **Select date range** - Process data spanning 270+ days
- **Choose geographic location** - US, UK, Canada, Australia, and more
- **Receive stitched daily data** - High-quality continuous time series
- **Validation metrics** - MAE, correlation, bias against ground truth
- **Download complete package** - CSV + plots + metadata + validation report

## 📊 Method

This tool uses the **Smooth Alpha** stitching algorithm, which extends hierarchical constrained optimization with a smoothness penalty to reduce alpha volatility.

### Algorithm Overview

```
minimize: ||A·α - b||² + λ·||D·α||²
```

Where:
- **A·α - b**: Hierarchical constraints (monthly + weekly + overlap)
- **D·α**: First differences (smoothness penalty)
- **λ**: Smoothness parameter (default: 0.1)

### Performance

- **MAE**: 0.3-0.5 (Mean Absolute Error)
- **Correlation**: 0.95-0.97
- **Alpha CV**: < 20% (Coefficient of Variation)

## 🎯 How It Works

1. **Fetches overlapping daily chunks** from Google Trends (≤266 days each)
2. **Fetches ground truth** weekly and monthly data
3. **Runs optimization** using cvxpy convex solver
4. **Validates results** against independent ground truth
5. **Packages output** as downloadable ZIP file

## 📦 Download Package Contents

When you run the pipeline, you'll receive a ZIP file containing:

- `stitched_daily.csv` - Final daily time series
- `metadata.json` - Configuration and diagnostics
- `validation_report.txt` - Human-readable metrics summary
- `plot_timeseries.png` - Time series visualization (stitched vs ground truth)
- `plot_alpha.png` - Alpha progression chart
- `weekly_ground_truth.csv` - Weekly validation data
- `monthly_ground_truth.csv` - Monthly validation data

## 🔧 Technical Details

### Why Stitching is Needed

Google Trends only provides daily data for short ranges (≤270 days). For longer periods:
- Option 1: Get monthly data (low resolution)
- Option 2: Stitch overlapping daily chunks (this tool!)

### Problem Statement

When fetching overlapping daily chunks, each chunk is independently scaled by Google. Our algorithm finds optimal scaling factors (alphas) that:
1. Match monthly ground truth
2. Match weekly ground truth
3. Ensure consistency in overlap regions
4. Minimize alpha volatility (smoothness)

### Constraints

- **Monthly**: `Σ(α_k · daily_k) = monthly_truth` for each month
- **Weekly**: `Σ(α_k · daily_k) = weekly_truth` for each week
- **Overlap**: `α_k · daily_k ≈ α_{k+1} · daily_{k+1}` in overlap regions
- **Smoothness**: `Minimize Σ(α_k - α_{k-1})²`

## 🌐 Use Cases

- **Academic Research**: Track search interest for research topics over time
- **Market Analysis**: Monitor brand, product, or competitor interest
- **Trend Forecasting**: Analyze historical patterns for predictions
- **Data Journalism**: Investigate public interest in events or topics
- **SEO/Marketing**: Understand search demand across extended periods

## 📚 Repository

Full source code, research notebooks, and methodology documentation available at:
[GitHub - google-trends-daily-data-preprocessing](https://github.com/mikhailarutyunov/google-trends-daily-data-preprocessing)

### Repository Contents

- `src/` - Core stitching algorithms and API wrappers
- `nb/` - Jupyter notebooks with research and validation
- `gradio_app/` - This Gradio web interface
- `reports/` - Validation reports and metrics
- `results/` - Stitched data outputs

## 🛠️ Local Development

To run this app locally:

```bash
# Clone repository
git clone https://github.com/mikhailarutyunov/google-trends-daily-data-preprocessing.git
cd google-trends-daily-data-preprocessing

# Install dependencies
pip install -r requirements.txt

# Set API key
export SERPAPI_KEY="your_serpapi_key_here"

# Run app
cd gradio_app
python app.py
```

## 📄 License

MIT License - See repository for full details

## 🙏 Acknowledgments

- **SerpAPI** for providing Google Trends API access
- **Gradio** for the easy-to-use web framework
- **Hugging Face** for free hosting on Spaces
- **cvxpy** for convex optimization solver

## 📧 Contact

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/mikhailarutyunov/google-trends-daily-data-preprocessing).

---

**Powered by SerpAPI | Built with Gradio | Deployed on Hugging Face Spaces**
