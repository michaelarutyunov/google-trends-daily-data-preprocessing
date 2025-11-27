"""
Plotting utilities for Gradio app.
Creates interactive Plotly charts for visualization.
"""

from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_timeseries(
    stitched_daily: pd.DataFrame,
    weekly_data: Optional[pd.DataFrame],
    monthly_data: pd.DataFrame,
    search_term: str
) -> go.Figure:
    """
    Create interactive Plotly line chart showing stitched data vs ground truth.

    Args:
        stitched_daily: Stitched daily time series
        weekly_data: Weekly ground truth (optional)
        monthly_data: Monthly ground truth
        search_term: Search term for title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Stitched daily data (main line)
    fig.add_trace(go.Scatter(
        x=stitched_daily['date'],
        y=stitched_daily['value'],
        mode='lines',
        name='Stitched Daily',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:.1f}<extra></extra>'
    ))

    # Weekly ground truth (if available)
    if weekly_data is not None and not weekly_data.empty:
        fig.add_trace(go.Scatter(
            x=weekly_data['date'],
            y=weekly_data['value'],
            mode='markers',
            name='Weekly Ground Truth',
            marker=dict(color='#2ca02c', size=6, symbol='circle'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:.1f}<extra></extra>'
        ))

    # Monthly ground truth
    fig.add_trace(go.Scatter(
        x=monthly_data['date'],
        y=monthly_data['value'],
        mode='markers',
        name='Monthly Ground Truth',
        marker=dict(color='#d62728', size=8, symbol='square'),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:.1f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f'Google Trends: {search_term}',
        xaxis_title='Date',
        yaxis_title='Search Interest (0-100)',
        hovermode='closest',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def plot_alpha_progression(alpha_values: np.ndarray) -> go.Figure:
    """
    Create bar chart showing alpha scaling factors per chunk.

    Args:
        alpha_values: Array of alpha values (one per chunk)

    Returns:
        Plotly Figure object
    """
    # Calculate deviation from 1.0 for color coding
    deviations = np.abs(alpha_values - 1.0)
    max_deviation = max(deviations) if len(deviations) > 0 else 1.0

    # Color scale: green for values near 1.0, red for values far from 1.0
    colors = []
    for dev in deviations:
        if max_deviation > 0:
            ratio = dev / max_deviation
            # Interpolate between green and red
            r = int(255 * ratio)
            g = int(255 * (1 - ratio))
            colors.append(f'rgb({r}, {g}, 50)')
        else:
            colors.append('rgb(50, 255, 50)')

    fig = go.Figure()

    # Bar chart of alpha values
    fig.add_trace(go.Bar(
        x=list(range(len(alpha_values))),
        y=alpha_values,
        marker=dict(color=colors),
        hovertemplate='<b>Chunk</b>: %{x}<br><b>Alpha</b>: %{y:.3f}<extra></extra>',
        showlegend=False
    ))

    # Add horizontal reference line at alpha=1.0
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="α = 1.0 (no scaling)",
        annotation_position="right"
    )

    # Update layout
    fig.update_layout(
        title='Alpha Progression: Scaling Factors per Chunk',
        xaxis_title='Chunk ID',
        yaxis_title='Alpha (Scaling Factor)',
        template='plotly_white',
        height=400,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig
