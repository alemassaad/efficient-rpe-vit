"""
Streamlit Dashboard for Benchmark Visualization

Usage:
    streamlit run experiments/dashboard.py

Then navigate to http://localhost:8501 and enter the path to your benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def load_benchmark_data(benchmark_dir: Path) -> Dict:
    """
    Load all benchmark data from a directory.

    Args:
        benchmark_dir: Path to benchmark results directory

    Returns:
        Dict with structure:
        {
            'config': {...},
            'models': {
                'baseline': {
                    'aggregated': {...},
                    'runs': [...]
                },
                ...
            }
        }
    """
    benchmark_dir = Path(benchmark_dir)

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Directory not found: {benchmark_dir}")

    # Load config
    config_file = benchmark_dir / 'benchmark_config.json'
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Load all models
    models = {}
    for model_dir in benchmark_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Load aggregated stats
        agg_file = model_dir / 'aggregated_stats.json'
        if not agg_file.exists():
            continue

        with open(agg_file, 'r') as f:
            aggregated = json.load(f)

        # Load individual runs
        runs = []
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                continue

            # Find metrics file
            metrics_files = list(run_dir.glob('*_metrics.json'))
            if not metrics_files:
                continue

            with open(metrics_files[0], 'r') as f:
                run_data = json.load(f)
                runs.append(run_data)

        models[model_name] = {
            'aggregated': aggregated,
            'runs': runs
        }

    return {
        'config': config,
        'models': models
    }


def create_summary_table(data: Dict) -> pd.DataFrame:
    """Create summary statistics table."""
    rows = []

    for model_name, model_data in data['models'].items():
        stats = model_data['aggregated']['statistics']

        row = {
            'Model': model_name,
            'Runs': model_data['aggregated']['num_runs'],
        }

        # Best accuracy
        if 'best_test_accuracy' in stats:
            s = stats['best_test_accuracy']
            row['Best Acc (%)'] = f"{s['mean']:.2f} ± {s['std']:.2f}"
            row['Acc Range'] = f"[{s['min']:.2f}, {s['max']:.2f}]"

        # Training time
        if 'avg_train_time_per_epoch' in stats:
            s = stats['avg_train_time_per_epoch']
            row['Train Time (s)'] = f"{s['mean']:.2f} ± {s['std']:.2f}"

        # Inference latency
        if 'inference_latency_mean_ms' in stats:
            s = stats['inference_latency_mean_ms']
            row['Inference (ms)'] = f"{s['mean']:.2f} ± {s['std']:.2f}"

        # Convergence
        if 'epochs_to_95_percent' in stats and stats['epochs_to_95_percent']['mean']:
            s = stats['epochs_to_95_percent']
            row['Epochs to 95%'] = f"{s['mean']:.1f}"

        # Parameters
        if 'total_parameters' in stats:
            s = stats['total_parameters']
            row['Parameters'] = f"{int(s['mean']):,}"

        rows.append(row)

    return pd.DataFrame(rows)


def plot_accuracy_comparison(data: Dict):
    """Line plot with percentile bands comparing test accuracy across models."""
    fig = go.Figure()

    model_names = list(data['models'].keys())
    colors = px.colors.qualitative.Plotly

    for idx, model_name in enumerate(model_names):
        model_data = data['models'][model_name]
        stats = model_data['aggregated']['statistics']

        if 'best_test_accuracy' not in stats:
            continue

        values = [v for v in stats['best_test_accuracy']['values'] if v is not None]
        if not values:
            continue

        # Compute percentiles and mean
        p5, p25, p75, p95 = np.percentile(values, [5, 25, 75, 95])
        mean_val = np.mean(values)

        color = colors[idx % len(colors)]
        x_pos = idx

        # Add 5-95 percentile range (outer, lighter)
        fig.add_trace(go.Scatter(
            x=[x_pos, x_pos],
            y=[p5, p95],
            mode='lines',
            line=dict(color=color, width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add filled area for 5-95 range
        fig.add_trace(go.Scatter(
            x=[x_pos-0.15, x_pos+0.15, x_pos+0.15, x_pos-0.15],
            y=[p5, p5, p95, p95],
            fill='toself',
            fillcolor=color,
            opacity=0.15,
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add 25-75 percentile range (inner, darker)
        fig.add_trace(go.Scatter(
            x=[x_pos-0.15, x_pos+0.15, x_pos+0.15, x_pos-0.15],
            y=[p25, p25, p75, p75],
            fill='toself',
            fillcolor=color,
            opacity=0.3,
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add dashed lines at percentiles
        for percentile_val in [p5, p25, p75, p95]:
            fig.add_trace(go.Scatter(
                x=[x_pos-0.15, x_pos+0.15],
                y=[percentile_val, percentile_val],
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                showlegend=False,
                hovertemplate=f'{model_name}<br>Value: %{{y:.2f}}%<extra></extra>'
            ))

        # Add mean as solid line
        fig.add_trace(go.Scatter(
            x=[x_pos-0.2, x_pos+0.2],
            y=[mean_val, mean_val],
            mode='lines',
            line=dict(color=color, width=3),
            name=model_name,
            hovertemplate=f'{model_name}<br>Mean: %{{y:.2f}}%<extra></extra>'
        ))

        # Add individual data points
        fig.add_trace(go.Scatter(
            x=[x_pos] * len(values),
            y=values,
            mode='markers',
            marker=dict(
                size=8,
                color=color,
                line=dict(color='white', width=1)
            ),
            showlegend=False,
            hovertemplate=f'{model_name}<br>Run value: %{{y:.2f}}%<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title='Test Accuracy Distribution with Percentiles',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(model_names))),
            ticktext=model_names,
            title='Model'
        ),
        yaxis_title='Accuracy (%)',
        height=500,
        hovermode='closest'
    )

    return fig


def plot_training_curves(data: Dict, metric='accuracy'):
    """Plot training curves with percentile confidence bands."""
    fig = go.Figure()
    color_idx = 0

    for model_name, model_data in data['models'].items():
        runs = model_data['runs']

        if not runs or 'per_epoch' not in runs[0]:
            continue

        # Collect data from all runs
        epochs = []
        values = []

        for run in runs:
            per_epoch = run['per_epoch']
            for epoch_data in per_epoch:
                epoch = epoch_data['epoch']
                if metric == 'accuracy':
                    val = epoch_data.get('test_accuracy', 0)
                else:  # loss
                    val = epoch_data.get('test_loss', 0)

                epochs.append(epoch)
                values.append(val)

        if not epochs:
            continue

        # Group by epoch and compute mean and percentiles
        df = pd.DataFrame({'epoch': epochs, 'value': values})
        grouped = df.groupby('epoch')['value']

        x = []
        means = []
        p5_vals = []
        p25_vals = []
        p75_vals = []
        p95_vals = []

        for epoch, group in grouped:
            vals = group.values
            x.append(epoch)
            means.append(np.mean(vals))
            p5, p25, p75, p95 = np.percentile(vals, [5, 25, 75, 95])
            p5_vals.append(p5)
            p25_vals.append(p25)
            p75_vals.append(p75)
            p95_vals.append(p95)

        color = px.colors.qualitative.Plotly[color_idx % 10]
        color_idx += 1

        # Add 5-95 percentile band (outer, lighter)
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=p95_vals + p5_vals[::-1],
            fill='toself',
            fillcolor=color,
            opacity=0.15,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
            name=f'{model_name} (5-95%)'
        ))

        # Add 25-75 percentile band (inner, darker)
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=p75_vals + p25_vals[::-1],
            fill='toself',
            fillcolor=color,
            opacity=0.3,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
            name=f'{model_name} (25-75%)'
        ))

        # Plot mean line (solid, on top)
        fig.add_trace(go.Scatter(
            x=x,
            y=means,
            name=model_name,
            mode='lines+markers',
            line=dict(width=2, color=color),
            marker=dict(size=4)
        ))

    ylabel = 'Test Accuracy (%)' if metric == 'accuracy' else 'Test Loss'
    fig.update_layout(
        title=f'Training Curves: {ylabel}',
        xaxis_title='Epoch',
        yaxis_title=ylabel,
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_efficiency_comparison(data: Dict):
    """Bar charts for training time and inference latency."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Time per Epoch', 'Inference Latency')
    )

    models = []
    train_times = []
    train_errs = []
    inference_times = []
    inference_errs = []

    for model_name, model_data in data['models'].items():
        stats = model_data['aggregated']['statistics']
        models.append(model_name)

        # Training time
        if 'avg_train_time_per_epoch' in stats:
            s = stats['avg_train_time_per_epoch']
            train_times.append(s['mean'])
            train_errs.append(s['std'])
        else:
            train_times.append(0)
            train_errs.append(0)

        # Inference latency
        if 'inference_latency_mean_ms' in stats:
            s = stats['inference_latency_mean_ms']
            inference_times.append(s['mean'])
            inference_errs.append(s['std'])
        else:
            inference_times.append(0)
            inference_errs.append(0)

    # Training time bars
    fig.add_trace(
        go.Bar(x=models, y=train_times, error_y=dict(type='data', array=train_errs),
               name='Train Time', showlegend=False),
        row=1, col=1
    )

    # Inference latency bars
    fig.add_trace(
        go.Bar(x=models, y=inference_times, error_y=dict(type='data', array=inference_errs),
               name='Inference', showlegend=False),
        row=1, col=2
    )

    fig.update_xaxes(title_text='Model', row=1, col=1)
    fig.update_xaxes(title_text='Model', row=1, col=2)
    fig.update_yaxes(title_text='Time (seconds)', row=1, col=1)
    fig.update_yaxes(title_text='Latency (ms)', row=1, col=2)

    fig.update_layout(height=400)

    return fig


def plot_efficiency_scatter(data: Dict):
    """Scatter plot: accuracy vs training time."""
    models = []
    accuracies = []
    train_times = []

    for model_name, model_data in data['models'].items():
        stats = model_data['aggregated']['statistics']

        if 'best_test_accuracy' in stats and 'avg_train_time_per_epoch' in stats:
            models.append(model_name)
            accuracies.append(stats['best_test_accuracy']['mean'])
            train_times.append(stats['avg_train_time_per_epoch']['mean'])

    df = pd.DataFrame({
        'Model': models,
        'Accuracy (%)': accuracies,
        'Training Time (s/epoch)': train_times
    })

    fig = px.scatter(df, x='Training Time (s/epoch)', y='Accuracy (%)',
                     text='Model', title='Efficiency Frontier: Accuracy vs Training Time',
                     size_max=15)

    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(height=500)

    return fig


def plot_convergence_comparison(data: Dict):
    """Bar chart comparing convergence metrics."""
    models = []
    epochs_90 = []
    epochs_95 = []
    epochs_99 = []

    for model_name, model_data in data['models'].items():
        stats = model_data['aggregated']['statistics']
        models.append(model_name)

        epochs_90.append(stats.get('epochs_to_90_percent', {}).get('mean') or 0)
        epochs_95.append(stats.get('epochs_to_95_percent', {}).get('mean') or 0)
        epochs_99.append(stats.get('epochs_to_99_percent', {}).get('mean') or 0)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=models, y=epochs_90, name='90% Accuracy'))
    fig.add_trace(go.Bar(x=models, y=epochs_95, name='95% Accuracy'))
    fig.add_trace(go.Bar(x=models, y=epochs_99, name='99% Accuracy'))

    fig.update_layout(
        title='Convergence Speed: Epochs to Reach Accuracy Thresholds',
        xaxis_title='Model',
        yaxis_title='Epochs',
        barmode='group',
        height=500
    )

    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Benchmark Dashboard", layout="wide")

    st.title("🔬 Vision Transformer Benchmark Dashboard")
    st.markdown("---")

    # Sidebar for configuration
    st.sidebar.title("Configuration")

    # Path input
    default_path = "results/benchmark_mnist_20251111_113441"
    benchmark_path = st.sidebar.text_input(
        "Benchmark Results Directory",
        value=default_path,
        help="Path to benchmark results directory"
    )

    # Try to load data
    try:
        data = load_benchmark_data(benchmark_path)
        st.sidebar.success(f"✅ Loaded {len(data['models'])} models")

        # Show config info
        if data['config']:
            st.sidebar.subheader("Benchmark Info")
            st.sidebar.write(f"Dataset: {data['config'].get('dataset', 'N/A')}")
            st.sidebar.write(f"Epochs: {data['config'].get('epochs', 'N/A')}")
            st.sidebar.write(f"Batch Size: {data['config'].get('batch_size', 'N/A')}")
            st.sidebar.write(f"Seeds: {data['config'].get('seeds', 'N/A')}")

    except Exception as e:
        st.error(f"❌ Error loading benchmark: {e}")
        st.info("Please enter a valid benchmark results directory path.")
        return

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Accuracy",
        "⏱️ Training Dynamics",
        "⚡ Efficiency",
        "🎯 Convergence",
        "🔍 Per-Run Details"
    ])

    with tab1:
        st.header("Summary Statistics")
        summary_df = create_summary_table(data)
        st.dataframe(summary_df, use_container_width=True)

        # Key insights
        st.subheader("Key Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Best model by accuracy
            best_model = None
            best_acc = 0
            for model_name, model_data in data['models'].items():
                stats = model_data['aggregated']['statistics']
                if 'best_test_accuracy' in stats:
                    acc = stats['best_test_accuracy']['mean']
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model_name

            if best_model:
                st.metric("🏆 Best Accuracy", f"{best_acc:.2f}%", f"{best_model}")

        with col2:
            # Fastest training
            fastest_model = None
            fastest_time = float('inf')
            for model_name, model_data in data['models'].items():
                stats = model_data['aggregated']['statistics']
                if 'avg_train_time_per_epoch' in stats:
                    time = stats['avg_train_time_per_epoch']['mean']
                    if time < fastest_time:
                        fastest_time = time
                        fastest_model = model_name

            if fastest_model:
                st.metric("⚡ Fastest Training", f"{fastest_time:.2f}s", f"{fastest_model}")

        with col3:
            # Fastest inference
            fastest_inf = None
            fastest_lat = float('inf')
            for model_name, model_data in data['models'].items():
                stats = model_data['aggregated']['statistics']
                if 'inference_latency_mean_ms' in stats:
                    lat = stats['inference_latency_mean_ms']['mean']
                    if lat < fastest_lat:
                        fastest_lat = lat
                        fastest_inf = model_name

            if fastest_inf:
                st.metric("🚀 Fastest Inference", f"{fastest_lat:.2f}ms", f"{fastest_inf}")

    with tab2:
        st.header("Accuracy Analysis")
        fig = plot_accuracy_comparison(data)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Solid lines show mean accuracy. Dashed lines show percentiles (5th, 25th, 75th, 95th). Shaded regions: light = 5-95% range (90% of runs), darker = 25-75% range (IQR).")

    with tab3:
        st.header("Training Dynamics")

        metric_choice = st.radio("Select Metric", ['accuracy', 'loss'], horizontal=True)

        fig = plot_training_curves(data, metric=metric_choice)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Lines show mean across runs. Light shaded region: 5-95 percentile (90% of runs). Darker shaded region: 25-75 percentile (interquartile range, 50% of runs).")

    with tab4:
        st.header("Efficiency Analysis")

        st.subheader("Training Time & Inference Latency")
        fig = plot_efficiency_comparison(data)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Efficiency Frontier")
        fig = plot_efficiency_scatter(data)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Ideal models are in the top-left: high accuracy, low training time.")

    with tab5:
        st.header("Convergence Analysis")

        fig = plot_convergence_comparison(data)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Lower bars = faster convergence. Models that don't reach a threshold show 0.")

    with tab6:
        st.header("Per-Run Details")

        # Select model
        model_names = list(data['models'].keys())
        selected_model = st.selectbox("Select Model", model_names)

        if selected_model:
            model_data = data['models'][selected_model]

            # Select run
            run_options = [f"Run {i} (seed {run['metadata']['seed']})"
                          for i, run in enumerate(model_data['runs'])]
            selected_run_idx = st.selectbox("Select Run", range(len(run_options)),
                                           format_func=lambda x: run_options[x])

            if selected_run_idx is not None:
                run_data = model_data['runs'][selected_run_idx]

                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Accuracy", f"{run_data['aggregate']['best_test_accuracy']:.2f}%")
                with col2:
                    st.metric("Final Accuracy", f"{run_data['aggregate']['final_test_accuracy']:.2f}%")
                with col3:
                    st.metric("Train Time", f"{run_data['aggregate']['avg_train_time_per_epoch']:.2f}s")
                with col4:
                    st.metric("Total Params", f"{run_data['aggregate']['total_parameters']:,}")

                # Per-epoch data
                st.subheader("Per-Epoch Training History")
                per_epoch_df = pd.DataFrame(run_data['per_epoch'])
                st.dataframe(per_epoch_df, use_container_width=True)

                # Full JSON
                with st.expander("View Full Metrics JSON"):
                    st.json(run_data)


if __name__ == "__main__":
    main()
