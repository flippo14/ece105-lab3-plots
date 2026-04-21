"""Generate publication-quality sensor data visualizations.

This module creates synthetic temperature sensor data and produces a
combined figure containing scatter, histogram, and boxplot panels plus
summary statistics.

Usage
-----
Run as a script::

    python generate_plots.py
"""

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def generate_data(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic temperature time series for two sensors.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility; used to construct a
        :class:`numpy.random.Generator` via :func:`np.random.default_rng`.

    Returns
    -------
    sensor_a : numpy.ndarray
        1D array of shape (200,) with float64 temperature values (°C)
        for sensor A, drawn from N(loc=25.0, scale=3.0).
    sensor_b : numpy.ndarray
        1D array of shape (200,) with float64 temperature values (°C)
        for sensor B, drawn from N(loc=27.0, scale=4.5).
    timestamps : numpy.ndarray
        1D array of shape (200,) with float64 timestamps sampled uniformly
        between 0.0 and 10.0 seconds.
    """
    rng = np.random.default_rng(seed)
    n = 200
    timestamps = rng.uniform(low=0.0, high=10.0, size=n).astype(np.float64)
    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n).astype(np.float64)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n).astype(np.float64)
    return sensor_a, sensor_b, timestamps


def plot_scatter(sensor_a: np.ndarray, sensor_b: np.ndarray, timestamps: np.ndarray, ax) -> None:
    """Plot two temperature time series as scatter points on ``ax``.

    Parameters
    ----------
    sensor_a : numpy.ndarray
        1D temperature array for sensor A.
    sensor_b : numpy.ndarray
        1D temperature array for sensor B.
    timestamps : numpy.ndarray
        1D array of time values (seconds).
    ax : matplotlib.axes.Axes
        Axes to draw into (modified in place).

    Returns
    -------
    None
    """
    ax.scatter(timestamps, sensor_a, color='blue', label='Sensor A', s=20, alpha=0.8)
    ax.scatter(timestamps, sensor_b, color='orange', label='Sensor B', s=20, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Sensor Temperatures vs Time')
    ax.legend()
    ax.grid(True)


def plot_histogram(sensor_a: np.ndarray, sensor_b: np.ndarray, ax, bins: int = 30) -> None:
    """Plot overlaid histograms for both sensors on ``ax``.

    Parameters
    ----------
    sensor_a : numpy.ndarray
        1D temperature array for sensor A.
    sensor_b : numpy.ndarray
        1D temperature array for sensor B.
    ax : matplotlib.axes.Axes
        Axes to draw into (modified in place).
    bins : int, optional
        Number of bins (default 30).

    Returns
    -------
    None
    """
    valid_min = np.nanmin([np.nanmin(sensor_a), np.nanmin(sensor_b)])
    valid_max = np.nanmax([np.nanmax(sensor_a), np.nanmax(sensor_b)])
    bin_edges = np.linspace(valid_min, valid_max, bins + 1)
    ax.hist(sensor_a, bins=bin_edges, alpha=0.5, color='blue', label='Sensor A')
    ax.hist(sensor_b, bins=bin_edges, alpha=0.5, color='orange', label='Sensor B')
    mean_a = np.nanmean(sensor_a)
    mean_b = np.nanmean(sensor_b)
    ax.axvline(mean_a, color='blue', linestyle='--', linewidth=1.5, label=f'Mean A ({mean_a:.2f}°C)')
    ax.axvline(mean_b, color='orange', linestyle='--', linewidth=1.5, label=f'Mean B ({mean_b:.2f}°C)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Sensor Temperatures')
    ax.legend()
    ax.grid(alpha=0.3)


def plot_boxplot(sensor_a: np.ndarray, sensor_b: np.ndarray, ax) -> None:
    """Plot side-by-side boxplots for the two sensors on ``ax``.

    Parameters
    ----------
    sensor_a : numpy.ndarray
        1D temperature array for sensor A.
    sensor_b : numpy.ndarray
        1D temperature array for sensor B.
    ax : matplotlib.axes.Axes
        Axes to draw into (modified in place).

    Returns
    -------
    None
    """
    overall_mean = np.nanmean(np.concatenate([sensor_a, sensor_b]))
    ax.boxplot([sensor_a, sensor_b], labels=['Sensor A', 'Sensor B'], widths=0.6,
               patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red'))
    ax.set_ylabel('Temperature (deg C)')
    ax.set_title('Sensor Temperature Comparison')
    ax.axhline(overall_mean, color='gray', linestyle='--', linewidth=1.5,
               label=f'Overall mean ({overall_mean:.2f} °C)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def main() -> None:
    """Generate data, draw a 2x2 figure and save as 'sensor_analysis.png'.

    The bottom-right panel displays summary statistics (mean and std) for
    each sensor as text.
    """
    sensor_a, sensor_b, timestamps = generate_data(seed=1522)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Top-left: scatter
    plot_scatter(sensor_a, sensor_b, timestamps, axes[0, 0])

    # Top-right: histogram
    plot_histogram(sensor_a, sensor_b, axes[0, 1])

    # Bottom-left: boxplot
    plot_boxplot(sensor_a, sensor_b, axes[1, 0])

    # Bottom-right: summary statistics
    ax_text = axes[1, 1]
    ax_text.axis('off')
    mean_a = np.nanmean(sensor_a)
    std_a = np.nanstd(sensor_a)
    mean_b = np.nanmean(sensor_b)
    std_b = np.nanstd(sensor_b)
    summary_lines = [
        'Summary Statistics',
        '',
        f'Sensor A: mean = {mean_a:.2f} °C',
        f'Sensor A: std  = {std_a:.2f} °C',
        '',
        f'Sensor B: mean = {mean_b:.2f} °C',
        f'Sensor B: std  = {std_b:.2f} °C',
    ]
    ax_text.text(0.5, 0.5, '\n'.join(summary_lines), ha='center', va='center', fontsize=12)

    fig.tight_layout()
    out_path = os.path.join('sensor_analysis.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
