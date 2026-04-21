"""Generate publication-quality sensor data visualizations.

This script creates synthetic temperature sensor data using NumPy
and produces scatter, histogram, and box plot visualizations saved
as PNG files.

Usage
-----
    python generate_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(seed):
    """Generate synthetic temperature sensor readings.

    Parameters
    ----------
    seed : int
        Random number generator seed for reproducibility.

    Returns
    -------
    sensor_a : numpy.ndarray
        Temperature readings from sensor A in Celsius, shape (200,).
    sensor_b : numpy.ndarray
        Temperature readings from sensor B in Celsius, shape (200,).
    timestamps : numpy.ndarray
        Measurement timestamps in seconds, shape (200,).
    """
    rng = np.random.default_rng(seed=seed)
    sensor_a = rng.normal(loc=25.0, scale=3.0, size=200)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=200)
    timestamps = rng.uniform(low=0.0, high=10.0, size=200)
    return sensor_a, sensor_b, timestamps


def plot_scatter(sensor_a, sensor_b, timestamps, ax):
    """Draw a scatter plot of sensor readings vs timestamps.

    Parameters
    ----------
    sensor_a : numpy.ndarray
        Temperature readings from sensor A, shape (200,).
    sensor_b : numpy.ndarray
        Temperature readings from sensor B, shape (200,).
    timestamps : numpy.ndarray
        Measurement timestamps in seconds, shape (200,).
    ax : matplotlib.axes.Axes
        Axes object to draw the plot on.

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


def plot_histogram(sensor_a, sensor_b, ax):
    """Draw overlaid histograms of sensor temperature distributions.

    Parameters
    ----------
    sensor_a : numpy.ndarray
        Temperature readings from sensor A, shape (200,).
    sensor_b : numpy.ndarray
        Temperature readings from sensor B, shape (200,).
    ax : matplotlib.axes.Axes
        Axes object to draw the plot on.

    Returns
    -------
    None
    """
    bins = np.linspace(min(sensor_a.min(), sensor_b.min()), max(sensor_a.max(), sensor_b.max()), 31)
    ax.hist(sensor_a, bins=bins, alpha=0.5, color='blue', label='Sensor A')
    ax.hist(sensor_b, bins=bins, alpha=0.5, color='orange', label='Sensor B')
    ax.axvline(sensor_a.mean(), color='blue', linestyle='--', linewidth=1.5, label=f'Mean A ({sensor_a.mean():.2f}°C)')
    ax.axvline(sensor_b.mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Mean B ({sensor_b.mean():.2f}°C)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Sensor Temperatures')
    ax.legend()
    ax.grid(alpha=0.3)


def plot_boxplot(sensor_a, sensor_b, ax):
    """Draw a side-by-side box plot of sensor temperature distributions.

    Parameters
    ----------
    sensor_a : numpy.ndarray
        Temperature readings from sensor A, shape (200,).
    sensor_b : numpy.ndarray
        Temperature readings from sensor B, shape (200,).
    ax : matplotlib.axes.Axes
        Axes object to draw the plot on.

    Returns
    -------
    None
    """
    overall_mean = np.concatenate([sensor_a, sensor_b]).mean()
    ax.boxplot([sensor_a, sensor_b], tick_labels=['Sensor A', 'Sensor B'], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red'))
    ax.set_ylabel('Temperature (deg C)')
    ax.set_title('Sensor Temperature Comparison')
    ax.axhline(overall_mean, color='gray', linestyle='--', linewidth=1.5, label=f'Overall mean ({overall_mean:.2f} deg C)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def main():
    """Generate sensor data and save all three plots as sensor_analysis.png.

    Returns
    -------
    None
    """
    sensor_a, sensor_b, timestamps = generate_data(seed=1522)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_scatter(sensor_a, sensor_b, timestamps, axes[0])
    plot_histogram(sensor_a, sensor_b, axes[1])
    plot_boxplot(sensor_a, sensor_b, axes[2])
    plt.tight_layout()
    plt.savefig('sensor_analysis.png', dpi=150, bbox_inches='tight')
    print('Saved sensor_analysis.png')


if __name__ == '__main__':
    main()