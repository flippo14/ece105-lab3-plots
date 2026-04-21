"""Generate publication-quality sensor data visualizations.

This script creates synthetic temperature sensor data using NumPy
and produces scatter, histogram, and box plot visualizations saved
as PNG files.

Usage
-----
    python generate_plots.py
"""

import numpy as np


def generate_data(seed: int):
    """Generate synthetic temperature time series for two sensors.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sensor_a : numpy.ndarray
        1D array of shape (200,) with float64 temperature values (°C)
        for sensor A.
    sensor_b : numpy.ndarray
        1D array of shape (200,) with float64 temperature values (°C)
        for sensor B.
    timestamps : numpy.ndarray
        1D array of shape (200,) with float64 timestamps sampled uniformly
        between 0.0 and 10.0 seconds.

    Notes
    -----
    The generated series are drawn from normal and uniform distributions
    using a numpy.default_rng for reproducible, independent streams.
    """
    rng = np.random.default_rng(seed)
    n = 200
    timestamps = rng.uniform(low=0.0, high=10.0, size=n).astype(np.float64)

    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n).astype(np.float64)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n).astype(np.float64)

    return sensor_a, sensor_b, timestamps


if __name__ == "__main__":
    # quick sanity check when run as a script
    a, b, ts = generate_data(1522)
    print(f"sensor_a.shape={a.shape}, sensor_b.shape={b.shape}, timestamps.shape={ts.shape}")
