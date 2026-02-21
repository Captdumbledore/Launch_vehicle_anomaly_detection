import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# CONFIGURATION DEFAULTS
# ---------------------------------------------------------------
DEFAULT_MAGNITUDE   = 100    # How big a spike is (±magnitude)
DEFAULT_PROBABILITY = 0.01   # Probability that any given sample gets a spike


def inject_spike(
    signal: np.ndarray,
    magnitude: float = DEFAULT_MAGNITUDE,
    probability: float = DEFAULT_PROBABILITY,
) -> np.ndarray:
    """
    Inject sudden, large spikes into a signal at random positions.

    Model
    -----
    A spike at index i means the sensor momentarily reports an
    extreme value (e.g., a transient voltage surge, bit-flip, or
    sudden pressure drop).  Each sample is independently spiked
    with probability `probability`:

        corrupted[i] = signal[i] + spike_i

    where spike_i ~ Uniform(-magnitude, +magnitude) if spiked,
    else 0.

    Parameters
    ----------
    signal      : np.ndarray  – 1-D original sensor readings.
    magnitude   : float       – Maximum absolute spike amplitude
                                (added on TOP of the real value).
    probability : float       – Probability [0, 1] that any single
                                sample is spiked. Default 0.01 = 1%.

    Returns
    -------
    corrupted   : np.ndarray  – Copy of `signal` with spikes added.
                                Shape is identical to input.

    Notes
    -----
    * Returns a **copy** — the original array is never mutated.
    * Spike direction is random (can be positive or negative).
    * Use np.random.seed() before calling for reproducible results.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if magnitude < 0:
        raise ValueError(f"magnitude must be >= 0, got {magnitude}")

    corrupted = signal.copy()
    n = len(signal)

    # Boolean mask: True where a spike will be injected
    spike_mask = np.random.random(n) < probability

    # Random spike amplitudes: uniform in [-magnitude, +magnitude]
    spike_amplitudes = np.random.uniform(-magnitude, magnitude, size=n)

    # Apply only where mask is True
    corrupted[spike_mask] += spike_amplitudes[spike_mask]

    spike_indices = np.where(spike_mask)[0]
    print(f"[inject_spike] {len(spike_indices)} spike(s) injected "
          f"out of {n} samples "
          f"({100 * len(spike_indices) / n:.2f}% hit rate).")

    return corrupted


# ---------------------------------------------------------------
# Quick self-test / visual check when run directly
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Reproduce the Day 2 pressure signal as the test input
    DURATION_SECONDS = 600
    SAMPLING_RATE_HZ = 10
    TOTAL_POINTS     = DURATION_SECONDS * SAMPLING_RATE_HZ

    np.random.seed(42)    # reproducible demo

    time     = np.linspace(0, DURATION_SECONDS, TOTAL_POINTS)
    # Clean exponential decay signal (from day2_physics.py)
    clean    = 5000 * np.exp(-0.008 * time)

    # Inject spikes
    corrupted = inject_spike(clean, magnitude=500, probability=0.02)

    # --- Stats ---
    print("\n=== inject_spike() Quick Stats ===")
    print(f"  Samples total  : {TOTAL_POINTS}")
    print(f"  Clean   max    : {clean.max():.2f}  |  min: {clean.min():.2f}")
    print(f"  Corrupt max    : {corrupted.max():.2f}  |  min: {corrupted.min():.2f}")

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(time, clean,     color="steelblue",  linewidth=0.8, label="Clean Signal",   alpha=0.7)
    plt.plot(time, corrupted, color="tomato",      linewidth=0.8, label="Spiked Signal",  alpha=0.9)

    # Highlight spike locations
    diff = corrupted - clean
    spike_times = time[diff != 0]
    spike_vals  = corrupted[diff != 0]
    plt.scatter(spike_times, spike_vals, color="red", s=20, zorder=5, label="Spike Points")

    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (units)")
    plt.title("Day 3 – inject_spike(): Sudden Anomaly Injection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
