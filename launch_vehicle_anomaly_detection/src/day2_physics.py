import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INITIAL_PRESSURE = 5000   # Starting tank pressure (arbitrary units)
DECAY_RATE       = 0.008  # Controls how fast pressure drops (higher = faster decay)
NOISE_STD        = 50     # Standard deviation of Gaussian noise (units)


def simulate_pressure(time_vector: np.ndarray) -> np.ndarray:
    """
    Simulates rocket fuel tank pressure over a flight.

    Physics model
    -------------
    Fuel is consumed continuously, so tank pressure decays exponentially:

        P(t) = P0 * exp(-k * t)  +  noise

    Where:
        P0    – initial pressure (5000 units)
        k     – decay rate constant (controls consumption speed)
        noise – Gaussian random noise ~ N(0, NOISE_STD)

    Parameters
    ----------
    time_vector : np.ndarray
        1-D array of time values (in seconds).

    Returns
    -------
    pressure : np.ndarray
        1-D array of pressure values (same shape as time_vector).
    """
    # Deterministic exponential decay
    clean_signal = INITIAL_PRESSURE * np.exp(-DECAY_RATE * time_vector)

    # Gaussian noise – models sensor jitter / micro-turbulence
    noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=time_vector.shape)

    pressure = clean_signal + noise
    return pressure


# ------------------------------------------------------------------
# Quick self-test / visual check when run directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Use the same time vector convention as day1_generator.py
    DURATION_SECONDS = 600
    SAMPLING_RATE_HZ = 10
    TOTAL_POINTS     = DURATION_SECONDS * SAMPLING_RATE_HZ

    time     = np.linspace(0, DURATION_SECONDS, TOTAL_POINTS)
    pressure = simulate_pressure(time)

    print("=== simulate_pressure() Quick Stats ===")
    print(f"  Time points   : {len(time)}")
    print(f"  Pressure start: {pressure[0]:.2f} units")
    print(f"  Pressure end  : {pressure[-1]:.2f} units")
    print(f"  Min / Max     : {pressure.min():.2f} / {pressure.max():.2f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(time, pressure, color='royalblue', linewidth=0.8, label='Tank Pressure')
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (units)")
    plt.title("Day 2 – Simulated Fuel Tank Pressure (Exponential Decay + Noise)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
