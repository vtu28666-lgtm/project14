import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

def generate_industrial_dataset(filename="industrial_vibration_dataset.csv", num_samples=3000):
    """
    Synthesizes calibrated vibration windows matching the NASA Bearing / MIMII profiles:
    - Class 0: Normal Operation (Low variance, Gaussian, low kurtosis) -> Aggressive Approx
    - Class 1: Incipient / Mild Fault (Micro-shocks, rising crest factor) -> Mild Approx
    - Class 2: Severe Fault / Spalling (High impacts, heavy tails, high kurtosis) -> Exact Precision
    """
    np.random.seed(42)
    records = []
    chunk_size = 512

    for i in range(num_samples):
        label_type = np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
        
        if label_type == 0:
            # Normal steady vibration
            raw_signal = np.random.normal(loc=0.0, scale=0.20, size=chunk_size)
            target_mode = "aggressive_approx"
            anomaly_flag = 0
        elif label_type == 1:
            # Mild fault (bearing inner race defect harmonic)
            t = np.linspace(0, 1, chunk_size)
            carrier = np.random.normal(0, 0.25, size=chunk_size)
            fault_harmonic = 0.45 * np.sin(2 * np.pi * 120 * t)
            raw_signal = carrier + fault_harmonic
            target_mode = "mild_approx"
            anomaly_flag = 0
        else:
            # Severe fault (cavitation / cracked outer race shock pulses)
            raw_signal = np.random.normal(loc=0.0, scale=0.35, size=chunk_size)
            shock_indices = np.random.choice(chunk_size, size=15, replace=False)
            raw_signal[shock_indices] += np.random.uniform(1.8, 3.2, size=15) * np.random.choice([-1, 1], size=15)
            target_mode = "exact"
            anomaly_flag = 1

        # Extract standard industrial condition-monitoring features
        rms = float(np.sqrt(np.mean(raw_signal**2)))
        peak = float(np.max(np.abs(raw_signal)))
        kurt_val = float(kurtosis(raw_signal))
        skew_val = float(skew(raw_signal))
        var_val = float(np.var(raw_signal))
        crest_factor = float(peak / (rms + 1e-6))

        records.append({
            "sample_id": i + 1,
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "kurtosis": round(kurt_val, 3),
            "skewness": round(skew_val, 3),
            "variance": round(var_val, 4),
            "crest_factor": round(crest_factor, 3),
            "target_mode": target_mode,
            "is_anomaly": anomaly_flag,
            "signal_preview": ",".join([str(round(v, 2)) for v in raw_signal[:16]]) # sample preview
        })

    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Dataset generated successfully: {filename} ({len(df)} samples)")

if __name__ == "__main__":
    generate_industrial_dataset()