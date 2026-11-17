import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def clean_and_segment_signal(
    raw_signal: np.ndarray,
    chunk_size: int = 512,
    overlap: float = 0.5,
    min_variance: float = 1e-5
) -> pd.DataFrame:
    """
    Step 2 Processing Pipeline:
    1. Sliding window chunking matching hardware MAC buffer size.
    2. Drops flatlined/dead-sensor windows (variance < min_variance).
    3. Handles infinite or corrupt calculations per window.
    4. Computes 6 core time-domain statistical metrics:
       - RMS, Peak, Kurtosis, Skewness, Variance, Crest Factor.
    5. Applies dynamic physical labeling for precision modes.
    """
    step = max(1, int(chunk_size * (1.0 - overlap)))
    total_samples = len(raw_signal)
    records = []
    dropped_dead_windows = 0

    print(f"[Step 2] Slicing signal of length {total_samples} into windows (chunk_size={chunk_size}, step={step})...")

    sample_id = 0
    for start in range(0, total_samples - chunk_size + 1, step):
        window = raw_signal[start : start + chunk_size]

        # 1. Clean window-level NaNs or Infs if present
        if np.isnan(window).any() or np.isinf(window).any():
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Check for dead sensor flatline (disconnect / zero telemetry)
        var_val = float(np.var(window))
        if var_val < min_variance:
            dropped_dead_windows += 1
            continue

        # 3. Compute Time-Domain Condition Indicators
        rms_val = float(np.sqrt(np.mean(window ** 2)))
        peak_val = float(np.max(np.abs(window)))
        kurt_val = float(kurtosis(window))
        skew_val = float(skew(window))
        crest_factor = float(peak_val / (rms_val + 1e-7))

        # 4. Target Mode Allocation (Ground Truth for AI Precision Controller)
        # - High impulse shock / outer-race cracking: Exact Mode (16-bit)
        # - Incipient harmonic fluctuation: Mild Approx (10-bit)
        # - Normal continuous background humming: Aggressive Approx (6-bit)
        if kurt_val > 3.5 or crest_factor > 4.5 or rms_val > 0.40 or peak_val > 2.0:
            target_mode = "exact"
            is_anomaly = 1
        elif kurt_val > 2.0 or crest_factor > 2.8 or rms_val > 0.15 or peak_val > 0.6:
            target_mode = "mild_approx"
            is_anomaly = 0
        else:
            target_mode = "aggressive_approx"
            is_anomaly = 0

        sample_id += 1
        records.append({
            "sample_id": sample_id,
            "rms": round(rms_val, 4),
            "peak": round(peak_val, 4),
            "kurtosis": round(kurt_val, 3),
            "skewness": round(skew_val, 3),
            "variance": round(var_val, 4),
            "crest_factor": round(crest_factor, 3),
            "target_mode": target_mode,
            "is_anomaly": is_anomaly
        })

    print(f"[Step 2] Processing complete. Filtered {dropped_dead_windows} flatlined windows.")
    return pd.DataFrame(records)


def run_step2_pipeline(
    input_file: str = "test_trace.csv",
    output_file: str = "step2_segmented_features.csv"
):
    """
    Loads raw or pre-filtered signal CSV, executes Step 2 windowing & cleaning,
    and writes the segmented feature table without overwriting the master dataset.
    """
    if not os.path.exists(input_file):
        print(f"[Step 2] Input trace '{input_file}' not found. Generating a calibrated benchmark...")
        np.random.seed(42)
        normal = np.random.normal(0, 0.12, 10000)
        harmonics = np.random.normal(0, 0.20, 5000) + 0.35 * np.sin(np.linspace(0, 10, 5000))
        spikes = np.random.normal(0, 0.35, 4000)
        impulse_idx = np.random.choice(4000, 80, replace=False)
        spikes[impulse_idx] += np.random.uniform(2.0, 3.5, 80)
        dead = np.zeros(2000)  # Simulating dead sensor dropouts
        synth_data = np.concatenate([normal, harmonics, spikes, dead, normal])
        pd.DataFrame({"vibration": synth_data}).to_csv(input_file, index=False)

    print(f"[Step 2] Reading signal stream from '{input_file}'...")
    try:
        df = pd.read_csv(input_file, sep=r"[\s,;]+", engine="python", on_bad_lines="skip")
        signal_data = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=np.float32)
    except Exception:
        df = pd.read_csv(input_file)
        signal_data = df.iloc[:, 0].to_numpy(dtype=np.float32)

    # Execute segmentation and statistical feature cleaning
    features_df = clean_and_segment_signal(signal_data, chunk_size=512, overlap=0.5)

    if not features_df.empty:
        features_df = features_df.drop_duplicates(subset=["rms", "peak", "kurtosis", "variance"])
        features_df.to_csv(output_file, index=False)
        print("-" * 60)
        print(f"[Step 2] Output written to: {output_file}")
        print(f"Total Usable Window Samples: {len(features_df)}")
        print("Class Balance:\n", features_df["target_mode"].value_counts())
        print("-" * 60)
    else:
        print("[Step 2] Warning: No features extracted from input.")


if __name__ == "__main__":
    run_step2_pipeline(
        input_file="test_trace.csv",
        output_file="step2_segmented_features.csv"
    )
