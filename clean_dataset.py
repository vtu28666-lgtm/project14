import os
import glob
import time
import traceback
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew


class IndustrialDataCleaner:
    def __init__(
        self,
        sampling_rate: int = 20000,
        chunk_size: int = 512,
        overlap: float = 0.5,
        lowcut: float = 10.0,
        highcut: float = 8000.0
    ):
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.step_size = max(1, int(chunk_size * (1.0 - overlap)))
        self.lowcut = lowcut
        self.highcut = highcut

        # Pre-compute filter coefficients
        nyquist = 0.5 * self.sampling_rate
        low = max(0.001, self.lowcut / nyquist)
        high = min(0.999, self.highcut / nyquist)
        if low < high:
            self.b, self.a = signal.butter(4, [low, high], btype="band")
        else:
            self.b, self.a = None, None

    def clean_raw_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Cleans vibration telemetry:
        1. Replace infs and interpolate/fill NaNs.
        2. Detrend static DC offset.
        3. 4th-order Butterworth bandpass filter.
        4. Remove non-physical electrical sensor saturation spikes (> 8 std dev).
        """
        if len(raw_signal) == 0:
            return np.zeros(self.chunk_size, dtype=np.float32)

        # 1. Handle NaNs and Infs
        series = pd.Series(raw_signal)
        series = series.replace([np.inf, -np.inf], np.nan)
        series = series.interpolate(method="linear").bfill().ffill()

        if series.isna().all() or len(series) == 0:
            return np.zeros(max(len(series), self.chunk_size), dtype=np.float32)

        cleaned = series.to_numpy(dtype=np.float32)

        # 2. Detrend (removes static DC bias)
        cleaned = signal.detrend(cleaned)

        # 3. Butterworth 4th-order Bandpass Filter
        if self.b is not None and len(cleaned) > 30:
            try:
                cleaned = signal.filtfilt(self.b, self.a, cleaned)
            except Exception:
                pass

        # 4. Remove extreme electrical sensor saturation
        std_val = float(np.std(cleaned))
        if std_val > 1e-6:
            mean_val = float(np.mean(cleaned))
            cleaned = np.clip(cleaned, mean_val - 8 * std_val, mean_val + 8 * std_val)

        return cleaned.astype(np.float32)

    def extract_window_features(self, window: np.ndarray) -> dict:
        """Calculates 6 time-domain statistical features for a single window."""
        rms = float(np.sqrt(np.mean(window**2)))
        peak = float(np.max(np.abs(window)))
        kurt_val = float(kurtosis(window))
        skew_val = float(skew(window))
        var_val = float(np.var(window))
        crest_factor = float(peak / (rms + 1e-7))

        if kurt_val > 3.5 or crest_factor > 4.5 or rms > 0.40 or peak > 2.0:
            target_mode = "exact"
            is_anomaly = 1
        elif kurt_val > 2.0 or crest_factor > 2.8 or rms > 0.15 or peak > 0.6:
            target_mode = "mild_approx"
            is_anomaly = 0
        else:
            target_mode = "aggressive_approx"
            is_anomaly = 0

        return {
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "kurtosis": round(kurt_val, 3),
            "skewness": round(skew_val, 3),
            "variance": round(var_val, 4),
            "crest_factor": round(crest_factor, 3),
            "target_mode": target_mode,
            "is_anomaly": is_anomaly
        }

    def extract_windows_features_vectorized(self, cleaned_signal: np.ndarray, source_file: str = "") -> List[dict]:
        """
        Fast vectorized sliding-window extraction for entire signal traces.
        """
        total_len = len(cleaned_signal)
        if total_len < self.chunk_size:
            return []

        # Stride sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(cleaned_signal, window_shape=self.chunk_size)[::self.step_size]
        
        rms = np.sqrt(np.mean(windows**2, axis=1))
        peak = np.max(np.abs(windows), axis=1)
        var_val = np.var(windows, axis=1)
        kurt_val = kurtosis(windows, axis=1)
        skew_val = skew(windows, axis=1)
        crest_factor = peak / (rms + 1e-7)

        # Mode classification rules
        is_exact = (kurt_val > 3.5) | (crest_factor > 4.5) | (rms > 0.40) | (peak > 2.0)
        is_mild = ~is_exact & ((kurt_val > 2.0) | (crest_factor > 2.8) | (rms > 0.15) | (peak > 0.6))
        modes = np.where(is_exact, "exact", np.where(is_mild, "mild_approx", "aggressive_approx"))
        is_anomaly = np.where(is_exact, 1, 0)

        n = len(windows)
        records = [
            {
                "rms": round(float(rms[i]), 4),
                "peak": round(float(peak[i]), 4),
                "kurtosis": round(float(kurt_val[i]), 3),
                "skewness": round(float(skew_val[i]), 3),
                "variance": round(float(var_val[i]), 4),
                "crest_factor": round(float(crest_factor[i]), 3),
                "target_mode": str(modes[i]),
                "is_anomaly": int(is_anomaly[i]),
                "source_file": source_file
            }
            for i in range(n)
        ]
        return records

    def load_signal_from_file(self, filepath: str) -> np.ndarray:
        """
        Loads signal from CSV, TSV, space-delimited, or unextended raw files safely and quickly.
        """
        try:
            with open(filepath, "r", errors="ignore") as f:
                first_line = f.readline()
            if not first_line.strip():
                return np.array([], dtype=np.float32)

            if "\t" in first_line:
                # Tab-separated (NASA format)
                df = pd.read_csv(filepath, sep="\t", engine="c", header=None, on_bad_lines="skip")
                return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=np.float32)
            elif "," in first_line:
                # Comma-separated
                df = pd.read_csv(filepath, sep=",", engine="c", header=None, on_bad_lines="skip")
                for col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(s) >= self.chunk_size:
                        return s.to_numpy(dtype=np.float32)
            else:
                # Space or whitespace separated
                df = pd.read_csv(filepath, sep=r"\s+", engine="python", header=None, on_bad_lines="skip")
                return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=np.float32)
        except Exception:
            try:
                return np.loadtxt(filepath, dtype=np.float32)
            except Exception:
                return np.array([], dtype=np.float32)

        return np.array([], dtype=np.float32)

    def process_directory(
        self,
        input_dir: str,
        max_files: Optional[int] = None
    ) -> List[dict]:
        """
        Cleans and segments ALL files in input_dir (and subdirectories).
        When max_files is None, it processes the ENTIRE dataset without truncation.
        """
        if not os.path.exists(input_dir):
            print(f"[Cleaner] Path not found: {input_dir}")
            return []

        all_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                # Ignore non-data documents
                if not file.endswith((".pdf", ".zip", ".tar", ".gz", ".py", ".json", ".md", ".bat")):
                    all_files.append(os.path.join(root, file))

        all_files.sort()
        total_found = len(all_files)
        if total_found == 0:
            print(f"[Cleaner] No data files found in: {input_dir}")
            return []

        if max_files and max_files > 0 and total_found > max_files:
            indices = np.linspace(0, total_found - 1, max_files, dtype=int)
            selected_files = [all_files[i] for i in indices]
        else:
            selected_files = all_files

        print(f"[Cleaner] Cleaning ENTIRE dataset ({len(selected_files)} files) from: {input_dir}...")
        t_start = time.time()
        dataset_records = []

        for idx, filepath in enumerate(selected_files):
            try:
                raw_data = self.load_signal_from_file(filepath)
                if len(raw_data) < self.chunk_size:
                    continue

                cleaned_data = self.clean_raw_signal(raw_data)
                records = self.extract_windows_features_vectorized(
                    cleaned_data,
                    source_file=os.path.basename(filepath)
                )
                dataset_records.extend(records)

            except Exception as e:
                pass

            if (idx + 1) % 200 == 0 or (idx + 1) == len(selected_files):
                elapsed = time.time() - t_start
                rate = (idx + 1) / (elapsed + 1e-6)
                print(f"  Processed {idx + 1}/{len(selected_files)} files ({rate:.1f} files/sec) -> {len(dataset_records)} windows extracted")

        total_time = time.time() - t_start
        print(f"[Cleaner] Completed directory in {total_time:.1f}s ({len(dataset_records)} total windows).")
        return dataset_records


def clean_entire_dataset(output_csv: str = "industrial_vibration_dataset.csv") -> pd.DataFrame:
    """
    Cleans the entire dataset across all tests and sources:
    1. Entire NASA Bearing Dataset (all files in 2nd_test, 1st_test, etc.)
    2. All local benchmark traces (temp_raw_data/trace_01.csv, test_trace.csv, bearing_flaw.csv, fan_unbalance.csv)
    3. Balanced physical augmentations if needed.
    """
    print("=" * 70)
    print("STARTING FULL DATASET CLEANING PIPELINE")
    print("=" * 70)

    cleaner = IndustrialDataCleaner(sampling_rate=20000, chunk_size=512, overlap=0.5)
    all_records = []

    # 1. Clean entire Kaggle NASA Bearing Dataset
    kaggle_dir = os.path.expanduser(r"~/.cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1")
    if os.path.exists(kaggle_dir):
        # We clean the entire 2nd_test (984 files from healthy to severe outer-race failure)
        test2_dir = os.path.join(kaggle_dir, "2nd_test", "2nd_test")
        if os.path.exists(test2_dir):
            print(f"\n[1/3] Processing entire 2nd_test run-to-failure dataset (984 files)...")
            records_test2 = cleaner.process_directory(test2_dir, max_files=None)
            all_records.extend(records_test2)

        # Also clean 1st_test (2,156 files)
        test1_dir = os.path.join(kaggle_dir, "1st_test", "1st_test")
        if os.path.exists(test1_dir):
            print(f"\n[2/3] Processing 1st_test dataset...")
            # Clean 1st_test fully
            records_test1 = cleaner.process_directory(test1_dir, max_files=None)
            all_records.extend(records_test1)
    else:
        print(f"[Cleaner] Kaggle bearing-dataset not found in cache. Checking local paths...")

    # 2. Clean workspace scenario traces & temp raw traces
    print(f"\n[3/3] Processing local workspace scenario traces and temp raw data...")
    local_trace_files = ["bearing_flaw.csv", "test_trace.csv", "fan_unbalance.csv", os.path.join("temp_raw_data", "trace_01.csv")]
    for trace_file in local_trace_files:
        if os.path.exists(trace_file):
            raw = cleaner.load_signal_from_file(trace_file)
            if len(raw) >= cleaner.chunk_size:
                cleaned = cleaner.clean_raw_signal(raw)
                recs = cleaner.extract_windows_features_vectorized(cleaned, source_file=os.path.basename(trace_file))
                all_records.extend(recs)
                print(f"  Loaded {len(recs)} windows from {trace_file}")

    # 3. Compile DataFrame & hygiene
    if not all_records:
        print("[Cleaner] No external records found. Generating calibrated benchmark signals...")
        # Fallback generator for self-contained testing
        from prepare_dataset import generate_industrial_dataset
        generate_industrial_dataset(output_csv, num_samples=5000)
        return pd.read_csv(output_csv)

    df = pd.DataFrame(all_records)
    print(f"\nTotal raw windows extracted: {len(df)}")

    # Add sample_id
    df["sample_id"] = np.arange(1, len(df) + 1)

    # Balance check: ensure sufficient samples in each class
    counts = df["target_mode"].value_counts().to_dict()
    print("Initial Class Distribution:\n", df["target_mode"].value_counts())

    min_samples = 200
    for mode_name in ["aggressive_approx", "mild_approx", "exact"]:
        if counts.get(mode_name, 0) < min_samples:
            deficit = min_samples - counts.get(mode_name, 0)
            print(f"[Cleaner] Augmenting {deficit} calibrated samples for mode '{mode_name}' to maintain robust balance...")
            np.random.seed(42)
            aug_records = []
            for _ in range(deficit):
                if mode_name == "aggressive_approx":
                    sig = np.random.normal(0, 0.06, cleaner.chunk_size).astype(np.float32)
                elif mode_name == "mild_approx":
                    t = np.linspace(0, 0.025, cleaner.chunk_size)
                    sig = (np.random.normal(0, 0.12, cleaner.chunk_size) + 0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
                else:
                    sig = np.random.normal(0, 0.22, cleaner.chunk_size).astype(np.float32)
                    imp_idx = np.random.choice(cleaner.chunk_size, 15, replace=False)
                    sig[imp_idx] += np.random.uniform(2.0, 3.5, 15) * np.random.choice([-1, 1], 15)

                f = cleaner.extract_window_features(cleaner.clean_raw_signal(sig))
                f["target_mode"] = mode_name
                f["sample_id"] = len(df) + len(aug_records) + 1
                f["source_file"] = f"calibrated_{mode_name}.trace"
                aug_records.append(f)
            df = pd.concat([df, pd.DataFrame(aug_records)], ignore_index=True)

    # Hygiene: Drop identical duplicates
    df = df.drop_duplicates(subset=["rms", "peak", "kurtosis", "variance"])
    df["sample_id"] = np.arange(1, len(df) + 1)

    # Save final cleaned dataset
    df.to_csv(output_csv, index=False)

    print("=" * 70)
    print("DATASET CLEANING COMPLETE")
    print(f"Output File: {output_csv}")
    print(f"Total Cleaned Samples: {len(df):,}")
    print("Final Target Mode Distribution:")
    for mode, cnt in df["target_mode"].value_counts().items():
        pct = (cnt / len(df)) * 100
        print(f"  {mode:<20}: {cnt:>7,} ({pct:5.1f}%)")
    print("=" * 70)
    return df


if __name__ == "__main__":
    clean_entire_dataset(output_csv="industrial_vibration_dataset.csv")
