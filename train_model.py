import os
import sys
import shutil
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_precision_model(
    dataset_path: str = "industrial_vibration_dataset.csv",
    model_output: str = "precision_model.json"
):
    print("=" * 70)
    print("AI PRECISION CONTROLLER: MODEL TRAINING PIPELINE")
    print("SPLIT CONFIGURATION: 80% TRAINING / 20% TESTING")
    print("=" * 70)

    # 1. Verify dataset existence and recover from master backup if needed
    master_backup = "industrial_vibration_master.csv"
    if (not os.path.exists(dataset_path) or os.path.getsize(dataset_path) < 100000) and os.path.exists(master_backup):
        print(f"[Training] Restoring comprehensive dataset from '{master_backup}'...")
        shutil.copyfile(master_backup, dataset_path)

    if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) < 100:
        print(f"[Training] Dataset '{dataset_path}' missing. Cleaning entire dataset first...")
        from clean_dataset import clean_entire_dataset
        clean_entire_dataset(dataset_path)

    print(f"\n[1/5] Loading cleaned dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    total_samples = len(df)
    print(f"  Total Cleaned Samples in Dataset: {total_samples:,}")

    # 2. Extract feature matrix & target labels
    feature_cols = ["rms", "peak", "kurtosis", "skewness", "variance", "crest_factor"]
    df = df.dropna(subset=feature_cols + ["target_mode"])

    X = df[feature_cols].astype(np.float32)
    mode_map = {"aggressive_approx": 0, "mild_approx": 1, "exact": 2}
    inv_map = {0: "Aggressive Approx (6-Bit)", 1: "Mild Approx (10-Bit)", 2: "Exact Precision (16-Bit)"}
    
    y = df["target_mode"].map(mode_map)
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    print("\n[2/5] Full Dataset Class Distribution:")
    for cls_idx, cls_name in inv_map.items():
        count = int((y == cls_idx).sum())
        pct = (count / len(y)) * 100
        print(f"  Class {cls_idx} [{cls_name:<26}]: {count:>7,} samples ({pct:5.1f}%)")

    # 3. Train / Test Split (Exact 80% Train, 20% Test)
    print("\n[3/5] Splitting dataset: 80% Training (train_size=0.80) and 20% Testing (test_size=0.20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        train_size=0.80,
        random_state=42,
        stratify=y
    )

    n_train = len(X_train)
    n_test = len(X_test)
    print(f"  Training Set (80%): {n_train:>7,} samples ({n_train / len(X) * 100:.1f}%)")
    print(f"  Testing Set  (20%): {n_test:>7,} samples ({n_test / len(X) * 100:.1f}%)")

    print("\n  Training Set (80%) Class Breakdown:")
    for cls_idx, cls_name in inv_map.items():
        c_train = int((y_train == cls_idx).sum())
        print(f"    Class {cls_idx} [{cls_name:<26}]: {c_train:>7,} samples ({c_train / n_train * 100:5.1f}%)")

    print("\n  Testing Set (20%) Class Breakdown:")
    for cls_idx, cls_name in inv_map.items():
        c_test = int((y_test == cls_idx).sum())
        print(f"    Class {cls_idx} [{cls_name:<26}]: {c_test:>7,} samples ({c_test / n_test * 100:5.1f}%)")

    # 4. Train Model on 80% Training Set
    print(f"\n[4/5] Training XGBoost Multi-Class Classifier on 80% training set ({n_train:,} samples)...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("  Training finished successfully.")

    # 5. Evaluate on 20% Test Set
    print(f"\n[5/5] Evaluating Model Exclusively on 20% Testing Set ({n_test:,} samples)...")
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n" + "-" * 50)
    print(f"TEST SET ACCURACY (20% Evaluation): {test_accuracy * 100:.2f}%")
    print("-" * 50)

    present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [inv_map[c] for c in present_classes]

    print("\nClassification Report (20% Test Set):")
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=target_names, digits=4))

    print("Confusion Matrix (20% Test Set):")
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)

    print("\nFeature Importances:")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<15}: {imp * 100:5.2f}%")

    # 6. Save Model Artifact
    model.save_model(model_output)
    print(f"\nTrained model artifact saved to: '{model_output}'")
    print("=" * 70)
    return model, test_accuracy


if __name__ == "__main__":
    train_precision_model()
