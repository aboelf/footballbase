#!/usr/bin/env python3
"""
XGBoost分类器训练脚本 - 比赛结果预测模型
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from xgboost_data import load_training_data
from xgboost_features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgboost_model.json")

reverse_map = {0: "H", 1: "D", 2: "A"}
feature_names = (
    [f"sax_home_{i}" for i in range(12)]
    + [f"sax_draw_{i}" for i in range(12)]
    + [f"sax_away_{i}" for i in range(12)]
    + [
        "init_win",
        "init_draw",
        "init_lose",
        "final_win",
        "final_draw",
        "final_lose",
        "init_win_change",
        "init_draw_change",
        "init_lose_change",
        "final_win_change",
        "final_draw_change",
        "final_lose_change",
        "init_win_ratio",
        "init_draw_ratio",
        "init_lose_ratio",
        "final_win_ratio",
        "final_draw_ratio",
        "final_lose_ratio",
        "home_away_diff",
        "init_implied_win",
        "init_implied_draw",
        "init_implied_lose",
        "final_implied_win",
        "final_implied_draw",
        "final_implied_lose",
    ]
)


def train_model():
    """训练XGBoost模型"""
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} matches")

    print("Extracting features...")
    X, y = extract_features(df)
    print(f"Features shape: {X.shape}")
    print(
        f"Labels distribution: H={np.sum(y == 0)}, D={np.sum(y == 1)}, A={np.sum(y == 2)}"
    )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"  Validation: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iteration = model.best_iteration
    print(f"Best iteration: {best_iteration}")

    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

    test_preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    print("\n=== Feature Importance (Top 10) ===")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {rank}. {fname}: {importance[idx]:.4f}")

    print(f"\nSaving model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print("Model saved successfully!")

    return model, test_accuracy


def predict(model, X):
    """使用训练好的模型进行预测"""
    preds = model.predict(X)
    return [reverse_map[p] for p in preds]


if __name__ == "__main__":
    model, test_accuracy = train_model()

    if test_accuracy > 0.55:
        print(f"\n✓ Test accuracy {test_accuracy * 100:.2f}% > 55%")
    else:
        print(f"\n✗ Test accuracy {test_accuracy * 100:.2f}% <= 55%")
