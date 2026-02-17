#!/usr/bin/env python3
"""
特征工程模块：将SAX模式和赔率数据转换为机器学习特征
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_sax_onehot(
    sax_series: pd.Series, word_size: int = 4, alphabet_size: int = 3
) -> np.ndarray:
    """对SAX字符串进行one-hot编码

    Args:
        sax_series: SAX字符串Series，每个元素如 "aaab"
        word_size: SAX字符串长度，默认4
        alphabet_size: 字母表大小，默认3 (a,b,c)

    Returns:
        one-hot编码后的numpy数组，形状为 (n_samples, word_size * alphabet_size)
    """
    alphabet = [chr(ord("a") + i) for i in range(alphabet_size)]

    all_chars = []
    for sax_str in sax_series:
        if pd.isna(sax_str) or len(sax_str) != word_size:
            chars = [alphabet[0]] * word_size
        else:
            chars = list(sax_str.lower())
            for i, c in enumerate(chars):
                if c not in alphabet:
                    chars[i] = alphabet[0]
        all_chars.append(chars)

    encoder = OneHotEncoder(
        categories=[alphabet] * word_size, sparse_output=False, handle_unknown="ignore"
    )
    encoded = encoder.fit_transform(all_chars)

    return encoded


def extract_odds_features(df: pd.DataFrame) -> np.ndarray:
    """提取赔率统计特征

    特征包括:
    - 初始赔率: init_win, init_draw, init_lose
    - 最终赔率: final_win, final_draw, final_lose
    - 赔率变化: init - final
    - 赔率比率: init / final
    - 主客赔率差: init_win - init_lose
    - 赔率隐含概率: 1/odds

    Returns:
        赔率特征矩阵，形状为 (n_samples, 19)
    """
    odds_cols = [
        "init_win",
        "init_draw",
        "init_lose",
        "final_win",
        "final_draw",
        "final_lose",
    ]
    df_odds = df[odds_cols].fillna(df[odds_cols].median())

    features = []

    init_odds = df_odds[["init_win", "init_draw", "init_lose"]].values
    final_odds = df_odds[["final_win", "final_draw", "final_lose"]].values

    features.append(init_odds)
    features.append(final_odds)

    odds_change = init_odds - final_odds
    features.append(odds_change)

    with np.errstate(divide="ignore", invalid="ignore"):
        odds_ratio = np.where(final_odds != 0, init_odds / final_odds, 1.0)
        odds_ratio = np.nan_to_num(odds_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(odds_ratio)

    home_away_diff = (df_odds["init_win"] - df_odds["init_lose"]).values.reshape(-1, 1)
    features.append(home_away_diff)

    init_implied = np.where(init_odds != 0, 1.0 / init_odds, 0.0)
    final_implied = np.where(final_odds != 0, 1.0 / final_odds, 0.0)
    features.append(init_implied)
    features.append(final_implied)

    return np.hstack(features)


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """从DataFrame中提取特征和标签

    Args:
        df: 包含SAX模式和赔率数据的DataFrame

    Returns:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 标签向量，形状为 (n_samples,)
    """
    sax_home_encoded = encode_sax_onehot(df["sax_home"], word_size=4, alphabet_size=3)
    sax_draw_encoded = encode_sax_onehot(df["sax_draw"], word_size=4, alphabet_size=3)
    sax_away_encoded = encode_sax_onehot(df["sax_away"], word_size=4, alphabet_size=3)

    odds_features = extract_odds_features(df)

    X = np.hstack([sax_home_encoded, sax_draw_encoded, sax_away_encoded, odds_features])

    label_map = {"H": 0, "D": 1, "A": 2}
    y = df["result_label"].map(label_map).values

    return X, y


if __name__ == "__main__":
    from xgboost_data import load_training_data

    df = load_training_data()
    print(f"Loaded {len(df)} matches")

    X, y = extract_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature dimensions breakdown:")
    print(f"  - SAX home (4 pos * 3 alphabet): 12")
    print(f"  - SAX draw (4 pos * 3 alphabet): 12")
    print(f"  - SAX away (4 pos * 3 alphabet): 12")
    print(f"  - Odds features: 19")
    print(f"  - Total: {X.shape[1]}")
