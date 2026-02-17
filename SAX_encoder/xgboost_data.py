#!/usr/bin/env python3
"""
XGBoost训练数据加载模块
从SAX编码、赔率和比赛结果数据加载并合并训练数据
"""

import json
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from postgrest import SyncPostgrestClient

load_dotenv()

SAX_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "sax_results", "Bet 365_sax_individual.json"
)


def get_supabase_client():
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        raise ValueError("请在 .env 文件中设置 SUPABASE_URL 和 SUPABASE_KEY")

    return SyncPostgrestClient(
        f"{url}/rest/v1",
        headers={
            "Authorization": f"Bearer {key}",
            "apikey": key,
            "Content-Type": "application/json",
        },
    )


def load_sax_data() -> pd.DataFrame:
    """加载SAX编码数据"""
    with open(SAX_JSON_PATH, "r", encoding="utf-8") as f:
        sax_list = json.load(f)

    df = pd.DataFrame(sax_list)
    df = df.rename(columns={"scheduleId": "schedule_id"})
    df = df[["schedule_id", "sax_home", "sax_draw", "sax_away"]]
    return df


def load_betting_odds(client, bookmaker: str = "Bet 365") -> pd.DataFrame:
    """从Supabase加载投注赔率数据"""
    response = (
        client.from_("betting_odds")
        .select("*")
        .eq("bookmaker_name", bookmaker)
        .execute()
    )
    df = pd.DataFrame(response.data)
    if df.empty:
        return df
    df = df[
        [
            "schedule_id",
            "init_win",
            "init_draw",
            "init_lose",
            "final_win",
            "final_draw",
            "final_lose",
        ]
    ]
    return df


def load_league_matches(client) -> pd.DataFrame:
    """从Supabase加载联赛比赛数据"""
    response = client.from_("league_matches").select("match_id, final_score").execute()
    df = pd.DataFrame(response.data)
    return df


def parse_result_label(final_score: str) -> Optional[str]:
    """从final_score解析比赛结果标签
    格式: "2-1" -> H(主胜), D(平局), A(客胜)
    """
    if not final_score or pd.isna(final_score):
        return None

    try:
        parts = final_score.split("-")
        if len(parts) != 2:
            return None
        home_goals = int(parts[0])
        away_goals = int(parts[1])

        if home_goals > away_goals:
            return "H"
        elif home_goals == away_goals:
            return "D"
        else:
            return "A"
    except (ValueError, IndexError):
        return None


def load_training_data() -> pd.DataFrame:
    client = get_supabase_client()

    sax_df = load_sax_data()
    odds_df = load_betting_odds(client)
    matches_df = load_league_matches(client)

    merged = sax_df.merge(odds_df, on="schedule_id", how="inner")

    merged = merged.merge(
        matches_df, left_on="schedule_id", right_on="match_id", how="inner"
    )

    merged["result_label"] = merged["final_score"].apply(parse_result_label)

    merged = merged.dropna(subset=["result_label"])

    required_cols = [
        "schedule_id",
        "sax_home",
        "sax_draw",
        "sax_away",
        "init_win",
        "init_draw",
        "init_lose",
        "final_win",
        "final_draw",
        "final_lose",
        "result_label",
    ]
    merged = merged[required_cols]

    return merged


if __name__ == "__main__":
    df = load_training_data()
    print(f"Loaded {len(df)} matches")
    print(df.head())
