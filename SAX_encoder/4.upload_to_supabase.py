#!/usr/bin/env python3
"""
主要改动：
1. 支持多庄家 - 自动发现 sax_results/ 目录下的所有庄家文件
2. 两种编码类型 - 支持联合编码 (joint) 和分别编码 (individual)
3. 命令行参数：
   --bookmaker    # 指定庄家 (Bet 365, Easybets)
   --type         # joint/individual/both
   --results-dir  # 结果目录
   --show-schema  # 显示 SQL
   
4. SQL 表结构 - 包含 bookmaker 字段：
   - sax_encoding - 联合编码表 (sax_interleaved, sax_delta)
   - sax_encoding_individual - 分别编码表 (sax_home, sax_draw, sax_away)
使用步骤：
# 1. 创建表
python upload_to_supabase.py --show-schema
# 复制输出到 Supabase SQL Editor 执行
# 2. 上传所有庄家的联合编码
python upload_to_supabase.py --type joint
# 3. 只上传 Easybets 的分别编码
python upload_to_supabase.py --bookmaker Easybets --type individual
文件发现测试结果：
联合编码:
  - Bet 365: sax_results/Bet 365_sax_joint.json
  - Easybets: sax_results/Easybets_sax_joint.json
分别编码:
  - Bet 365: sax_results/Bet 365_sax_individual.json
  - Easybets: sax_results/Easybets_sax_individual.json
注意：上传需要先在 Supabase 执行 python upload_to_supabase.py --show-schema 输出的 SQL 创建表。
"""

import json
import os
import sys
import argparse
from typing import Optional
from postgrest import SyncPostgrestClient
from dotenv import load_dotenv

load_dotenv()


def get_supabase_client():
    """获取 Supabase REST 客户端"""
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


def discover_available_files(results_dir: str = "sax_results") -> dict:
    """发现可用的 SAX 结果文件"""
    if not os.path.exists(results_dir):
        return {}

    available = {"individual": {}, "joint": {}}

    for filename in os.listdir(results_dir):
        if filename.endswith("_sax_individual.json"):
            bookmaker = filename.replace("_sax_individual.json", "")
            available["individual"][bookmaker] = os.path.join(results_dir, filename)
        elif filename.endswith("_sax_joint.json"):
            bookmaker = filename.replace("_sax_joint.json", "")
            available["joint"][bookmaker] = os.path.join(results_dir, filename)

    return available


def upload_joint_data(json_file: str, bookmaker: str, table_name: str = "sax_encoding"):
    """上传联合编码数据到 Supabase"""

    print(f"加载联合编码数据: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  庄家: {bookmaker}")
    print(f"  总记录数: {len(data)}")

    # 准备数据
    records = []
    for item in data:
        # 跳过没有有效编码的记录
        if not item.get("sax_interleaved") and not item.get("sax_delta"):
            continue

        record = {
            "bookmaker": bookmaker,
            "match_id": item.get("scheduleId"),
            "hometeam": item.get("hometeam"),
            "guestteam": item.get("guestteam"),
            "season": item.get("season"),
            "sax_interleaved": item.get("sax_interleaved"),
            "sax_delta": item.get("sax_delta"),
            "home_mean": item.get("stats", {}).get("home_mean"),
            "draw_mean": item.get("stats", {}).get("draw_mean"),
            "away_mean": item.get("stats", {}).get("away_mean"),
            "running_odds_count": item.get("stats", {}).get("running_odds_count"),
        }
        records.append(record)

    print(f"  有效记录数: {len(records)}")

    if not records:
        print("  无有效数据可上传")
        return 0

    # 获取客户端
    client = get_supabase_client()
    table = client.table(table_name)

    # 批量上传
    batch_size = 100
    total = 0
    errors = 0

    print(f"\n开始上传 (每批 {batch_size})...")

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            result = table.upsert(batch, on_conflict="bookmaker,match_id").execute()
            # Supabase postgrest returns data or None
            if result is not None:
                total += len(batch)
            print(f"  已上传: {total}/{len(records)}")
        except Exception as e:
            print(f"  批次 {i // batch_size + 1} 失败: {e}")
            errors += batch_size

    print(f"\n上传完成!")
    print(f"  成功: {total - errors}")
    print(f"  失败: {errors}")

    return total - errors


def upload_individual_data(
    json_file: str, bookmaker: str, table_name: str = "sax_encoding_individual"
):
    """上传分别编码数据到 Supabase"""

    print(f"加载分别编码数据: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  庄家: {bookmaker}")
    print(f"  总记录数: {len(data)}")

    # 准备数据
    records = []
    for item in data:
        if (
            not item.get("sax_home")
            and not item.get("sax_draw")
            and not item.get("sax_away")
        ):
            continue

        record = {
            "bookmaker": bookmaker,
            "match_id": item.get("scheduleId"),
            "hometeam": item.get("hometeam"),
            "guestteam": item.get("guestteam"),
            "season": item.get("season"),
            "sax_home": item.get("sax_home"),
            "sax_draw": item.get("sax_draw"),
            "sax_away": item.get("sax_away"),
            "home_mean": item.get("stats", {}).get("home_mean"),
            "draw_mean": item.get("stats", {}).get("draw_mean"),
            "away_mean": item.get("stats", {}).get("away_mean"),
            "home_std": item.get("stats", {}).get("home_std"),
            "draw_std": item.get("stats", {}).get("draw_std"),
            "away_std": item.get("stats", {}).get("away_std"),
            "running_odds_count": item.get("stats", {}).get("running_odds_count"),
        }
        records.append(record)

    print(f"  有效记录数: {len(records)}")

    if not records:
        print("  无有效数据可上传")
        return 0

    # 获取客户端
    client = get_supabase_client()
    table = client.table(table_name)

    # 批量上传
    batch_size = 100
    total = 0
    errors = 0

    print(f"\n开始上传 (每批 {batch_size})...")

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            result = table.upsert(batch, on_conflict="bookmaker,match_id").execute()
            # Supabase postgrest returns data or None
            if result is not None:
                total += len(batch)
            print(f"  已上传: {total}/{len(records)}")
        except Exception as e:
            print(f"  批次 {i // batch_size + 1} 失败: {e}")
            errors += batch_size

    print(f"\n上传完成!")
    print(f"  成功: {total - errors}")
    print(f"  失败: {errors}")

    return total - errors


def upload_all(
    bookmaker: Optional[str] = None,
    encoding_type: str = "joint",
    results_dir: str = "sax_results",
):
    """上传所有或指定庄家的数据"""

    available = discover_available_files(results_dir)

    if not available["joint"] and not available["individual"]:
        print(f"错误: 在 {results_dir} 目录下找不到 SAX 结果文件")
        print("请先运行 3.run_sax.py 生成编码结果")
        return

    print("=" * 60)
    print("SAX 数据上传到 Supabase")
    print("=" * 60)

    print("\n发现的文件:")
    if available["joint"]:
        print("  联合编码:")
        for bm, path in sorted(available["joint"].items()):
            print(f"    - {bm}: {path}")
    if available["individual"]:
        print("  分别编码:")
        for bm, path in sorted(available["individual"].items()):
            print(f"    - {bm}: {path}")

    # 确定要上传的庄家
    if bookmaker:
        bookmakers = [bookmaker]
        if (
            bookmaker not in available["joint"]
            and bookmaker not in available["individual"]
        ):
            print(f"\n错误: 找不到庄家 '{bookmaker}' 的文件")
            print(
                f"可用的庄家: {list(available['joint'].keys()) + list(available['individual'].keys())}"
            )
            return
    else:
        bookmakers = list(
            set(available["joint"].keys()) | set(available["individual"].keys())
        )

    total_uploaded = 0

    for bm in sorted(bookmakers):
        print(f"\n{'=' * 60}")
        print(f"处理庄家: {bm}")
        print("=" * 60)

        if encoding_type in ["joint", "both"]:
            if bm in available["joint"]:
                uploaded = upload_joint_data(available["joint"][bm], bm)
                total_uploaded += uploaded

        if encoding_type in ["individual", "both"]:
            if bm in available["individual"]:
                uploaded = upload_individual_data(available["individual"][bm], bm)
                total_uploaded += uploaded

    return total_uploaded


def print_sql_schema():
    """打印 Supabase 表创建 SQL"""

    print("""
-- 联合编码表（sax_interleaved + sax_delta）
CREATE TABLE IF NOT EXISTS public.sax_encoding (
  id BIGSERIAL PRIMARY KEY,
  bookmaker VARCHAR(50) NOT NULL DEFAULT 'Bet 365',
  match_id BIGINT NOT NULL,
  hometeam VARCHAR(100),
  guestteam VARCHAR(100),
  season VARCHAR(20),
  sax_interleaved VARCHAR(100),
  sax_delta VARCHAR(100),
  home_mean NUMERIC(6, 3),
  draw_mean NUMERIC(6, 3),
  away_mean NUMERIC(6, 3),
  running_odds_count INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(bookmaker, match_id)
);

CREATE INDEX IF NOT EXISTS idx_sax_encoding_bookmaker ON sax_encoding(bookmaker);
CREATE INDEX IF NOT EXISTS idx_sax_encoding_match_id ON sax_encoding(match_id);
CREATE INDEX IF NOT EXISTS idx_sax_encoding_season ON sax_encoding(season);

-- 分别编码表（sax_home + sax_draw + sax_away）
CREATE TABLE IF NOT EXISTS public.sax_encoding_individual (
  id BIGSERIAL PRIMARY KEY,
  bookmaker VARCHAR(50) NOT NULL DEFAULT 'Bet 365',
  match_id BIGINT NOT NULL,
  hometeam VARCHAR(100),
  guestteam VARCHAR(100),
  season VARCHAR(20),
  sax_home VARCHAR(50),
  sax_draw VARCHAR(50),
  sax_away VARCHAR(50),
  home_mean NUMERIC(6, 3),
  draw_mean NUMERIC(6, 3),
  away_mean NUMERIC(6, 3),
  home_std NUMERIC(6, 3),
  draw_std NUMERIC(6, 3),
  away_std NUMERIC(6, 3),
  running_odds_count INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(bookmaker, match_id)
);

CREATE INDEX IF NOT EXISTS idx_sax_ind_bookmaker ON sax_encoding_individual(bookmaker);
CREATE INDEX IF NOT EXISTS idx_sax_ind_match_id ON sax_encoding_individual(match_id);
""")


def main():
    parser = argparse.ArgumentParser(
        description="上传 SAX 编码结果到 Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 上传所有庄家的联合编码
  python upload_to_supabase.py --type joint

  # 只上传 Bet 365 的联合编码
  python upload_to_supabase.py --bookmaker "Bet 365" --type joint

  # 只上传 Easybets 的分别编码
  python upload_to_supabase.py --bookmaker Easybets --type individual

  # 上传所有数据
  python upload_to_supabase.py --type both

  # 查看 SQL 表结构
  python upload_to_supabase.py --show-schema
        """,
    )

    parser.add_argument(
        "--bookmaker",
        type=str,
        default=None,
        help="指定庄家 (如 'Bet 365', 'Easybets')，默认上传所有",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["joint", "individual", "both"],
        default="joint",
        help="编码类型: joint(联合), individual(分别), both(全部)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="sax_results",
        help="SAX 结果目录 (默认: sax_results)",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="显示 Supabase 表创建 SQL 并退出",
    )

    args = parser.parse_args()

    if args.show_schema:
        print_sql_schema()
        return

    try:
        upload_all(
            bookmaker=args.bookmaker,
            encoding_type=args.type,
            results_dir=args.results_dir,
        )
    except ValueError as e:
        print(f"\n配置错误: {e}")
        print("\n请更新 .env 文件:")
        print("  SUPABASE_URL=https://your-project.supabase.co")
        print("  SUPABASE_KEY=your-anon-key")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
