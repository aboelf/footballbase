#!/usr/bin/env python3
"""
解析总亚盘数据并上传到 Supabase

功能：
1. 解析 rawdata/odds/total_handicap 目录下的 HTML 文件
2. 提取指定庄家 (bet365, easybets, macau) 的初始赔率
3. 解析 oddsDetail 表格数据
4. 上传到 Supabase total_handicap 表

用法：
    python parse_total_handicap.py [--limit N] [--match-id ID] [--dry-run]
"""

import os
import sys
import json
import argparse
import re
import multiprocessing
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

from bs4 import BeautifulSoup
from dotenv import load_dotenv

# 尝试使用更快的 lxml 解析器
HTML_PARSER = "html.parser"
try:
    import lxml

    HTML_PARSER = "lxml"
except ImportError:
    pass

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 加载环境变量
load_dotenv()


def get_supabase_client():
    """获取 Supabase REST 客户端"""
    from postgrest import SyncPostgrestClient

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


@dataclass
class TotalHandicapRecord:
    """总亚盘数据记录"""

    match_id: int
    bookmaker: str
    init_odds_home: Optional[float] = None
    handicap: Optional[str] = None
    init_odds_away: Optional[float] = None
    final_handicap: Optional[str] = None
    final_odds_home: Optional[float] = None
    final_odds_away: Optional[float] = None
    odds_detail: Optional[str] = None


# 庄家映射
BOOKMAKER_PATTERNS = {
    "bet365": "36*",
    "easybets": "易胜*",  # oddsDetail table
    "macau": "澳*",
}

# 主赔率表中的庄家映射（与oddsDetail不同）
BOOKMAKER_PATTERNS_MAIN = {
    "bet365": "36*",
    "easybets": "易*",  # main odds table
    "macau": "澳*",
}


def _parse_worker(args: tuple) -> list[dict]:
    """
    多进程工作函数 - 供 multiprocessing 使用
    """
    filepath, verbose = args
    try:
        records = parse_html_file(filepath)
        if verbose and records:
            return records
        elif verbose:
            return []
        return records
    except Exception as e:
        if verbose:
            print(f"  Error processing {filepath}: {e}")
        return []


def parse_handicap_value(handicap_text: str) -> str:
    """
    解析盘口值 - 保留原始格式

    例如: "平手/半球" 保持不变
    """
    if not handicap_text:
        return handicap_text

    return handicap_text


def parse_html_file(filepath: str) -> list[dict]:
    """
    解析单个 HTML 文件

    Returns:
        list[dict]: 包含 match_id, bookmaker, init_odds_home, handicap, init_odds_away, odds_detail
    """
    # 从文件名提取 match_id
    filename = os.path.basename(filepath)
    match_id_match = re.match(r"(\d+)_total\.html", filename)
    if not match_id_match:
        print(f"  警告: 无法从文件名提取 match_id: {filename}")
        return []
    match_id = int(match_id_match.group(1))

    # 读取 HTML 文件 (GBK 编码)
    with open(filepath, "rb") as f:
        html_content = f.read().decode("gbk", errors="replace")

    soup = BeautifulSoup(html_content, HTML_PARSER)

    results = []

    # 查找赔率表格 (id="odds")
    odds_table = soup.find("table", {"id": "odds"})
    if not odds_table:
        print(f"  警告: 未找到赔率表格 (id=odds)")
        return results

    # 查找所有数据行 (tr)
    rows = odds_table.find_all("tr", bgcolor=["#FFFFFF", "#FAFAFA"])

    for row in rows:
        # 获取庄家名称（第二列）
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        bookmaker_cell = cells[1]
        bookmaker_text = bookmaker_cell.get_text(strip=True)

        # 检查是否是需要提取的庄家 (使用主表映射)
        matched_bookmaker = None
        for bm_name, bm_pattern in BOOKMAKER_PATTERNS_MAIN.items():
            if bookmaker_text.startswith(bm_pattern):
                matched_bookmaker = bm_name
                break

        if not matched_bookmaker:
            continue

        # 提取初始赔率 (前3列数据)
        # 格式: init_odds_home, handicap, init_odds_away
        if len(cells) >= 5:
            # 获取初始赔率 (使用 title 属性中的时间戳来判断)
            init_odds_home = None
            init_odds_away = None
            handicap_text = None

            # 查找有 title 属性的单元格 (这些是初始赔率)
            for i, cell in enumerate(cells[3:6]):
                title = cell.get("title", "")
                cell_text = cell.get_text(strip=True)

                if title:
                    # 这是一个有时间戳的单元格
                    if i == 1:  # 盘口列 (cell_text 是盘口名称，不是数字)
                        handicap_text = cell_text
                    else:
                        # 尝试解析数字
                        try:
                            value = float(cell_text)
                            if i == 0:
                                init_odds_home = value
                            elif i == 2:
                                init_odds_away = value
                        except ValueError:
                            pass
                elif i == 1:  # 盘口列（没有title时也取）
                    handicap_text = cell_text

            # 如果没有找到初始赔率，尝试获取 oddstype="wholeLastOdds" 的值
            if init_odds_home is None:
                last_odds_cells = row.find_all("td", {"oddstype": "wholeLastOdds"})
                if last_odds_cells and len(last_odds_cells) >= 3:
                    try:
                        init_odds_home = float(last_odds_cells[0].get_text(strip=True))
                        handicap_text = last_odds_cells[1].get_text(strip=True)
                        init_odds_away = float(last_odds_cells[2].get_text(strip=True))
                    except ValueError:
                        pass

            # 如果还是没有，尝试 oddstype="wholeOdds" (当前赔率)
            if init_odds_home is None:
                current_odds_cells = row.find_all("td", {"oddstype": "wholeOdds"})
                if current_odds_cells and len(current_odds_cells) >= 3:
                    try:
                        init_odds_home = float(
                            current_odds_cells[0].get_text(strip=True)
                        )
                        handicap_text = current_odds_cells[1].get_text(strip=True)
                        init_odds_away = float(
                            current_odds_cells[2].get_text(strip=True)
                        )
                    except ValueError:
                        pass

            # 如果还是没有，直接从 cells[3:6] 获取初始赔率（第一组数据）
            if init_odds_home is None and len(cells) >= 6:
                try:
                    # cells[3] = init_odds_home, cells[4] = handicap, cells[5] = init_odds_away
                    home_text = (
                        cells[3].get_text(strip=True)
                        if hasattr(cells[3], "get_text")
                        else cells[3]
                    )
                    handicap_text = (
                        cells[4].get_text(strip=True)
                        if hasattr(cells[4], "get_text")
                        else cells[4]
                    )
                    away_text = (
                        cells[5].get_text(strip=True)
                        if hasattr(cells[5], "get_text")
                        else cells[5]
                    )
                    init_odds_home = float(home_text) if home_text else None
                    init_odds_away = float(away_text) if away_text else None
                    if not handicap_text:
                        handicap_text = None
                except (ValueError, IndexError):
                    pass

            # 解析盘口值
            handicap = parse_handicap_value(handicap_text) if handicap_text else None

            # 提取终盘赔率 (cells[9:12] - 临场赔率)
            final_handicap = None
            final_odds_home = None
            final_odds_away = None
            if len(cells) >= 12:
                try:
                    home_text = (
                        cells[9].get_text(strip=True)
                        if hasattr(cells[9], "get_text")
                        else cells[9]
                    )
                    handicap_text = (
                        cells[10].get_text(strip=True)
                        if hasattr(cells[10], "get_text")
                        else cells[10]
                    )
                    away_text = (
                        cells[11].get_text(strip=True)
                        if hasattr(cells[11], "get_text")
                        else cells[11]
                    )
                    final_handicap = handicap_text if handicap_text else None
                    final_odds_home = float(home_text) if home_text else None
                    final_odds_away = float(away_text) if away_text else None
                except (ValueError, IndexError):
                    pass

            # 解析 oddsDetail 表格
            odds_detail = parse_odds_detail(soup, matched_bookmaker)

            record = {
                "match_id": match_id,
                "bookmaker": matched_bookmaker,
                "init_odds_home": init_odds_home,
                "handicap": handicap,
                "init_odds_away": init_odds_away,
                "final_handicap": final_handicap,
                "final_odds_home": final_odds_home,
                "final_odds_away": final_odds_away,
                "odds_detail": odds_detail,
            }

            results.append(record)

    return results


def parse_odds_detail(soup: BeautifulSoup, bookmaker: str) -> Optional[str]:
    """
    解析 oddsDetail 表格中指定庄家的数据

    Returns:
        JSON string of odds detail data
    """
    # 查找 oddsDetail 表格
    detail_table = soup.find("table", {"id": "oddsDetail"})
    if not detail_table:
        return None

    # 查找表头行
    header_row = detail_table.find("tr", {"class": "thead2"})
    if not header_row:
        return None

    # 找到庄家对应的列索引
    headers = header_row.find_all("th")
    bookmaker_col_index = None

    bm_pattern = BOOKMAKER_PATTERNS.get(bookmaker)
    if not bm_pattern:
        return None

    for i, th in enumerate(headers):
        th_text = th.get_text(strip=True)
        if th_text.startswith(bm_pattern):
            bookmaker_col_index = i
            break

    if bookmaker_col_index is None:
        return None

    # 获取表头的列索引，用于找到比分列的位置
    # 倒数第二列是比分 (�ȷ�)，最后一列是时间
    total_cols = len(headers)
    score_col_index = total_cols - 2  # 比分列在倒数第二
    time_col_index = total_cols - 1  # 时间列在最后

    # 提取该列的所有数据行
    detail_data = []
    data_rows = detail_table.find_all("tr", align="center")

    for row in data_rows:
        cells = row.find_all("td")
        if len(cells) <= bookmaker_col_index:
            continue

        # 获取比分列（倒数第二列），过滤掉已开始的比赛
        score_text = ""
        if len(cells) > score_col_index:
            score_cell = cells[score_col_index]
            score_text = score_cell.get_text(strip=True)

        # 过滤掉有比分的行（已开始的比赛）
        if score_text and "-" in score_text:
            continue

        cell = cells[bookmaker_col_index]
        cell_text = cell.get_text(strip=True)

        if not cell_text:
            continue

        # 解析赔率文本
        # 格式: "平手/半球1.080.72" 或 "平手1.100.70"
        # 需要提取: handicap, home, away
        handicap = None
        home = None
        away = None

        # 尝试匹配 handicap + home + away
        # 匹配模式: 中文盘口 + 小数赔率
        # 例如: 平手/半球1.080.72 -> handicap="平手/半球", home=1.08, away=0.72
        match = re.match(r"^(.+?)(\d+\.\d{2})(\d+\.\d{2})$", cell_text)
        if match:
            handicap = match.group(1)
            home = float(match.group(2))
            away = float(match.group(3))
        else:
            # 尝试更简单的模式：数字开头
            # 例如: 1.100.70 -> home=1.10, away=0.70
            match2 = re.match(r"^(\d+\.\d{2})(\d+\.\d{2})$", cell_text)
            if match2:
                home = float(match2.group(1))
                away = float(match2.group(2))

        # 获取时间信息 (最后一列)
        time_text = None
        if len(cells) > time_col_index:
            time_cell = cells[time_col_index]
            time_text = time_cell.get_text(strip=True)

        if home is not None and away is not None:
            detail_data.append(
                {
                    "handicap": handicap,
                    "home": home,
                    "away": away,
                    "time": time_text,
                }
            )

    if detail_data:
        return json.dumps(detail_data, ensure_ascii=False)

    return None


def upload_to_supabase(records: list[dict], batch_size: int = 100) -> int:
    """上传数据到 Supabase"""
    if not records:
        print("  无数据可上传")
        return 0

    client = get_supabase_client()
    table = client.table("total_handicap")

    total = 0
    errors = 0

    print(f"  开始上传 ({len(records)} 条记录)...")

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            result = table.upsert(batch, on_conflict="match_id,bookmaker").execute()
            if result is not None:
                total += len(batch)
            print(f"    已上传: {total}/{len(records)}")
        except Exception as e:
            print(f"    批次 {i // batch_size + 1} 失败: {e}")
            errors += len(batch)

    return total


def get_sql_schema() -> str:
    """返回 Supabase 表创建 SQL"""
    return """
-- 总亚盘赔率表
CREATE TABLE IF NOT EXISTS public.total_handicap (
  id BIGSERIAL PRIMARY KEY,
  match_id BIGINT NOT NULL,
  bookmaker VARCHAR(50) NOT NULL,
  init_odds_home NUMERIC(6, 3),
  handicap VARCHAR(50),
  init_odds_away NUMERIC(6, 3),
  final_handicap VARCHAR(50),
  final_odds_home NUMERIC(6, 3),
  final_odds_away NUMERIC(6, 3),
  odds_detail JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(match_id, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_total_handicap_match_id ON total_handicap(match_id);
CREATE INDEX IF NOT EXISTS idx_total_handicap_bookmaker ON total_handicap(bookmaker);
"""


def parse_directory(
    input_dir: str,
    limit: Optional[int] = None,
    match_id: Optional[int] = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    解析目录下所有 HTML 文件

    Args:
        input_dir: 输入目录
        limit: 最多处理文件数
        match_id: 只处理指定比赛
        dry_run: 试运行模式

    Returns:
        所有解析后的记录
    """
    # 获取所有 HTML 文件
    html_files = []
    for f in os.listdir(input_dir):
        if f.endswith("_total.html"):
            if match_id:
                # 只处理指定比赛
                file_match_id = re.match(r"(\d+)_total\.html", f)
                if file_match_id and int(file_match_id.group(1)) == match_id:
                    html_files.append(os.path.join(input_dir, f))
            else:
                html_files.append(os.path.join(input_dir, f))

    # 按 match_id 排序
    html_files.sort(key=lambda x: int(re.match(r".*/(\d+)_total\.html", x).group(1)))

    if limit:
        html_files = html_files[:limit]

    print(f"\n找到 {len(html_files)} 个 HTML 文件")
    if match_id:
        print(f"  过滤: 只处理 match_id = {match_id}")

    all_records = []

    for i, filepath in enumerate(html_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(html_files)}] 处理: {filename}", end=" ")

        records = parse_html_file(filepath)
        if records:
            print(f"✓ ({len(records)} 条记录)")
            all_records.extend(records)
        else:
            print("✗ (无数据)")

    return all_records


def parse_directory_parallel(
    input_dir: str,
    limit: Optional[int] = None,
    match_id: Optional[int] = None,
    dry_run: bool = False,
    workers: int = 8,
    quiet: bool = False,
) -> list[dict]:
    """
    并行解析目录下所有 HTML 文件

    Args:
        input_dir: 输入目录
        limit: 最多处理文件数
        match_id: 只处理指定比赛
        dry_run: 试运行模式
        workers: 并行进程数
        verbose: 是否显示详细进度

    Returns:
        所有解析后的记录
    """
    # 获取所有 HTML 文件
    html_files = []
    for f in os.listdir(input_dir):
        if f.endswith("_total.html"):
            if match_id:
                file_match_id = re.match(r"(\d+)_total\.html", f)
                if file_match_id and int(file_match_id.group(1)) == match_id:
                    html_files.append(os.path.join(input_dir, f))
            else:
                html_files.append(os.path.join(input_dir, f))

    if limit:
        html_files = html_files[:limit]

    total_files = len(html_files)
    print(f"\n找到 {total_files} 个 HTML 文件")
    print(f"  使用 {workers} 个并行进程")

    if match_id:
        print(f"  过滤: 只处理 match_id = {match_id}")

    if dry_run:
        print("[试运行模式] 不实际解析")
        return []

    # 准备任务参数
    args_list = [(fp, not quiet) for fp in html_files]

    # 使用多进程池并行处理
    all_records = []
    processed = 0

    with multiprocessing.Pool(processes=workers) as pool:
        # 使用 imap_unordered 提高效率
        for records in pool.imap_unordered(_parse_worker, args_list, chunksize=10):
            processed += 1
            if records:
                all_records.extend(records)
            if not quiet and processed % 100 == 0:
                print(f"  进度: {processed}/{total_files} ({len(all_records)} 条记录)")

    print(f"  完成: {processed}/{total_files} 文件, {len(all_records)} 条记录")

    return all_records


def main():
    parser = argparse.ArgumentParser(
        description="解析总亚盘数据并上传到 Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 解析所有数据
    python parse_total_handicap.py

    # 只解析前 10 个文件
    python parse_total_handicap.py --limit 10

    # 只解析指定比赛
    python parse_total_handicap.py --match-id 1877210

    # 试运行（不解压，不上传）
    python parse_total_handicap.py --dry-run

    # 显示 SQL 表结构
    python parse_total_handicap.py --show-schema
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="./rawdata/odds/total_handicap",
        help="HTML 文件目录 (默认: ./rawdata/odds/total_handicap)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="最多处理文件数 (默认: 全部)"
    )
    parser.add_argument(
        "--match-id",
        type=int,
        default=None,
        help="只处理指定比赛 ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，只打印不解压，不上传",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="显示 Supabase 表创建 SQL 并退出",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="批量上传大小 (默认: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行进程数 (默认: 8)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细进度",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不显示进度",
    )

    args = parser.parse_args()

    if args.show_schema:
        print(get_sql_schema())
        return

    print("=" * 60)
    print("解析总亚盘数据")
    print("=" * 60)
    print(f"  输入目录: {args.input_dir}")

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"\n错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 解析 HTML 文件 (并行处理)
    records = parse_directory_parallel(
        input_dir=args.input_dir,
        limit=args.limit,
        match_id=args.match_id,
        dry_run=args.dry_run,
        workers=args.workers,
        quiet=args.quiet,
    )

    if not records:
        print("\n未找到任何数据")
        sys.exit(0)

    # 统计
    print("\n" + "-" * 40)
    print(f"总记录数: {len(records)}")

    # 按庄家统计
    bookmaker_stats = {}
    for r in records:
        bm = r.get("bookmaker", "unknown")
        bookmaker_stats[bm] = bookmaker_stats.get(bm, 0) + 1

    print("按庄家统计:")
    for bm, count in sorted(bookmaker_stats.items()):
        print(f"  - {bm}: {count}")

    # 如果是 dry_run，不上传
    if args.dry_run:
        print("\n[试运行模式] 不上传数据")
        print("\n前 5 条记录示例:")
        for r in records[:5]:
            print(f"  {json.dumps(r, ensure_ascii=False)}")
        return

    # 上传到 Supabase
    print("\n" + "-" * 40)
    print("上传到 Supabase...")

    try:
        uploaded = upload_to_supabase(records, batch_size=args.batch_size)
        print(f"\n上传完成! 成功: {uploaded}")
    except ValueError as e:
        print(f"\n配置错误: {e}")
        print("请确保 .env 文件中已设置 SUPABASE_URL 和 SUPABASE_KEY")
        print("\n可以先运行 --show-schema 获取 SQL 表结构")
        sys.exit(1)
    except Exception as e:
        print(f"\n上传失败: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
