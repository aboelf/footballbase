#!/usr/bin/env python3
"""
批量下载总亚盘数据脚本

功能：
1. 从 Supabase league_matches 表获取所有 match_id
2. 复用 Playwright 浏览器批量下载总亚盘数据
3. 保存到 rawdata/odds/total_handicap 目录

用法：
    python download_total_handicap.py [--limit N] [--match-id ID] [--dry-run]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from downloader import DataDownloader


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


def fetch_all_match_ids(client) -> list[int]:
    """从 Supabase 获取所有比赛 ID"""
    print("从 Supabase 获取 league_matches 数据...")

    try:
        # 获取所有 match_id
        result = client.table("league_matches").select("match_id").execute()

        if not result.data:
            print("  警告: Supabase 中没有找到比赛数据")
            return []

        match_ids = [item["match_id"] for item in result.data]
        print(f"  已加载 {len(match_ids)} 个比赛 ID")
        return match_ids

    except Exception as e:
        print(f"  获取数据失败: {e}")
        return []


def download_total_handicap_batch(
    downloader: DataDownloader,
    match_ids: list[int],
    output_dir: str = "./rawdata/odds/total_handicap",
    limit: int | None = None,
    dry_run: bool = False,
    delay: float = 0.5,
):
    """
    批量下载总亚盘数据（复用浏览器）

    Args:
        downloader: DataDownloader 实例
        match_ids: 要下载的比赛 ID 列表
        output_dir: 输出目录
        limit: 最多下载数量 (None 表示全部)
        dry_run: 试运行模式，只打印不下载
        delay: 下载间隔（秒）
    """
    if limit:
        match_ids = match_ids[:limit]

    total = len(match_ids)
    print(f"\n准备下载 {total} 场比赛的总亚盘数据...")
    print(f"  输出目录: {output_dir}")
    print(f"  试运行: {'是' if dry_run else '否'}")
    print(f"  下载间隔: {delay} 秒")
    print()

    # 确保输出目录存在
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # 更新 downloader 的 base_path
    original_base_path = downloader.base_path
    downloader.base_path = output_dir

    success_count = 0
    fail_count = 0
    skipped_count = 0

    # 复用浏览器
    browser = None
    page = None

    try:
        from playwright.sync_api import sync_playwright

        if not dry_run:
            print("🚀 启动 Playwright 浏览器...")
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            print("✅ 浏览器已启动，将复用浏览器批量下载\n")

        for i, match_id in enumerate(match_ids, 1):
            print(f"[{i}/{total}] 比赛 ID: {match_id}", end=" ")

            if dry_run:
                print("⏭ 跳过 (试运行模式)")
                continue

            try:
                # 检查是否已下载过
                filename = f"{output_dir}/{match_id}_total.html"
                if os.path.exists(filename):
                    print(f"⏭ 已存在，跳过")
                    skipped_count += 1
                    continue

                # 使用复用浏览器下载
                url = f"https://vip.titan007.com/AsianOdds_n.aspx?id={match_id}&l=0"

                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(1000)  # 等待动态内容
                html_content = page.content()

                if html_content:
                    # 使用 GBK 编码保存（titan007.com 使用中文编码）
                    with open(filename, "w", encoding="gbk", errors="replace") as f:
                        f.write(html_content)

                    print("✓ 成功")
                    success_count += 1
                else:
                    print("✗ 失败: 无内容")
                    fail_count += 1

            except Exception as e:
                print(f"✗ 错误: {e}")
                fail_count += 1

            # 等待间隔
            if i < total:
                time.sleep(delay)

    except ImportError:
        print("⚠ Playwright 未安装，无法使用浏览器下载总亚盘数据")
        print("  请安装 Playwright: pip install playwright && playwright install")
    finally:
        # 关闭浏览器
        if browser:
            browser.close()
            print("\n🔒 浏览器已关闭")

    # 恢复原始设置
    downloader.base_path = original_base_path

    # 打印统计
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f"  总数: {total}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  输出目录: {output_dir}")

    return {
        "total": total,
        "success": success_count,
        "failed": fail_count,
        "skipped": skipped_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="批量下载总亚盘数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 下载所有比赛的总亚盘数据
    python download_total_handicap.py

    # 只下载前 10 个
    python download_total_handicap.py --limit 10

    # 只下载指定比赛
    python download_total_handicap.py --match-id 2799893

    # 试运行（不实际下载）
    python download_total_handicap.py --dry-run
        """,
    )

    parser.add_argument(
        "--limit", type=int, default=None, help="最多下载数量 (默认: 全部)"
    )
    parser.add_argument(
        "--match-id",
        type=int,
        default=None,
        help="只下载指定比赛 ID (与 --limit 互斥)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rawdata/odds/total_handicap",
        help="输出目录 (默认: ./rawdata/odds/total_handicap)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3, help="下载间隔秒数 (默认: 0.3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，只打印不下载",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("批量下载总亚盘数据")
    print("=" * 60)

    # 创建下载器
    downloader = DataDownloader(base_path=args.output_dir)

    # 获取比赛 ID
    if args.match_id:
        match_ids = [args.match_id]
        print(f"使用指定的比赛 ID: {args.match_id}")
    else:
        try:
            client = get_supabase_client()
            match_ids = fetch_all_match_ids(client)
        except ValueError as e:
            print(f"\n配置错误: {e}")
            print("请确保 .env 文件中已设置 SUPABASE_URL 和 SUPABASE_KEY")
            sys.exit(1)
        except Exception as e:
            print(f"\n连接 Supabase 失败: {e}")
            sys.exit(1)

    if not match_ids:
        print("没有找到可下载的比赛 ID")
        sys.exit(0)

    # 下载总亚盘数据（复用浏览器）
    result = download_total_handicap_batch(
        downloader=downloader,
        match_ids=match_ids,
        output_dir=args.output_dir,
        limit=args.limit,
        dry_run=args.dry_run,
        delay=args.delay,
    )

    # 保存下载记录
    if not args.dry_run:
        log_file = os.path.join(args.output_dir, "download_log.json")
        result["download_time"] = datetime.now().isoformat()
        result["output_dir"] = args.output_dir

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n下载记录已保存: {log_file}")


if __name__ == "__main__":
    main()
