#!/usr/bin/env python3
"""
上传 SAX 编码结果到 Supabase

用法:
1. 更新 .env 文件中的 SUPABASE_URL 和 SUPABASE_KEY
2. 在 Supabase SQL Editor 中执行表创建 SQL
3. 运行: python upload_to_supabase.py
"""

import json
import os
import sys
from postgrest import SyncPostgrestClient

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


def get_supabase_client():
    """获取 Supabase REST 客户端"""
    url = os.getenv('SUPABASE_URL', '').rstrip('/')
    key = os.getenv('SUPABASE_KEY', '')

    if not url or not key:
        raise ValueError("请在 .env 文件中设置 SUPABASE_URL 和 SUPABASE_KEY")

    return SyncPostgrestClient(f"{url}/rest/v1", headers={
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": "application/json"
    })


def upload_data(json_file: str):
    """上传数据到 Supabase"""

    print(f"加载数据: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  总记录数: {len(data)}")

    # 准备数据
    records = []
    for item in data:
        record = {
            'match_id': item.get('scheduleId'),
            'sax_interleaved': item.get('sax_interleaved'),
            'sax_delta': item.get('sax_delta'),
            'home_mean': item.get('stats', {}).get('home_mean'),
            'draw_mean': item.get('stats', {}).get('draw_mean'),
            'away_mean': item.get('stats', {}).get('away_mean'),
        }
        records.append(record)

    # 获取客户端
    client = get_supabase_client()
    table = client.table('bet365sax')

    # 批量上传
    batch_size = 100
    total = 0
    errors = 0

    print(f"\n开始上传 (每批 {batch_size})...")

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            # 使用 upsert 插入或更新
            result = table.upsert(batch).execute()
            if hasattr(result, 'data') and result.data:
                total += len(batch)
            elif result is None or (hasattr(result, 'errors') and not result.errors):
                total += len(batch)
            print(f"  已上传: {total}/{len(records)}")
        except Exception as e:
            print(f"  批次 {i//batch_size + 1} 失败: {e}")
            errors += batch_size

    print(f"\n上传完成!")
    print(f"  成功: {total - errors}")
    print(f"  失败: {errors}")


def main():
    print("=" * 60)
    print("SAX 数据上传到 Supabase")
    print("=" * 60)

    # 查找 JSON 文件
    json_file = 'bet365_sax_results/bet365_sax_joint.json'
    if not os.path.exists(json_file):
        json_file = '../SAX encoder/bet365_sax_results/bet365_sax_joint.json'
    if not os.path.exists(json_file):
        print(f"错误: 找不到 {json_file}")
        sys.exit(1)

    try:
        upload_data(json_file)
    except ValueError as e:
        print(f"\n配置错误: {e}")
        print("\n请更新 .env 文件:")
        print("  SUPABASE_URL=https://your-project.supabase.co")
        print("  SUPABASE_KEY=your-anon-key")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保已创建 bet365sax 表")
        print("\n在 Supabase SQL Editor 中执行:")
        print("""
CREATE TABLE IF NOT EXISTS public.bet365sax (
  id BIGSERIAL PRIMARY KEY,
  match_id BIGINT NOT NULL,
  sax_interleaved VARCHAR(50),
  sax_delta VARCHAR(50),
  home_mean NUMERIC(6, 3),
  draw_mean NUMERIC(6, 3),
  away_mean NUMERIC(6, 3),
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(match_id)
);

CREATE INDEX IF NOT EXISTS idx_bet365sax_match_id ON bet365sax(match_id);
        """)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
