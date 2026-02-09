"""
足球比赛赔率分析系统 - 数据下载模块
使用 Playwright 处理动态加载的数据
"""

import requests
from datetime import datetime
import os
import re
import json
import time
from typing import Dict, List, Optional
from enum import Enum


class OddsType(Enum):
    HANDICAP = "handicap"
    ODDS = "odds"
    OVERUNDER = "overunder"


class Bookmaker(Enum):
    MACAU = "macau"
    BET365 = "bet365"
    EASYBET = "easybet"
    WILLIAM = "william"
    BETFAIR = "betfair"

    @property
    def name(self) -> str:
        return {
            Bookmaker.MACAU: "澳门",
            Bookmaker.BET365: "bet365",
            Bookmaker.EASYBET: "易胜博",
            Bookmaker.WILLIAM: "威廉",
            Bookmaker.BETFAIR: "Betfair",
        }[self]

    def get_company_id(self, odds_type: OddsType) -> int:
        """根据赔率类型返回对应的庄家ID"""
        if odds_type == OddsType.HANDICAP:
            return {
                Bookmaker.MACAU: 1,
                Bookmaker.BET365: 8,
                Bookmaker.EASYBET: 12,
            }[self]
        elif odds_type == OddsType.ODDS:
            return {
                Bookmaker.BET365: 281,
                Bookmaker.EASYBET: 90,
                Bookmaker.WILLIAM: 115,
                Bookmaker.MACAU: 80,
                Bookmaker.BETFAIR: 2,
            }[self]
        elif odds_type == OddsType.OVERUNDER:
            return {
                Bookmaker.MACAU: 1,
                Bookmaker.BET365: 8,
                Bookmaker.EASYBET: 12,
            }[self]
        raise ValueError(f"Unknown odds type: {odds_type}")


BOOKMAKER_HANDICAP_IDS = {
    OddsType.HANDICAP: [Bookmaker.MACAU, Bookmaker.BET365, Bookmaker.EASYBET],
    OddsType.ODDS: [
        Bookmaker.MACAU,
        Bookmaker.WILLIAM,
        Bookmaker.BET365,
        Bookmaker.EASYBET,
        # Bookmaker.BETFAIR,  # 欧赔暂无数据
    ],
    OddsType.OVERUNDER: [Bookmaker.MACAU, Bookmaker.BET365, Bookmaker.EASYBET],
}


class DataDownloader:
    """数据下载器类"""

    def __init__(self, base_path="./data"):
        self.base_path = base_path
        self.session = requests.Session()

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        self.session.headers.update(self.headers)

        # 不自动创建子目录，由各下载脚本自行管理

    def _ensure_dir(self, path: str) -> None:
        """确保目录存在，不存在则自动创建"""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def get_odds_company_ids(self, match_id: str) -> Dict[int, Dict]:
        """
        从oddslist页面获取欧赔庄家的id参数
        返回格式: {company_id: {"id": xxx, "name": "庄家名", "initial_odds": "x/x/x"}}
        """
        url = f"https://1x2d.titan007.com/{match_id}.js"

        try:
            response = self.session.get(url, timeout=30)
            if response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding

            if response.status_code != 200:
                return {}

            js_content = response.text
            game_match = re.search(r"var game=Array\((.*?)\);", js_content, re.DOTALL)

            if not game_match:
                return {}

            games_str = game_match.group(1)
            games = games_str.split('","')

            result = {}
            for game in games:
                game = game.strip('"')
                parts = game.split("|")

                if len(parts) >= 6:
                    company_id = int(parts[0])
                    data_id = parts[1]
                    name = parts[2]
                    initial_odds = f"{parts[3]}/{parts[4]}/{parts[5]}"

                    result[company_id] = {
                        "id": data_id,
                        "name": name,
                        "initial_odds": initial_odds,
                    }

            return result

        except Exception as e:
            print(f"获取欧赔庄家ID失败: {str(e)}")
            return {}

    def _get_odds_url(
        self,
        match_id: str,
        odds_type: OddsType,
        company_id: int,
        data_id: Optional[str] = None,
    ) -> str:
        if odds_type == OddsType.HANDICAP:
            return f"https://vip.titan007.com/changeDetail/handicap.aspx?id={match_id}&companyID={company_id}&l=0"
        elif odds_type == OddsType.ODDS:
            if data_id:
                return f"https://1x2.titan007.com/OddsHistory.aspx?id={data_id}&sid={match_id}&cid={company_id}"
            else:
                return f"https://1x2.titan007.com/OddsHistory.aspx?sid={match_id}&cid={company_id}"
        elif odds_type == OddsType.OVERUNDER:
            return f"https://vip.titan007.com/changeDetail/overunder.aspx?id={match_id}&companyID={company_id}&l=0"
        raise ValueError(f"Unknown odds type: {odds_type}")

    def _get_odds_subdir(self, odds_type: OddsType) -> str:
        return {
            OddsType.HANDICAP: "handicap",
            OddsType.ODDS: "odds",
            OddsType.OVERUNDER: "overunder",
        }[odds_type]

    def download_analysis_data(self, match_id, use_browser=True):
        """
        下载比赛分析数据
        URL: https://zq.titan007.com/analysis/[比赛编号]cn.htm
        use_browser: 是否使用 Playwright 浏览器获取动态内容
        """
        url = f"https://zq.titan007.com/analysis/{match_id}cn.htm"

        try:
            print(f"正在下载比赛分析数据: {url}")

            if use_browser:
                # 使用 Playwright 获取动态内容
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()

                    # 访问页面
                    page.goto(url, wait_until="networkidle")

                    # 等待动态内容加载完成
                    # 尝试等待包含比分的元素出现
                    try:
                        # 等待最多 10 秒让动态内容加载
                        page.wait_for_timeout(2000)

                        # 检查是否有特定的选择器需要等待
                        # 常见的动态内容容器
                        possible_selectors = [
                            'div[id*="score"]',
                            'span[id*="score"]',
                            ".score-box",
                            "#比分",
                            '[class*="score"]',
                        ]

                        for selector in possible_selectors:
                            try:
                                page.wait_for_selector(selector, timeout=3000)
                                break
                            except:
                                continue

                    except Exception as e:
                        print(f"  等待元素超时: {e}")

                    # 获取完整页面内容（包括动态加载的）
                    html_content = page.content()

                    browser.close()

            else:
                # 使用 requests 获取静态内容
                response = self.session.get(url, timeout=30)
                response.encoding = "utf-8"
                html_content = response.text

            if html_content:
                # 保存原始HTML
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.base_path}/analysis/{match_id}_{timestamp}.html"
                self._ensure_dir(filename)

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(html_content)

                # 提取关键信息
                data = {
                    "match_id": match_id,
                    "url": url,
                    "download_time": timestamp,
                    "raw_file": filename,
                    "status": "success",
                    "content_length": len(html_content),
                    "use_browser": use_browser,
                }

                # 保存JSON元数据
                meta_file = f"{self.base_path}/analysis/{match_id}_{timestamp}.json"
                self._ensure_dir(meta_file)

                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"✓ 分析数据下载成功: {filename}")
                return data
            else:
                print(f"✗ 下载失败，未获取到内容")
                return {"status": "failed", "error": "No content retrieved"}

        except ImportError:
            print("⚠ Playwright 未安装，使用 requests 降级获取")
            return self._download_analysis_requests(match_id)
        except Exception as e:
            print(f"✗ 下载出错: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def _download_analysis_requests(self, match_id):
        """使用 requests 降级获取分析数据"""
        url = f"https://zq.titan007.com/analysis/{match_id}cn.htm"

        try:
            response = self.session.get(url, timeout=30)
            if response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding

            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.base_path}/analysis/{match_id}_{timestamp}.html"
                self._ensure_dir(filename)

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)

                data = {
                    "match_id": match_id,
                    "url": url,
                    "download_time": timestamp,
                    "raw_file": filename,
                    "status": "success",
                    "content_length": len(response.text),
                    "use_browser": False,
                    "warning": "使用静态请求获取，动态内容可能不完整",
                }

                meta_file = f"{self.base_path}/analysis/{match_id}_{timestamp}.json"
                self._ensure_dir(meta_file)

                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"✓ 分析数据下载成功 (静态): {filename}")
                return data
            else:
                return {"status": "failed", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def download_odds_data(
        self,
        match_id: str,
        odds_type: OddsType = OddsType.HANDICAP,
        bookmaker: Optional[Bookmaker] = None,
        use_browser: bool = True,
        data_id: Optional[str] = None,
    ) -> Dict:
        """
        下载赔率数据
        :param match_id: 比赛编号
        :param odds_type: 赔率类型 (handicap/odds/overunder)
        :param bookmaker: 庄家 (默认下载所有庄家)
        :param use_browser: 是否使用 Playwright
        :param data_id: 欧赔专用的动态ID (可选，如未提供则自动获取)
        """
        results = []
        bookmakers = (
            [bookmaker] if bookmaker else BOOKMAKER_HANDICAP_IDS.get(odds_type, [])
        )

        if not bookmakers:
            return {
                "status": "failed",
                "error": f"No bookmakers configured for {odds_type.value}",
            }

        for bm in bookmakers:
            company_id = bm.get_company_id(odds_type)
            actual_data_id = data_id

            if odds_type == OddsType.ODDS and not actual_data_id:
                company_ids = self.get_odds_company_ids(match_id)
                if company_id in company_ids:
                    actual_data_id = company_ids[company_id]["id"]

            url = self._get_odds_url(match_id, odds_type, company_id, actual_data_id)

            try:
                print(f"正在下载 {odds_type.value} 赔率 [{bm.name}]: {url}")

                if use_browser:
                    from playwright.sync_api import sync_playwright

                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()
                        page.goto(url, wait_until="networkidle")
                        page.wait_for_timeout(3000)
                        html_content = page.content()
                        browser.close()
                else:
                    response = self.session.get(url, timeout=30)
                    response.encoding = (
                        response.apparent_encoding
                        if response.apparent_encoding
                        else "gbk"
                    )
                    html_content = response.text

                if html_content:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    subdir = self._get_odds_subdir(odds_type)
                    filename = f"{self.base_path}/odds/{subdir}/{match_id}_{bm.value}_{timestamp}.html"
                    self._ensure_dir(filename)

                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    result = {
                        "match_id": match_id,
                        "odds_type": odds_type.value,
                        "bookmaker": bm.value,
                        "bookmaker_name": bm.name,
                        "company_id": company_id,
                        "data_id": actual_data_id,
                        "url": url,
                        "download_time": timestamp,
                        "raw_file": filename,
                        "status": "success",
                        "content_length": len(html_content),
                        "use_browser": use_browser,
                    }

                    meta_file = f"{self.base_path}/odds/{subdir}/{match_id}_{bm.value}_{timestamp}.json"
                    self._ensure_dir(meta_file)

                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    print(f"✓ {odds_type.value} [{bm.name}] 下载成功: {filename}")
                    results.append(result)
                else:
                    results.append(
                        {
                            "match_id": match_id,
                            "odds_type": odds_type.value,
                            "bookmaker": bm.value,
                            "bookmaker_name": bm.name,
                            "status": "failed",
                            "error": "No content retrieved",
                        }
                    )

            except ImportError:
                print("⚠ Playwright 未安装，使用 requests 降级获取")
                result = self._download_odds_requests(match_id, odds_type, bm)
                results.append(result)
            except Exception as e:
                print(f"✗ 下载 {bm.name} 出错: {str(e)}")
                results.append(
                    {
                        "match_id": match_id,
                        "odds_type": odds_type.value,
                        "bookmaker": bm.value,
                        "bookmaker_name": bm.name,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return {
            "status": "success"
            if all(r.get("status") == "success" for r in results)
            else "partial",
            "results": results,
            "total": len(results),
            "success_count": sum(1 for r in results if r.get("status") == "success"),
        }

    def _download_odds_requests(
        self,
        match_id: str,
        odds_type: OddsType,
        bookmaker: Bookmaker,
        data_id: Optional[str] = None,
    ) -> Dict:
        """使用 requests 降级获取赔率数据"""
        company_id = bookmaker.get_company_id(odds_type)
        url = self._get_odds_url(match_id, odds_type, company_id, data_id)

        try:
            response = self.session.get(url, timeout=30)
            if response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding

            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                subdir = self._get_odds_subdir(odds_type)
                filename = f"{self.base_path}/odds/{subdir}/{match_id}_{bookmaker.value}_{timestamp}.html"
                self._ensure_dir(filename)

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)

                result = {
                    "match_id": match_id,
                    "odds_type": odds_type.value,
                    "bookmaker": bookmaker.value,
                    "bookmaker_name": bookmaker.name,
                    "company_id": company_id,
                    "data_id": data_id,
                    "url": url,
                    "download_time": timestamp,
                    "raw_file": filename,
                    "status": "success",
                    "content_length": len(response.text),
                    "use_browser": False,
                }

                meta_file = f"{self.base_path}/odds/{subdir}/{match_id}_{bookmaker.value}_{timestamp}.json"
                self._ensure_dir(meta_file)

                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                print(
                    f"✓ {odds_type.value} [{bookmaker.name}] 下载成功 (静态): {filename}"
                )
                return result
            else:
                return {
                    "match_id": match_id,
                    "odds_type": odds_type.value,
                    "bookmaker": bookmaker.value,
                    "bookmaker_name": bookmaker.name,
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {
                "match_id": match_id,
                "odds_type": odds_type.value,
                "bookmaker": bookmaker.value,
                "bookmaker_name": bookmaker.name,
                "status": "failed",
                "error": str(e),
            }

    def download_all_handicap(self, match_id: str, use_browser: bool = True) -> Dict:
        """下载亚盘所有庄家赔率"""
        results = {}
        for bookmaker in BOOKMAKER_HANDICAP_IDS[OddsType.HANDICAP]:
            result = self.download_odds_data(
                match_id,
                odds_type=OddsType.HANDICAP,
                bookmaker=bookmaker,
                use_browser=use_browser,
            )
            results[bookmaker.value] = result
            time.sleep(0.25)
        return results

    def download_all_odds(self, match_id: str, use_browser: bool = True) -> Dict:
        """下载欧赔所有庄家赔率 (澳门/威廉/bet365/易胜博)"""
        results = {}
        for bookmaker in BOOKMAKER_HANDICAP_IDS.get(OddsType.ODDS, []):
            result = self.download_odds_data(
                match_id,
                odds_type=OddsType.ODDS,
                bookmaker=bookmaker,
                use_browser=use_browser,
            )
            results[bookmaker.value] = result
            time.sleep(0.25)
        return results

    def download_all_overunder(self, match_id: str, use_browser: bool = True) -> Dict:
        """下载大小球所有庄家赔率"""
        results = {}
        for bookmaker in BOOKMAKER_HANDICAP_IDS.get(OddsType.OVERUNDER, []):
            result = self.download_odds_data(
                match_id,
                odds_type=OddsType.OVERUNDER,
                bookmaker=bookmaker,
                use_browser=use_browser,
            )
            results[bookmaker.value] = result
            time.sleep(0.25)
        return results

    def download_all_overunder_data(
        self, match_id: str, use_browser: bool = True
    ) -> Dict:
        """下载大小球所有庄家数据 (Macau/Bet365/EasyBet) 到 data/odds/overunder/ 目录"""
        overunder_companies = [
            (1, "macau"),
            (8, "bet365"),
            (12, "easybet"),
        ]

        results = {}
        for company_id, bookmaker_name in overunder_companies:
            result = self.download_overunder_to_data_overunder(
                match_id,
                company_id=company_id,
                bookmaker_name=bookmaker_name,
                use_browser=use_browser,
            )
            results[bookmaker_name] = result
            time.sleep(0.25)

        return results

    def download_overunder_to_data_overunder(
        self,
        match_id: str,
        company_id: int = 1,
        bookmaker_name: str = "macau",
        use_browser: bool = True,
    ) -> Dict:
        """下载大小球数据到 data/odds/overunder/ 目录"""
        url = f"https://vip.titan007.com/changeDetail/overunder.aspx?id={match_id}&companyID={company_id}&l=0"

        try:
            print(f"正在下载大小球数据 [{bookmaker_name}]: {url}")

            if use_browser:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="networkidle")
                    page.wait_for_timeout(3000)
                    html_content = page.content()
                    browser.close()
            else:
                response = self.session.get(url, timeout=30)
                if response.encoding == "ISO-8859-1":
                    response.encoding = response.apparent_encoding
                html_content = response.text

            if html_content:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.base_path}/odds/overunder/{match_id}_{bookmaker_name}_{timestamp}.html"
                self._ensure_dir(filename)

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(html_content)

                result = {
                    "match_id": match_id,
                    "company_id": company_id,
                    "bookmaker": bookmaker_name,
                    "url": url,
                    "download_time": timestamp,
                    "raw_file": filename,
                    "status": "success",
                    "content_length": len(html_content),
                    "use_browser": use_browser,
                }

                meta_file = f"{self.base_path}/odds/overunder/{match_id}_{bookmaker_name}_{timestamp}.json"
                self._ensure_dir(meta_file)

                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                print(f"✓ 大小球数据 [{bookmaker_name}] 下载成功: {filename}")
                return result
            else:
                return {
                    "match_id": match_id,
                    "company_id": company_id,
                    "bookmaker": bookmaker_name,
                    "url": url,
                    "status": "failed",
                    "error": "No content retrieved",
                }

        except ImportError:
            print("⚠ Playwright 未安装，使用 requests 降级获取")
            return self._download_overunder_requests(
                match_id, company_id, bookmaker_name
            )
        except Exception as e:
            print(f"✗ 下载大小球数据出错: {str(e)}")
            return {
                "match_id": match_id,
                "company_id": company_id,
                "url": url,
                "status": "failed",
                "error": str(e),
            }

    def _download_overunder_requests(
        self, match_id: str, company_id: int, bookmaker_name: str = "macau"
    ) -> Dict:
        """使用 requests 降级获取大小球数据"""
        url = f"https://vip.titan007.com/changeDetail/overunder.aspx?id={match_id}&companyID={company_id}&l=0"

        try:
            response = self.session.get(url, timeout=30)
            if response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding

            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.base_path}/odds/overunder/{match_id}_{bookmaker_name}_{timestamp}.html"
                self._ensure_dir(filename)

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)

                result = {
                    "match_id": match_id,
                    "company_id": company_id,
                    "bookmaker": bookmaker_name,
                    "url": url,
                    "download_time": timestamp,
                    "raw_file": filename,
                    "status": "success",
                    "content_length": len(response.text),
                    "use_browser": False,
                }

                meta_file = f"{self.base_path}/odds/overunder/{match_id}_{bookmaker_name}_{timestamp}.json"
                self._ensure_dir(meta_file)

                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                print(f"✓ 大小球数据 [{bookmaker_name}] 下载成功 (静态): {filename}")
                return result
            else:
                return {
                    "match_id": match_id,
                    "company_id": company_id,
                    "bookmaker": bookmaker_name,
                    "url": url,
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {
                "match_id": match_id,
                "company_id": company_id,
                "url": url,
                "status": "failed",
                "error": str(e),
            }

    def download_both(self, match_id: str, use_browser: bool = True) -> Dict:
        """同时下载分析数据和亚盘赔率(默认下载所有庄家)"""
        print(f"\n{'=' * 50}")
        print(f"开始下载比赛 {match_id} 的数据")
        print(f"{'=' * 50}\n")

        analysis_result = self.download_analysis_data(match_id, use_browser)
        time.sleep(1)
        handicap_result = self.download_all_handicap(match_id, use_browser)

        return {
            "match_id": match_id,
            "analysis": analysis_result,
            "handicap": handicap_result,
            "total_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def download_all_odds_data(
        self,
        match_id: str,
        use_browser: bool = True,
        include_handicap: bool = True,
        include_odds: bool = True,
        include_overunder: bool = True,
    ) -> Dict:
        """下载所有类型赔率数据"""
        print(f"\n{'=' * 50}")
        print(f"开始下载比赛 {match_id} 的所有赔率数据")
        print(f"{'=' * 50}\n")

        handicap_result = None
        odds_result = None
        overunder_result = None

        if include_handicap:
            print("\n--- 亚盘 Handciap ---")
            handicap_result = self.download_all_handicap(match_id, use_browser)
            time.sleep(0.25)

        if include_odds:
            print("\n--- 欧赔 Odds ---")
            odds_result = self.download_all_odds(match_id, use_browser)
            time.sleep(0.25)

        if include_overunder:
            print("\n--- 大小球 OverUnder ---")
            overunder_result = self.download_all_overunder(match_id, use_browser)

        return {
            "match_id": match_id,
            "total_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "handicap": handicap_result,
            "odds": odds_result,
            "overunder": overunder_result,
        }

    def get_downloaded_files(self, data_type: str = "all") -> Dict:
        """获取已下载的文件列表"""
        files = {"analysis": [], "handicap": [], "odds": [], "overunder": []}

        if data_type in ["all", "analysis"]:
            analysis_dir = f"{self.base_path}/analysis"
            if os.path.exists(analysis_dir):
                for f in os.listdir(analysis_dir):
                    if f.endswith(".html"):
                        files["analysis"].append(f)

        if data_type in ["all", "handicap", "odds"]:
            handicap_dir = f"{self.base_path}/odds/handicap"
            if os.path.exists(handicap_dir):
                for f in os.listdir(handicap_dir):
                    if f.endswith(".html"):
                        files["handicap"].append(f)

            odds_dir = f"{self.base_path}/odds/odds"
            if os.path.exists(odds_dir):
                for f in os.listdir(odds_dir):
                    if f.endswith(".html"):
                        files["odds"].append(f)

            overunder_dir = f"{self.base_path}/odds/overunder"
            if os.path.exists(overunder_dir):
                for f in os.listdir(overunder_dir):
                    if f.endswith(".html"):
                        files["overunder"].append(f)

        return files


if __name__ == "__main__":
    downloader = DataDownloader()

    print("赔率下载测试")
    print("1. 下载亚盘 (handicap) - 澳门/bet365/易胜博")
    print("2. 下载欧赔 (odds)")
    print("3. 下载大小球 (overunder)")
    print("4. 下载所有赔率")
    print("5. 下载分析数据")

    choice = input("\n请选择 (直接回车跳过): ").strip()

    test_id = input("请输入比赛编号: ").strip()

    if not test_id:
        print("未输入比赛编号，退出")
        exit(0)

    if choice == "1":
        result = downloader.download_all_handicap(test_id, use_browser=True)
    elif choice == "2":
        result = downloader.download_all_odds(test_id, use_browser=True)
    elif choice == "3":
        result = downloader.download_all_overunder(test_id, use_browser=True)
    elif choice == "4":
        result = downloader.download_all_odds_data(test_id, use_browser=True)
    elif choice == "5":
        result = downloader.download_analysis_data(test_id, use_browser=True)
    else:
        result = downloader.download_both(test_id, use_browser=True)

    print("\n下载结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
