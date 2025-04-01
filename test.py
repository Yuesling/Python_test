import requests
import os
import time
import re
import random
from urllib.parse import quote
from fake_useragent import UserAgent


class BaiduImageDownloader:
    def __init__(self, keyword, save_dir='downloads', max_pages=3, delay=1.5, proxy=None):
        self.keyword = keyword
        self.save_dir = save_dir
        self.max_pages = max_pages
        self.delay = delay
        self.proxy = proxy
        self.ua = UserAgent()

        # 初始化设置
        self._prepare()

    def _prepare(self):
        """创建目录并检查环境"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not hasattr(self, 'session'):
            self.session = requests.Session()
            self.session.verify = False  # 关闭SSL验证（必要时）

    def _get_headers(self):
        """生成动态请求头"""
        return {
            'User-Agent': self.ua.chrome,
            'Referer': 'https://image.baidu.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Connection': 'keep-alive',
            'DNT': '1'  # 开启禁止追踪
        }

    def _fetch_page(self, page):
        """获取单页数据"""
        encoded_word = quote(self.keyword)
        base_url = 'https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=utf8&word=%E5%B0%8F%E5%A7%90%E5%A7%90%E5%9B%BE%E7%89%87&fr=ala&ala=1&alatpl=normal&pos=0&dyTabStr=MCwxMiwzLDEsMiwxMyw3LDYsNSw5'

        params = {
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'fp': 'result',
            'queryWord': encoded_word,
            'word': encoded_word,
            'pn': page * 30,
            'rn': 30,
            'gsm': hex(page * 30)[2:],
            't': int(time.time() * 1000)  # 动态时间戳
        }

        try:
            # 使用随机延迟（0.5倍到1.5倍基础延迟）
            time.sleep(self.delay * random.uniform(0.5, 1.5))

            response = self.session.get(
                base_url,
                headers=self._get_headers(),
                params=params,
                proxies={'http': self.proxy, 'https': self.proxy} if self.proxy else None,
                timeout=(10, 15)
            )

            # 响应验证
            if response.status_code != 200:
                print(f"[错误] 非200状态码：{response.status_code}")
                return None

            content = response.text.strip()
            if not content or '<html>' in content.lower():
                print("[警告] 收到可能的重定向页面")
                return None

            return response.json()

        except Exception as e:
            print(f"[网络错误] 第{page + 1}页：{str(e)}")
            if hasattr(response, 'text'):
                print(f"响应片段：{response.text[:200]}")
            return None

    def _extract_urls(self, json_data):
        """解析图片真实地址"""
        urls = []
        for item in json_data.get('data', []):
            if not isinstance(item, dict):
                continue

            # 多个可能包含真实URL的字段
            for field in ['hoverURL', 'thumbURL', 'middleURL', 'pic_url', 'objURL']:
                url = item.get(field)
                if url and re.match(r'^https?://', url):
                    urls.append(url)
                    break
        return urls

    def _download_image(self, url, save_path):
        """下载单张图片"""
        try:
            res = self.session.get(
                url,
                headers={'Referer': 'https://image.baidu.com/'},
                stream=True,
                timeout=10,
                allow_redirects=True
            )

            if res.status_code == 200:
                # 智能获取文件类型
                content_type = res.headers.get('Content-Type', '')
                ext_match = re.search(r'image/(\w+)', content_type)
                ext = ext_match.group(1) if ext_match else 'jpg'

                with open(f"{save_path}.{ext}", 'wb') as f:
                    for chunk in res.iter_content(2048):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"[下载失败] {url[:50]}... - {str(e)}")
            return False

    def run(self):
        """执行主程序"""
        print(f"🔍 开始抓取关键词：{self.keyword}")

        for page in range(self.max_pages):
            print(f"\n📄 正在处理第 {page + 1}/{self.max_pages} 页")

            json_data = self._fetch_page(page)
            if not json_data:
                print("⏭️ 跳过本页")
                continue

            image_urls = self._extract_urls(json_data)
            print(f"🖼️ 发现 {len(image_urls)} 张图片")

            for idx, url in enumerate(image_urls):
                filename = f"{self.keyword}_p{page + 1}_{idx + 1}"
                save_path = os.path.join(self.save_dir, filename)

                print(f"⬇️ 正在下载：{filename}  原地址：{url[:40]}...")
                if self._download_image(url, save_path):
                    print("✅ 下载成功")
                else:
                    print("❌ 下载失败")

                time.sleep(random.uniform(0.8, 1.2))  # 下载间隔

        print("\n🎉 任务完成！保存目录：", os.path.abspath(self.save_dir))


if __name__ == '__main__':
    # 使用方法：
    downloader = BaiduImageDownloader(
        keyword="自然风光",  # 搜索关键词
        save_dir="nature",  # 保存目录
        max_pages=2,  # 抓取页数（每页约30张）
        delay=2.5,  # 推荐设置2-5秒延迟
        # proxy='http://user:pass@host:port'  # 如果需要代理
    )
    downloader.run()
