import os
import datetime
import requests
from bs4 import BeautifulSoup
class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.cache = {}
    
    def scrape(self, url, selector=None):
        """抓取网页内容"""
        try:
            # 检查缓存
            if url in self.cache:
                cached_data = self.cache[url]
                print(f"[SCRAPER] Fetched from cache: {url}")
                return cached_data
            
            # 发送请求
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # 正确处理编码
            if response.encoding == 'ISO-8859-1':
                # 如果 requests 检测到错误的编码，尝试从内容中检测
                response.encoding = response.apparent_encoding
            
            # 解析网页
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if selector:
                # 使用选择器提取内容
                elements = soup.select(selector)
                content = '\n'.join([element.get_text(strip=True) for element in elements])
            else:
                # 提取整个网页文本
                content = soup.get_text(strip=True)
            
            # 清理文本，去除多余空白和换行
            import re
            
            # 只做基本的空白字符清理，不做特定网站的过滤
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            # 缓存结果
            result = {
                'content': content,
                'url': url,
                'fetched_at': datetime.datetime.now().isoformat()
            }
            self.cache[url] = result
            
            print(f"已抓取网页: {url}")
            return result
        except Exception as e:
            print(f"抓取网页失败: {e}")
            return None
    
    def run(self, url=None):
        """Run web scraper"""
        if not url:
            print("[SCRAPER] Please provide a URL to scrape")
            return None

        result = self.scrape(url)
        if result:
            print(f"[SCRAPER] Scraped content, length: {len(result['content'])} chars")
            return result
        else:
            return None

    def get_cache(self):
        """Get cache"""
        return self.cache

    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        print("[SCRAPER] All cache cleared")
        return True
    
    def scrape_multiple(self, urls):
        """批量抓取多个网页"""
        results = {}
        for url in urls:
            result = self.scrape(url)
            results[url] = result
        return results
    
    def extract_links(self, url):
        """提取网页中的链接"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # 正确处理编码
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for a in soup.find_all('a', href=True):
                link = a['href']
                # 处理相对链接
                if link.startswith('/'):
                    from urllib.parse import urljoin
                    link = urljoin(url, link)
                links.append({
                    'text': a.get_text(strip=True),
                    'url': link
                })
            
            print(f"从网页中提取了 {len(links)} 个链接")
            return links
        except Exception as e:
            print(f"提取链接失败: {e}")
            return []
    
    def extract_images(self, url):
        """提取网页中的图片"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # 正确处理编码
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            images = []
            
            for img in soup.find_all('img'):
                img_url = img.get('src')
                if img_url:
                    # 处理相对链接
                    if img_url.startswith('/'):
                        from urllib.parse import urljoin
                        img_url = urljoin(url, img_url)
                    images.append({
                        'url': img_url,
                        'alt': img.get('alt', '')
                    })
            
            print(f"从网页中提取了 {len(images)} 个图片")
            return images
        except Exception as e:
            print(f"提取图片失败: {e}")
            return []
    
    def search_in_page(self, url, keyword):
        """在网页中搜索关键词"""
        result = self.scrape(url)
        if result:
            content = result['content']
            keyword_lower = keyword.lower()
            content_lower = content.lower()
            
            if keyword_lower in content_lower:
                # 找到关键词的位置
                positions = []
                start = 0
                while True:
                    pos = content_lower.find(keyword_lower, start)
                    if pos == -1:
                        break
                    # 提取关键词周围的上下文
                    context_start = max(0, pos - 100)
                    context_end = min(len(content), pos + len(keyword) + 100)
                    context = content[context_start:context_end]
                    positions.append({
                        'position': pos,
                        'context': context
                    })
                    start = pos + len(keyword)
                
                print(f"在网页中找到 {len(positions)} 个关键词匹配")
                return {
                    'url': url,
                    'keyword': keyword,
                    'matches': positions
                }
            else:
                print("[SCRAPER] Keyword not found in page")
                return None
        else:
            return None