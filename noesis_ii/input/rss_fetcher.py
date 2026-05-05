import os
import datetime
import feedparser
class RSSFetcher:
    def __init__(self):
        self.feeds = []
        self.cache = {}
    
    def add_feed(self, url, name=None):
        """添加RSS订阅源"""
        if not name:
            name = url.split('/')[-1]
        
        self.feeds.append({
            'url': url,
            'name': name,
            'last_fetched': None
        })
        print(f"[RSS] Added feed: {name} ({url})")
        return True
    
    def remove_feed(self, feed_index):
        """移除RSS订阅源"""
        if 0 <= feed_index < len(self.feeds):
            removed_feed = self.feeds.pop(feed_index)
            print(f"[RSS] Removed feed: {removed_feed['name']}")
            # 同时移除缓存
            if removed_feed['url'] in self.cache:
                del self.cache[removed_feed['url']]
            return True
        else:
            print(f"无效的订阅源索引: {feed_index}")
            return False
    
    def fetch_feed(self, feed_url):
        """获取单个RSS源"""
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                print(f"[RSS] Failed to fetch feed: {feed.bozo_exception}")
                return None
            
            # 缓存结果
            self.cache[feed_url] = {
                'content': feed,
                'fetched_at': datetime.datetime.now().isoformat()
            }
            
            # 更新最后获取时间
            for feed_item in self.feeds:
                if feed_item['url'] == feed_url:
                    feed_item['last_fetched'] = datetime.datetime.now().isoformat()
                    break
            
            print(f"[RSS] Fetched feed: {feed.feed.get('title', 'Unknown')}")
            return feed
        except Exception as e:
            print(f"[RSS] Failed to fetch feed: {e}")
            return None
    
    def fetch_all(self):
        """获取所有RSS源"""
        results = {}
        for feed in self.feeds:
            result = self.fetch_feed(feed['url'])
            results[feed['name']] = result
        return results
    
    def run(self):
        """运行RSS订阅器"""
        # 获取所有RSS源
        results = self.fetch_all()
        
        # 处理最新的内容
        latest_content = []
        for feed_name, feed in results.items():
            if feed and 'entries' in feed:
                # 获取最新的5条内容
                for entry in feed['entries'][:5]:
                    content = {
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', ''),
                        'feed': feed_name
                    }
                    latest_content.append(content)
        
        if latest_content:
            # 合并内容
            merged_content = '\n\n'.join([f"【{item['feed']}】{item['title']}\n{item['summary']}\n{item['link']}" for item in latest_content])
            return {
                'content': merged_content,
                'items': latest_content
            }
        else:
            return None
    
    def get_feeds(self):
        """获取所有订阅源"""
        return self.feeds
    
    def get_cache(self):
        """获取缓存"""
        return self.cache
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        print("[RSS] All cache cleared")
        return True
    
    def search_feeds(self, keyword):
        """搜索RSS内容"""
        results = []
        
        # 先获取所有RSS源
        self.fetch_all()
        
        # 搜索缓存中的内容
        for feed_url, feed_data in self.cache.items():
            feed = feed_data['content']
            if 'entries' in feed:
                for entry in feed['entries']:
                    if keyword.lower() in entry.get('title', '').lower() or keyword.lower() in entry.get('summary', '').lower():
                        results.append({
                            'title': entry.get('title', ''),
                            'link': entry.get('link', ''),
                            'summary': entry.get('summary', ''),
                            'published': entry.get('published', ''),
                            'feed': next((f['name'] for f in self.feeds if f['url'] == feed_url), feed_url)
                        })
        
        print(f"搜索到 {len(results)} 个匹配结果")
        return results
    
    def get_latest_entries(self, feed_index, limit=5):
        """获取指定订阅源的最新条目"""
        if 0 <= feed_index < len(self.feeds):
            feed = self.feeds[feed_index]
            feed_data = self.fetch_feed(feed['url'])
            
            if feed_data and 'entries' in feed_data:
                entries = []
                for entry in feed_data['entries'][:limit]:
                    entries.append({
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', '')
                    })
                return entries
            else:
                return []
        else:
            print(f"无效的订阅源索引: {feed_index}")
            return []