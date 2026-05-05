import os
import re
import textwrap
import requests
import time
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# 尝试导入必要的库
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None

# MinerU API 配置
MINERU_API_URL = "http://localhost:8000"
MINERU_PARSE_ENDPOINT = "/file_parse"
MINERU_API_TIMEOUT = 600

# MinerU 云 API 配置
MINERU_CLOUD_API_URL = "https://mineru.net/api/v4"
MINERU_CLOUD_POLL_INTERVAL = 5  # seconds between status checks
MINERU_CLOUD_TIMEOUT = 600  # max wait time for cloud parsing

class BookReader:
    def __init__(self, book_path=None, mineru_api_key=None):
        self.book_path = book_path
        self.current_book = None
        self.current_position = 0
        self.books = []
        self.mineru_api_key = mineru_api_key
        self.supported_formats = {".txt": self._read_txt, ".pdf": self._read_pdf, ".epub": self._read_epub, ".md": self._read_md}
    
    def load_book(self, book_path):
        """加载书籍"""
        if not os.path.exists(book_path):
            print(f"[BOOK] Book file not found: {book_path}")
            return False
        
        self.book_path = book_path
        self.current_book = book_path
        self.current_position = 0
        self.books.append(book_path)
        print(f"[BOOK] Loaded book: {os.path.basename(book_path)}")
        return True
    
    def read(self, num_lines=10):
        """Read book content"""
        if not self.current_book:
            print("[BOOK] Please load a book first")
            return None

        try:
            ext = os.path.splitext(self.current_book)[1].lower()
            if ext not in self.supported_formats:
                # 对于不支持的格式，使用默认的文本读取方式
                return self._read_default(num_lines)
            
            # 使用对应的读取函数
            read_func = self.supported_formats[ext]
            result = read_func(num_lines)
            return result
        except Exception as e:
            print(f"[BOOK] Failed to read book: {e}")
            return None
    
    def run(self):
        """Run book reader"""
        if not self.current_book:
            print("[BOOK] Please load a book first")
            return None

        result = self.read(20)
        if result:
            print(f"[BOOK] Read from book: {result['book']}")
        return result
    
    def get_book_info(self):
        """获取书籍信息"""
        if not self.current_book:
            return None
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(self.current_book)
            
            # 计算总行数或页数
            total_lines = 0
            read_percentage = 0.0
            
            ext = os.path.splitext(self.current_book)[1].lower()
            if ext == '.pdf' and PyPDF2 is not None:
                with open(self.current_book, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    total_lines = len(reader.pages)
                    # 简单估算已读比例
                    read_percentage = min(100.0, (self.current_position / total_lines) * 100) if total_lines > 0 else 0
            else:
                # 对于其他格式，使用默认的行数计算
                with open(self.current_book, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines = sum(1 for _ in f)
                
                with open(self.current_book, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self.current_position)
                    remaining_lines = sum(1 for _ in f)
                
                read_percentage = ((total_lines - remaining_lines) / total_lines) * 100 if total_lines > 0 else 0
            
            return {
                'book': os.path.basename(self.current_book),
                'path': self.current_book,
                'file_size': file_size,
                'total_lines': total_lines,
                'read_percentage': read_percentage,
                'current_position': self.current_position
            }
        except Exception as e:
            print(f"[BOOK] Failed to get book info: {e}")
            return None
    
    def set_position(self, position):
        """Set reading position"""
        if not self.current_book:
            print("[BOOK] Please load a book first")
            return False

        try:
            file_size = os.path.getsize(self.current_book)
            if position < 0 or position > file_size:
                print(f"[BOOK] Invalid position: {position}")
                return False

            self.current_position = position
            print(f"[BOOK] Position set to: {position}")
            return True
        except Exception as e:
            print(f"[BOOK] Failed to set position: {e}")
            return False
    
    def reset(self):
        """Reset reading state"""
        self.current_position = 0
        print("[BOOK] Reading state reset")
        return True

    def list_books(self):
        """List loaded books"""
        print("[BOOK] Loaded books:")
        for i, book in enumerate(self.books):
            print(f"  {i+1}. {os.path.basename(book)}")
        return self.books

    def remove_book(self, book_index):
        """Remove a book"""
        if 0 <= book_index < len(self.books):
            removed_book = self.books.pop(book_index)
            print(f"[BOOK] Removed book: {os.path.basename(removed_book)}")
            if self.current_book == removed_book:
                self.current_book = self.books[0] if self.books else None
                self.current_position = 0
            return True
        else:
            print(f"[BOOK] Invalid book index: {book_index}")
            return False
    
    def search_in_book(self, keyword):
        """Search keyword in book"""
        if not self.current_book:
            print("[BOOK] Please load a book first")
            return None

        try:
            ext = os.path.splitext(self.current_book)[1].lower()
            if ext == '.pdf' and PyPDF2 is not None:
                return self._search_pdf(keyword)
            else:
                return self._search_default(keyword)
        except Exception as e:
            print(f"[BOOK] Search failed: {e}")
            return None

    def _read_default(self, num_lines=10):
        """默认的文本读取方式"""
        try:
            with open(self.current_book, 'r', encoding='utf-8', errors='ignore') as f:
                # 移动到当前位置
                f.seek(self.current_position)
                
                # 读取指定行数
                lines = []
                for _ in range(num_lines):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.strip())
                
                # 更新当前位置
                self.current_position = f.tell()
                
                content = '\n'.join(lines)
                print(f"[BOOK] Read {len(lines)} lines")
                
                return {
                    'content': content,
                    'lines_read': len(lines),
                    'current_position': self.current_position,
                    'book': os.path.basename(self.current_book)
                }
        except Exception as e:
            print(f"[BOOK] Failed to read book: {e}")
            return None
    
    def _read_txt(self, num_lines=10):
        """读取TXT文件"""
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
            content = None
            
            for encoding in encodings:
                try:
                    with open(self.current_book, 'r', encoding=encoding) as f:
                        # 移动到当前位置
                        f.seek(self.current_position)
                        
                        # 读取内容
                        lines = []
                        for _ in range(num_lines * 5):  # 读取更多行以确保内容充足
                            line = f.readline()
                            if not line:
                                break
                            lines.append(line)
                        
                        # 更新当前位置
                        self.current_position = f.tell()
                        
                        content = ''.join(lines)
                        break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # 如果所有编码都失败，使用二进制模式读取并尝试解码
                try:
                    with open(self.current_book, 'rb') as f:
                        f.seek(self.current_position)
                        binary_content = f.read(4096)  # 读取4KB
                        self.current_position = f.tell()
                    
                    for encoding in encodings:
                        try:
                            content = binary_content.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                except Exception:
                    content = ""
            
            if content:
                # 按段落分割
                paragraphs = re.split(r'\n\s*\n', content)
                # 合并短段落，确保每个chunk有足够的内容
                current_chunk = []
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    current_chunk.append(para)
                    if len(' '.join(current_chunk)) > 1000:  # 每个chunk约1000字
                        break
                
                content = ' '.join(current_chunk)
                print(f"[BOOK] Read TXT file content")
                
                return {
                    'content': content,
                    'lines_read': len(content.split('\n')),
                    'current_position': self.current_position,
                    'book': os.path.basename(self.current_book)
                }
        except Exception as e:
            print(f"[BOOK] TXT read failed: {e}")
            return None
    
    def _read_pdf(self, num_pages=1):
        """读取PDF文件（使用MinerU API）"""
        pdf_path = Path(self.current_book)
        
        # 如果提供了API密钥，使用云端API
        if self.mineru_api_key:
            try:
                # 使用MinerU云API转换PDF为Markdown
                md_content = self._convert_pdf_with_mineru(pdf_path, self.mineru_api_key)
                # 将Markdown内容分割为片段
                chunks = self._split_markdown(md_content, num_pages)
                
                # Update current position
                self.current_position += len(chunks)

                print(f"[BOOK] Read PDF: {len(chunks)} pages")
                return {
                    'content': '\n'.join(chunks),
                    'lines_read': len(chunks),
                    'current_position': self.current_position,
                    'book': os.path.basename(self.current_book)
                }
            except Exception as e:
                # Cloud API failed, fall back to PyPDF2
                print(f"[BOOK] MinerU failed: {e}, falling back to PyPDF2")
        
        # 直接使用PyPDF2作为回退
        if PyPDF2 is not None:
            try:
                chunks = []
                with open(self.current_book, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num in range(min(num_pages, len(reader.pages))):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            # 清理文本
                            text = text.strip()
                            text = re.sub(r'\s+', ' ', text)
                            chunks.append(text)
                
                # Update current position
                self.current_position += len(chunks)

                print(f"[BOOK] Read PDF: {len(chunks)} pages")
                return {
                    'content': '\n'.join(chunks),
                    'lines_read': len(chunks),
                    'current_position': self.current_position,
                    'book': os.path.basename(self.current_book)
                }
            except Exception as e:
                print(f"[BOOK] PyPDF2 failed: {e}")
                return None
        else:
            print("[BOOK] PyPDF2 not installed, cannot read PDF")
            return None
    
    def _read_md(self, num_pages=1):
        """读取Markdown文件"""
        try:
            with open(self.current_book, 'r', encoding='utf-8', errors='ignore') as f:
                # 移动到当前位置
                f.seek(self.current_position)
                
                # 读取内容
                content = f.read(4096)  # 读取4KB
                
                # 更新当前位置
                self.current_position = f.tell()
                
                # 将Markdown内容分割为片段
                chunks = self._split_markdown(content, num_pages)
                
                print(f"[BOOK] Read Markdown: {len(chunks)} pages")
                return {
                    'content': '\n'.join(chunks),
                    'lines_read': len(chunks),
                    'current_position': self.current_position,
                    'book': os.path.basename(self.current_book)
                }
        except Exception as e:
            print(f"[BOOK] Markdown read failed: {e}")
            return None
    
    def _read_epub(self, num_pages=1):
        """Read EPUB file"""
        if ebooklib is None:
            print("[BOOK] ebooklib not installed, cannot read EPUB")
            return None

        try:
            book = epub.read_epub(self.current_book)
            chunks = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8', errors='ignore')
                    text = re.sub(r'<[^>]+>', '', content)
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    if text and len(text) > 100:
                        chunks.append(text)
                        if len(chunks) >= num_pages:
                            break

            self.current_position += len(chunks)

            print(f"[BOOK] Read EPUB: {len(chunks)} pages")
            return {
                'content': '\n'.join(chunks),
                'lines_read': len(chunks),
                'current_position': self.current_position,
                'book': os.path.basename(self.current_book)
            }
        except Exception as e:
            print(f"[BOOK] EPUB read failed: {e}")
            return None
    
    def _split_markdown(self, content: str, max_pages: int = 1) -> List[str]:
        """将Markdown内容分割为片段"""
        chunks = []
        
        # 按标题分割
        sections = re.split(r'(^#{1,6}\s+.+?)(?=^#{1,6}\s+|$)', content, flags=re.MULTILINE)
        
        current_chunk = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            current_chunk.append(section)
            if len(' '.join(current_chunk)) > 1000:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                if len(chunks) >= max_pages:
                    break
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks[:max_pages]
    
    def _search_default(self, keyword):
        """默认的搜索方式"""
        matches = []
        line_number = 1
        
        try:
            with open(self.current_book, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if keyword in line:
                        matches.append({
                            'line_number': line_number,
                            'content': line.strip()
                        })
                    line_number += 1
            
            print(f"[BOOK] Found {len(matches)} match(es) in book")
            return matches
        except Exception as e:
            print(f"[BOOK] Search failed: {e}")
            return None
    
    def _search_pdf(self, keyword):
        """在PDF文件中搜索"""
        matches = []
        
        try:
            with open(self.current_book, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text and keyword in text:
                        matches.append({
                            'page_number': page_num + 1,
                            'content': text.strip()
                        })
            
            print(f"[BOOK] Found {len(matches)} match(es) in PDF")
            return matches
        except Exception as e:
            print(f"[BOOK] PDF search failed: {e}")
            return None
    
    def _convert_pdf_with_mineru(self, pdf_path: Path, api_key: str) -> str:
        """使用MinerU API将PDF转换为Markdown"""
        # 这里实现与a-mem相同的MinerU API调用逻辑
        # 简化版实现，仅使用云端API
        cloud_url = MINERU_CLOUD_API_URL
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # Step 1: request signed upload URL
        data_id = pdf_path.stem
        payload = {
            "files": [{"name": pdf_path.name, "data_id": data_id}],
            "model_version": "pipeline",
            "enable_formula": True,
            "enable_table": True,
            "language": "ch",
        }
        
        try:
            resp = requests.post(
                f"{cloud_url}/file-urls/batch",
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            if resp.status_code != 200:
                raise Exception(f"云API HTTP {resp.status_code}: {resp.text[:200]}")
            
            resp_data = resp.json()
            if resp_data.get("code") != 0:
                raise Exception(f"云API错误: {resp_data.get('msg', resp.text[:200])}")
            
            batch_data = resp_data.get("data", {})
            batch_id = batch_data.get("batch_id", "")
            file_urls = batch_data.get("file_urls", [])
            if not file_urls:
                raise Exception("云API未返回上传URL")
            
            # 获取上传URL
            upload_url = file_urls[0] if isinstance(file_urls[0], str) else file_urls[0].get("url", "")
            if not upload_url:
                raise Exception("云API返回的上传URL为空")
            
            # Step 2: upload PDF via PUT
            with open(pdf_path, "rb") as f:
                put_resp = requests.put(
                    upload_url,
                    data=f,
                    timeout=120,
                )
            if put_resp.status_code not in (200, 201):
                raise Exception(f"PDF上传失败: HTTP {put_resp.status_code}")
            
            # Step 3: poll for results
            poll_headers = {"Authorization": f"Bearer {api_key}"}
            deadline = time.time() + MINERU_CLOUD_TIMEOUT
            
            while time.time() < deadline:
                time.sleep(MINERU_CLOUD_POLL_INTERVAL)
                try:
                    poll_resp = requests.get(
                        f"{cloud_url}/extract-results/batch/{batch_id}",
                        headers=poll_headers,
                        timeout=30,
                    )
                except requests.RequestException:
                    continue
                
                if poll_resp.status_code != 200:
                    continue
                
                try:
                    poll_data = poll_resp.json()
                except ValueError:
                    continue
                if poll_data.get("code") != 0:
                    continue
                
                extract_results = poll_data.get("data", {}).get("extract_result", [])
                if not extract_results:
                    continue
                
                item = extract_results[0]
                state = item.get("state", "")
                
                if state == "failed":
                    raise Exception(f"云端解析失败: {item.get('err_msg', 'unknown')}")
                
                if state == "done":
                    # 提取Markdown内容
                    md_content = None
                    # 尝试从不同位置提取Markdown
                    if isinstance(item, dict):
                        # 直接从item中提取
                        if "md_content" in item and isinstance(item["md_content"], str):
                            md_content = item["md_content"]
                        # 尝试从md_url下载
                        elif "md_url" in item:
                            try:
                                md_resp = requests.get(item["md_url"], timeout=60, proxies={"http": None, "https": None})
                                if md_resp.status_code == 200:
                                    md_content = md_resp.text
                            except Exception:
                                pass
                    
                    if md_content:
                        return md_content
                    else:
                        raise Exception("无法从云端结果提取Markdown内容")
                
                if state == "running":
                    pass  # 继续轮询
            
            raise Exception(f"云端解析超时（{MINERU_CLOUD_TIMEOUT}s）")
        except Exception as e:
            raise Exception(f"使用MinerU云API转换PDF失败: {str(e)}")