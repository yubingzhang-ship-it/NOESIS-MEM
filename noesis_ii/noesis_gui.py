#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOESIS-II GUI 应用
封装CLI功能，提供图形化界面
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
import sys
import yaml
import threading
import schedule
import time
from datetime import datetime

# 确保项目根目录在sys.path中
# _file_dir: noesis_gui.py 所在目录 (noesis_ii/)
# _project_root: 项目根目录 (NOESIS-II v1.0/)，用于 noesis_ii 包导入
_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_file_dir)
sys.path.insert(0, _project_root)

# 新架构核心模块
from noesis_ii.core.persona_profile import PersonaProfile
from noesis_ii.core.persona_extractor import PersonaExtractor, OCEANScores
from noesis_ii.core.persona_updater import PersonaUpdater
from noesis_ii.core.multi_criteria_retriever import MultiCriteriaRetriever, RetrievalCriteria
from noesis_ii.core.consistency_checker import ConsistencyChecker

# 废弃模块兼容导入（带警告）
try:
    from noesis_ii.core.schema import Schema
    from noesis_ii.core.working_memory import WorkingMemory
    from noesis_ii.core.long_term_memory import LongTermMemory
except ImportError as e:
    print(f"[WARN] Legacy module import failed: {e}")
    Schema = None
    WorkingMemory = None
    LongTermMemory = None

from noesis_ii.processes.consolidator import Consolidator
from noesis_ii.processes.deepener import Deepener
from noesis_ii.processes.scheduler import Scheduler
from noesis_ii.input.book_reader import BookReader
from noesis_ii.input.rss_fetcher import RSSFetcher
from noesis_ii.input.web_scraper import WebScraper
from noesis_ii.retrieval.retriever import Retriever
from noesis_ii.retrieval.hybrid_retriever import HybridRetriever
from noesis_ii.config_loader import ConfigLoader

# 数据库和配置路径（相对于 noesis_ii/ 目录）
DB_PATH = os.path.join(_file_dir, "data", "noesis.db")
CONFIG_PATH = os.path.join(_file_dir, "config", "default_config.yaml")

# 读取配置文件获取MinerU API密钥
def get_mineru_api_key():
    """从配置文件获取MinerU API密钥"""
    if os.path.exists(CONFIG_PATH):
        try:
            config_loader = ConfigLoader(CONFIG_PATH)
            config = config_loader.load()
            return config.get("mineru", {}).get("api_key", None)
        except Exception:
            pass
    return None

class NoesisIIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NOESIS-II 系统")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # 读取配置
        self.config = self._load_config()
        
        # 读取MinerU API密钥
        self.mineru_api_key = get_mineru_api_key()
        
        # 初始化数据库
        self.init_database()
        
        # ========== 性能优化：一次性初始化系统组件 ==========
        self._system_cache = None  # 系统组件缓存
        self._init_system_cache()
        
        # 初始化定时任务
        self.scheduled_tasks = []
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # 加载定时任务
        self.load_scheduled_tasks()
        
        # 启动调度器线程
        self.start_scheduler_thread()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建选项卡
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各个选项卡
        self.create_store_tab()
        self.create_retrieve_tab()
        self.create_context_tab()
        self.create_stats_tab()
        self.create_consolidate_tab()
        self.create_deepen_tab()
        self.create_book_reader_tab()
        self.create_rss_tab()
        self.create_web_scraper_tab()
        self.create_schedule_tab()
        self.create_scheduler_tab()
        self.create_cleanup_tab()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            config_loader = ConfigLoader(CONFIG_PATH)
            return config_loader.load()
        except Exception as e:
            messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")
            return {}
    
    def init_database(self):
        """初始化数据库"""
        try:
            # 确保数据目录存在
            data_dir = os.path.dirname(DB_PATH)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            
            # 初始化数据库
            schema = Schema(DB_PATH)
            schema.init_db()
        except Exception as e:
            messagebox.showerror("错误", f"数据库初始化失败: {str(e)}")
    
    def _init_system_cache(self):
        """性能优化：一次性初始化系统组件并缓存"""
        print("[GUI] 初始化系统组件...")
        try:
            # 加载LLM配置
            config_loader = ConfigLoader(CONFIG_PATH)
            config = config_loader.load()
            llm_config = config.get('llm', {})
            
            # 新架构核心组件
            persona_profile = PersonaProfile(db_path=DB_PATH)
            persona_extractor = PersonaExtractor()
            retriever = MultiCriteriaRetriever(db_path=DB_PATH)
            consistency_checker = ConsistencyChecker()
            
            # 旧架构兼容组件（如果可用）
            wm = WorkingMemory(DB_PATH) if WorkingMemory else None
            ltm = LongTermMemory(DB_PATH) if LongTermMemory else None
            consolidator = Consolidator(DB_PATH, llm_config)
            deepener = Deepener(DB_PATH, llm_config)
            scheduler = Scheduler(DB_PATH, llm_config)
            old_retriever = Retriever(DB_PATH)
            
            self._system_cache = {
                'persona_profile': persona_profile,
                'persona_extractor': persona_extractor,
                'retriever': retriever,
                'consistency_checker': consistency_checker,
                'wm': wm,
                'ltm': ltm,
                'consolidator': consolidator,
                'deepener': deepener,
                'scheduler': scheduler,
                'old_retriever': old_retriever,
                'llm_config': llm_config  # 缓存配置
            }
            print("[GUI] 系统组件初始化完成")
        except Exception as e:
            print(f"[GUI] 系统组件初始化失败: {e}")
            self._system_cache = None
    
    def get_system(self, use_real_llm=True):
        """获取系统实例（从缓存返回，避免重复初始化）"""
        if self._system_cache is not None:
            return self._system_cache
        
        # 缓存为空时，重新初始化
        print("[GUI] 系统缓存为空，重新初始化...")
        self._init_system_cache()
        return self._system_cache
    
    def create_store_tab(self):
        """创建存储记忆选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="存储记忆")
        
        # 创建输入区域
        input_frame = ttk.LabelFrame(tab, text="记忆内容", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 内容输入
        ttk.Label(input_frame, text="内容:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.content_text = scrolledtext.ScrolledText(input_frame, height=10, width=80)
        self.content_text.grid(row=0, column=1, sticky=tk.W+tk.E, pady=5)
        
        # 情绪
        ttk.Label(input_frame, text="情绪:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.emotion_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.emotion_var, width=50).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 按钮区域
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="存储记忆", command=self.store_memory).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空", command=self.clear_store_fields).pack(side=tk.LEFT, padx=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.store_result = scrolledtext.ScrolledText(result_frame, height=5, width=80)
        self.store_result.pack(fill=tk.BOTH, expand=True)
    
    def store_memory(self):
        """存储记忆（新架构）"""
        content = self.content_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("警告", "请输入记忆内容")
            return
        
        emotion = self.emotion_var.get()
        
        try:
            sys = self.get_system(use_real_llm=False)
            if not sys:
                raise Exception("系统初始化失败")
            
            # 使用新架构的 PersonaProfile 存储经验
            persona_profile = sys['persona_profile']
            trace_id = persona_profile.store_experience(
                experience=content,
                trace_type="user_input",
                intensity=0.7 if emotion else 0.5,
                context={"emotion": emotion} if emotion else {}
            )
            
            result = {
                "status": "ok",
                "id": trace_id,
                "content_preview": content[:60],
                "type": "memory_trace"
            }
            self.store_result.delete(1.0, tk.END)
            self.store_result.insert(tk.END, json.dumps(result, ensure_ascii=False, indent=2))
            messagebox.showinfo("成功", "记忆存储成功")
        except Exception as e:
            messagebox.showerror("错误", f"存储失败: {str(e)}")
    
    def clear_store_fields(self):
        """清空存储字段"""
        self.content_text.delete(1.0, tk.END)
        self.emotion_var.set("")
        self.store_result.delete(1.0, tk.END)
    
    def create_retrieve_tab(self):
        """创建检索记忆选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="检索记忆")

        # 输入区域
        input_frame = ttk.LabelFrame(tab, text="搜索", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="检索模式:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.retrieve_mode_var = tk.StringVar(value="hybrid")
        ttk.Radiobutton(input_frame, text="混合检索(向量+TF-IDF)", variable=self.retrieve_mode_var, value="hybrid").grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(input_frame, text="传统检索(TF-IDF)", variable=self.retrieve_mode_var, value="tfidf").grid(row=0, column=2, sticky=tk.W, pady=5)

        ttk.Label(input_frame, text="关键词:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.keyword_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.keyword_var, width=50).grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(input_frame, text="限制数量:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.limit_var = tk.IntVar(value=10)
        ttk.Entry(input_frame, textvariable=self.limit_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)

        ttk.Button(input_frame, text="搜索", command=self.retrieve_memory).grid(row=2, column=2, padx=10, pady=5)

        # 进度条
        self.retrieve_progress_var = tk.DoubleVar(value=0)
        progress_frame = ttk.LabelFrame(tab, text="进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Progressbar(progress_frame, variable=self.retrieve_progress_var, maximum=100).pack(fill=tk.X)
        self.retrieve_progress_label = ttk.Label(progress_frame, text="准备中...")
        self.retrieve_progress_label.pack(fill=tk.X, pady=5)

        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="搜索结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 整合结果
        integrated_frame = ttk.LabelFrame(result_frame, text="整合结果", padding="5")
        integrated_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.integrated_result = scrolledtext.ScrolledText(integrated_frame, height=15, width=80)
        self.integrated_result.pack(fill=tk.BOTH, expand=True)
    
    def retrieve_memory(self):
        """检索记忆"""
        keyword = self.keyword_var.get().strip()
        if not keyword:
            messagebox.showwarning("警告", "请输入搜索关键词")
            return

        limit = self.limit_var.get()
        mode = self.retrieve_mode_var.get()

        # 重置进度条
        self.retrieve_progress_var.set(0)
        self.retrieve_progress_label.config(text="开始检索记忆...")

        # 创建后台线程执行长时间操作
        def retrieve_thread():
            try:
                # 更新进度条
                def update_progress(value, text):
                    self.root.after(0, lambda: self.retrieve_progress_var.set(value))
                    self.root.after(0, lambda: self.retrieve_progress_label.config(text=text))

                # 执行检索
                update_progress(20, "初始化检索器...")

                if mode == "hybrid":
                    # 混合检索
                    update_progress(40, "加载混合检索器...")
                    retriever = HybridRetriever(DB_PATH)
                    update_progress(60, "执行混合检索...")
                    results = retriever.hybrid_search(keyword, top_k=limit)

                    # 转换结果格式
                    integrated_output = []
                    for r in results:
                        content = r.get("content", "")
                        integrated_output.append({
                            "seed_id": r.get("seed_id", ""),
                            "content": content[:200] + ("..." if len(content) > 200 else ""),
                            "similarity": round(r.get("similarity", 0), 4),
                            "vector_score": round(r.get("vector_score", 0), 4),
                            "tfidf_score": round(r.get("tfidf_score", 0), 4),
                            "mode": "混合检索"
                        })

                else:
                    # 传统检索
                    update_progress(40, "加载传统检索器...")
                    sys = self.get_system(use_real_llm=True)
                    if not sys:
                        raise Exception("系统初始化失败")
                    retriever = sys['old_retriever']
                    update_progress(60, "执行TF-IDF检索...")
                    results = retriever.retrieve(keyword, top_k=limit)

                    # 转换结果格式
                    integrated_output = []
                    for r in results.get("integrated", []):
                        content = r.get("content", "")
                        integrated_output.append({
                            "id": r.get("id", ""),
                            "content": content[:200] + ("..." if len(content) > 200 else ""),
                            "source": r.get("source", ""),
                            "weight": r.get("weight", 0),
                            "type": r.get("type", ""),
                            "level": r.get("level", ""),
                            "score": r.get("score", 0),
                            "mode": "传统检索"
                        })

                # 处理结果
                update_progress(80, "处理检索结果...")

                # 更新结果显示
                self.root.after(0, lambda: self.integrated_result.delete(1.0, tk.END))
                self.root.after(0, lambda: self.integrated_result.insert(tk.END, json.dumps(integrated_output, ensure_ascii=False, indent=2)))

                # 完成
                update_progress(100, "检索完成!")
            except Exception as e:
                # 显示错误消息
                self.root.after(0, lambda: messagebox.showerror("错误", f"检索失败: {str(e)}"))
                # 重置进度条
                self.root.after(0, lambda: self.retrieve_progress_var.set(0))
                self.root.after(0, lambda: self.retrieve_progress_label.config(text="准备中..."))

        # 启动后台线程
        thread = threading.Thread(target=retrieve_thread)
        thread.daemon = True
        thread.start()
    
    def create_context_tab(self):
        """创建上下文选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="上下文")
        
        # 按钮
        ttk.Button(tab, text="获取上下文", command=self.get_context).pack(pady=10)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="上下文摘要", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.context_result = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.context_result.pack(fill=tk.BOTH, expand=True)
    
    def get_context(self):
        """获取上下文（新架构）"""
        try:
            sys = self.get_system(use_real_llm=False)
            if not sys:
                return {"error": "系统初始化失败"}
            
            persona_profile = sys['persona_profile']
            retriever = sys['retriever']
            
            # 获取当前人格
            current_persona = persona_profile.get_current_persona()
            
            # 检索最近记忆
            recent_memories = retriever.retrieve(
                RetrievalCriteria(access_preference='recent'),
                top_k=20
            )
            
            # 旧架构兼容（如果可用）
            wm = sys.get('wm')
            ltm = sys.get('ltm')
            if wm and ltm:
                recent_wm = wm.get_all()[:20] if hasattr(wm, 'get_all') else []
                top_ltm = ltm.get_all_nodes(limit=10) if hasattr(ltm, 'get_all_nodes') else []
            else:
                recent_wm = []
                top_ltm = []
            
            output = {
                "current_persona": {
                    "openness": current_persona.openness,
                    "conscientiousness": current_persona.conscientiousness,
                    "extraversion": current_persona.extraversion,
                    "agreeableness": current_persona.agreeableness,
                    "neuroticism": current_persona.neuroticism
                },
                "recent_memories": [
                    {
                        "id": m.memory_id,
                        "content": (m.content[:100] + "..." if len(m.content) > 100 else m.content) if m.content else "(空)",
                        "score": m.score,
                        "matched_criteria": m.criteria_matched
                    }
                    for m in recent_memories[:20]
                ],
                "legacy_working_memory": [
                    {
                        "id": m.get("id"),
                        "content": m.get("content", "")[:100] + ("..." if len(m.get("content", "")) > 100 else ""),
                        "emotion": m.get("emotion", ""),
                        "created_at": m.get("timestamp", "")
                    }
                    for m in recent_wm
                ],
                "legacy_ltm": [
                    {
                        "id": n.get("id"),
                        "concept": n.get("content", "")[:100] + ("..." if len(n.get("content", "")) > 100 else ""),
                        "weight": n.get("weight", 0)
                    }
                    for n in top_ltm
                ]
            }
            
            self.context_result.delete(1.0, tk.END)
            self.context_result.insert(tk.END, json.dumps(output, ensure_ascii=False, indent=2))
        except Exception as e:
            messagebox.showerror("错误", f"获取上下文失败: {str(e)}")
    
    def create_stats_tab(self):
        """创建统计信息选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="统计信息")
        
        # 按钮
        ttk.Button(tab, text="获取统计", command=self.get_stats).pack(pady=10)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="系统统计", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.stats_result = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.stats_result.pack(fill=tk.BOTH, expand=True)
    
    def get_stats(self):
        """获取统计信息（新架构）"""
        try:
            sys = self.get_system(use_real_llm=False)
            if not sys:
                messagebox.showerror("错误", "系统初始化失败")
                return
            
            persona_profile = sys['persona_profile']
            retriever = sys['retriever']
            
            # 获取当前人格
            current_persona = persona_profile.get_current_persona()
            
            # 人格统计
            stats = {
                "persona": {
                    "openness": current_persona.openness,
                    "conscientiousness": current_persona.conscientiousness,
                    "extraversion": current_persona.extraversion,
                    "agreeableness": current_persona.agreeableness,
                    "neuroticism": current_persona.neuroticism
                },
                "memory_count": "N/A (新架构)"
            }
            
            # 旧架构兼容统计
            wm = sys.get('wm')
            ltm = sys.get('ltm')
            if wm and ltm:
                wm_count = len(wm.get_all()) if hasattr(wm, 'get_all') else 0
                ltm_count = len(ltm.get_all_nodes()) if hasattr(ltm, 'get_all_nodes') else 0
                stats["legacy"] = {
                    "working_memory": wm_count,
                    "long_term_memory": ltm_count
                }
            
            self.stats_result.delete(1.0, tk.END)
            self.stats_result.insert(tk.END, json.dumps(stats, ensure_ascii=False, indent=2))
        except Exception as e:
            messagebox.showerror("错误", f"获取统计失败: {str(e)}")
    
    def create_consolidate_tab(self):
        """创建整理记忆选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="整理记忆")
        
        # 输入区域
        input_frame = ttk.LabelFrame(tab, text="整理设置", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="最大处理条目数:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.max_items_var = tk.IntVar(value=50)
        ttk.Entry(input_frame, textvariable=self.max_items_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Button(input_frame, text="开始整理", command=self.consolidate_memory).grid(row=0, column=2, padx=10, pady=5)
        
        # 进度条
        self.consolidate_progress_var = tk.DoubleVar(value=0)
        progress_frame = ttk.LabelFrame(tab, text="进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Progressbar(progress_frame, variable=self.consolidate_progress_var, maximum=100).pack(fill=tk.X)
        self.consolidate_progress_label = ttk.Label(progress_frame, text="准备中...")
        self.consolidate_progress_label.pack(fill=tk.X, pady=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="整理结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.consolidate_result = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.consolidate_result.pack(fill=tk.BOTH, expand=True)
    
    def consolidate_memory(self):
        """整理记忆"""
        max_items = self.max_items_var.get()
        
        # 重置进度条
        self.consolidate_progress_var.set(0)
        self.consolidate_progress_label.config(text="开始整理记忆...")
        
        # 创建后台线程执行长时间操作
        def consolidate_thread():
            try:
                # 更新进度条
                def update_progress(value, text):
                    self.root.after(0, lambda: self.consolidate_progress_var.set(value))
                    self.root.after(0, lambda: self.consolidate_progress_label.config(text=text))
                
                # 获取系统实例
                sys = self.get_system(use_real_llm=True)
                if not sys:
                    raise Exception("系统初始化失败")
                consolidator = sys['consolidator']
                
                # 执行整理
                update_progress(30, "执行整理进程...")
                result = consolidator.run(limit=max_items)
                
                # 处理结果
                update_progress(80, "处理整理结果...")
                
                # 更新结果显示
                self.root.after(0, lambda: self.consolidate_result.delete(1.0, tk.END))
                self.root.after(0, lambda: self.consolidate_result.insert(tk.END, json.dumps(result, ensure_ascii=False, indent=2)))
                
                # 完成
                update_progress(100, "整理完成!")
                
                # 显示成功消息
                self.root.after(0, lambda: messagebox.showinfo("成功", f"记忆整理完成\n处理条目数: {result}\n创建节点数: {result}"))
            except Exception as e:
                # 显示错误消息
                self.root.after(0, lambda: messagebox.showerror("错误", f"整理失败: {str(e)}"))
                # 重置进度条
                self.root.after(0, lambda: self.consolidate_progress_var.set(0))
                self.root.after(0, lambda: self.consolidate_progress_label.config(text="准备中..."))
        
        # 启动后台线程
        thread = threading.Thread(target=consolidate_thread)
        thread.daemon = True
        thread.start()
    
    def create_deepen_tab(self):
        """创建深化记忆选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="深化记忆")
        
        # 按钮
        ttk.Button(tab, text="开始深化", command=self.deepen_memory).pack(pady=10)
        
        # 进度条
        self.deepen_progress_var = tk.DoubleVar(value=0)
        progress_frame = ttk.LabelFrame(tab, text="进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Progressbar(progress_frame, variable=self.deepen_progress_var, maximum=100).pack(fill=tk.X)
        self.deepen_progress_label = ttk.Label(progress_frame, text="准备中...")
        self.deepen_progress_label.pack(fill=tk.X, pady=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="深化结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.deepen_result = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.deepen_result.pack(fill=tk.BOTH, expand=True)
    
    def deepen_memory(self):
        """深化记忆"""
        # 重置进度条
        self.deepen_progress_var.set(0)
        self.deepen_progress_label.config(text="开始深化记忆...")
        
        # 创建后台线程执行长时间操作
        def deepen_thread():
            try:
                # 更新进度条
                def update_progress(value, text):
                    self.root.after(0, lambda: self.deepen_progress_var.set(value))
                    self.root.after(0, lambda: self.deepen_progress_label.config(text=text))
                
                # 获取系统实例
                sys = self.get_system(use_real_llm=True)
                if not sys:
                    raise Exception("系统初始化失败")
                deepener = sys['deepener']
                
                # 执行深化
                update_progress(30, "执行深化进程...")
                result = deepener.run()
                
                # 处理结果
                update_progress(80, "处理深化结果...")
                
                # 更新结果显示
                self.root.after(0, lambda: self.deepen_result.delete(1.0, tk.END))
                self.root.after(0, lambda: self.deepen_result.insert(tk.END, json.dumps(result, ensure_ascii=False, indent=2)))
                
                # 完成
                update_progress(100, "深化完成!")
                
                # 显示成功消息
                self.root.after(0, lambda: messagebox.showinfo("成功", f"记忆深化完成\n分析节点数: {result.get('high_weight_nodes_count', 0)}\n生成人格维度: {len(result.get('personality', {}))}"))
            except Exception as e:
                # 显示错误消息
                self.root.after(0, lambda: messagebox.showerror("错误", f"深化失败: {str(e)}"))
                # 重置进度条
                self.root.after(0, lambda: self.deepen_progress_var.set(0))
                self.root.after(0, lambda: self.deepen_progress_label.config(text="准备中..."))
        
        # 启动后台线程
        thread = threading.Thread(target=deepen_thread)
        thread.daemon = True
        thread.start()
    
    def create_book_reader_tab(self):
        """创建书籍阅读选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="书籍阅读")
        
        # 输入区域
        input_frame = ttk.LabelFrame(tab, text="书籍路径", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="书籍文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.book_path_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.book_path_var, width=50).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Button(input_frame, text="浏览", command=self.browse_book).grid(row=0, column=2, padx=10, pady=5)
        ttk.Button(input_frame, text="阅读", command=self.read_book).grid(row=0, column=3, padx=10, pady=5)
        
        # 进度条
        self.book_progress_var = tk.DoubleVar(value=0)
        progress_frame = ttk.LabelFrame(tab, text="进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Progressbar(progress_frame, variable=self.book_progress_var, maximum=100).pack(fill=tk.X)
        self.book_progress_label = ttk.Label(progress_frame, text="准备中...")
        self.book_progress_label.pack(fill=tk.X, pady=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="阅读结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.book_result = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.book_result.pack(fill=tk.BOTH, expand=True)
    
    def browse_book(self):
        """浏览书籍文件"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="选择书籍文件",
            filetypes=[("PDF文件", "*.pdf"), ("Markdown文件", "*.md"), ("文本文件", "*.txt"), ("EPUB文件", "*.epub"), ("所有文件", "*.*")]
        )
        if file_path:
            self.book_path_var.set(file_path)
    
    def read_book(self):
        """阅读书籍并存储到记忆"""
        book_path = self.book_path_var.get().strip()
        if not book_path or not os.path.exists(book_path):
            messagebox.showwarning("警告", "请选择有效的书籍文件")
            return
        
        # 重置进度条
        self.book_progress_var.set(0)
        self.book_progress_label.config(text="开始处理书籍...")
        
        # 创建后台线程执行长时间操作
        def read_book_thread():
            try:
                # 更新进度条
                def update_progress(value, text):
                    self.root.after(0, lambda: self.book_progress_var.set(value))
                    self.root.after(0, lambda: self.book_progress_label.config(text=text))
                
                # 创建BookReader实例，传递MinerU API密钥
                reader = BookReader(mineru_api_key=self.mineru_api_key)
                
                # 读取书籍
                update_progress(20, "读取书籍内容...")
                reader.load_book(book_path)
                result = reader.run()
                
                memory_id = None
                if result and 'content' in result:
                    # 存储到工作记忆
                    update_progress(60, "存储到工作记忆...")
                    sys = self.get_system(use_real_llm=False)
                    if not sys:
                        raise Exception("系统初始化失败")
                    wm = sys['wm']
                    if wm:
                        memory_id = wm.capture(result['content'], emotion=None)
                    else:
                        print("[WARN] WorkingMemory not available")
                
                # 处理结果
                update_progress(80, "处理阅读结果...")
                
                # 更新结果显示 - 更友好的格式
                def update_display():
                    self.book_result.delete(1.0, tk.END)
                    if result:
                        self.book_result.insert(tk.END, "=== 书籍阅读结果 ===\n")
                        self.book_result.insert(tk.END, f"书籍: {result.get('book', '未知')}\n")
                        self.book_result.insert(tk.END, f"阅读行数: {result.get('lines_read', 0)}\n")
                        if memory_id:
                            self.book_result.insert(tk.END, f"记忆ID: {memory_id}\n")
                        self.book_result.insert(tk.END, "\n=== 内容预览 ===\n")
                        content = result.get('content', '')
                        # 显示前2000个字符
                        self.book_result.insert(tk.END, content[:2000])
                        if len(content) > 2000:
                            self.book_result.insert(tk.END, "\n... (内容已截断)")
                
                self.root.after(0, update_display)
                
                # 完成
                update_progress(100, "处理完成!")
                
                # 显示成功消息
                success_message = f"书籍阅读完成"
                if memory_id:
                    success_message += f"，记忆ID: {memory_id}"
                self.root.after(0, lambda: messagebox.showinfo("成功", success_message))
            except Exception as e:
                # 显示错误消息
                self.root.after(0, lambda: messagebox.showerror("错误", f"阅读失败: {str(e)}"))
                # 重置进度条
                self.root.after(0, lambda: self.book_progress_var.set(0))
                self.root.after(0, lambda: self.book_progress_label.config(text="准备中..."))
        
        # 启动后台线程
        thread = threading.Thread(target=read_book_thread)
        thread.daemon = True
        thread.start()
    
    def create_rss_tab(self):
        """创建RSS订阅选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="RSS订阅")
        
        # 输入区域
        input_frame = ttk.LabelFrame(tab, text="RSS源", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="RSS URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.rss_url_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.rss_url_var, width=50).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 创建按钮区域
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=2, sticky=tk.W, pady=5)
        ttk.Button(button_frame, text="添加", command=self.add_rss_feed).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="获取", command=self.fetch_rss).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.rss_progress_var = tk.DoubleVar(value=0)
        progress_frame = ttk.LabelFrame(tab, text="进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Progressbar(progress_frame, variable=self.rss_progress_var, maximum=100).pack(fill=tk.X)
        self.rss_progress_label = ttk.Label(progress_frame, text="准备中...")
        self.rss_progress_label.pack(fill=tk.X, pady=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="RSS结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.rss_result = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.rss_result.pack(fill=tk.BOTH, expand=True)
    
    def add_rss_feed(self):
        """添加RSS订阅源"""
        rss_url = self.rss_url_var.get().strip()
        if not rss_url:
            messagebox.showwarning("警告", "请输入RSS URL")
            return
        
        try:
            rss_fetcher = RSSFetcher()
            rss_fetcher.add_feed(rss_url)
            messagebox.showinfo("成功", "RSS订阅源添加成功")
        except Exception as e:
            messagebox.showerror("错误", f"添加失败: {str(e)}")
    
    def fetch_rss(self):
        """获取RSS内容"""
        # 重置进度条
        self.rss_progress_var.set(0)
        self.rss_progress_label.config(text="开始获取RSS内容...")
        
        # 创建后台线程执行长时间操作
        def fetch_rss_thread():
            try:
                # 更新进度条
                def update_progress(value, text):
                    self.root.after(0, lambda: self.rss_progress_var.set(value))
                    self.root.after(0, lambda: self.rss_progress_label.config(text=text))
                
                # 创建RSSFetcher实例
                rss_fetcher = RSSFetcher()
                
                # 获取RSS内容
                update_progress(50, "获取RSS内容...")
                result = rss_fetcher.run()
                
                memory_id = None
                if result and 'content' in result:
                    # 存储到工作记忆
                    update_progress(60, "存储到工作记忆...")
                    sys = self.get_system(use_real_llm=False)
                    if not sys:
                        raise Exception("系统初始化失败")
                    wm = sys['wm']
                    if wm:
                        memory_id = wm.capture(result['content'], emotion=None)
                    else:
                        print("[WARN] WorkingMemory not available")
                
                # 处理结果
                update_progress(80, "处理RSS结果...")
                
                # 更新结果显示 - 更友好的格式
                def update_display():
                    self.rss_result.delete(1.0, tk.END)
                    if result:
                        self.rss_result.insert(tk.END, "=== RSS获取结果 ===\n")
                        if memory_id:
                            self.rss_result.insert(tk.END, f"记忆ID: {memory_id}\n")
                        self.rss_result.insert(tk.END, "\n=== 内容预览 ===\n")
                        content = result.get('content', '')
                        # 显示前2000个字符
                        self.rss_result.insert(tk.END, content[:2000])
                        if len(content) > 2000:
                            self.rss_result.insert(tk.END, "\n... (内容已截断)")
                
                self.root.after(0, update_display)
                
                # 完成
                update_progress(100, "获取完成!")
                
                # 显示成功消息
                success_message = f"RSS内容获取完成"
                if memory_id:
                    success_message += f"，记忆ID: {memory_id}"
                self.root.after(0, lambda: messagebox.showinfo("成功", success_message))
            except Exception as e:
                # 显示错误消息
                self.root.after(0, lambda: messagebox.showerror("错误", f"获取失败: {str(e)}"))
                # 重置进度条
                self.root.after(0, lambda: self.rss_progress_var.set(0))
                self.root.after(0, lambda: self.rss_progress_label.config(text="准备中..."))
        
        # 启动后台线程
        thread = threading.Thread(target=fetch_rss_thread)
        thread.daemon = True
        thread.start()
    
    def create_web_scraper_tab(self):
        """创建网页抓取选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="网页抓取")
        
        # 输入区域
        input_frame = ttk.LabelFrame(tab, text="网页URL", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.web_url_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.web_url_var, width=50).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 创建按钮区域
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=2, sticky=tk.W, pady=5)
        ttk.Button(button_frame, text="抓取", command=self.scrape_web).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.web_progress_var = tk.DoubleVar(value=0)
        progress_frame = ttk.LabelFrame(tab, text="进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Progressbar(progress_frame, variable=self.web_progress_var, maximum=100).pack(fill=tk.X)
        self.web_progress_label = ttk.Label(progress_frame, text="准备中...")
        self.web_progress_label.pack(fill=tk.X, pady=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="抓取结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.web_result = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.web_result.pack(fill=tk.BOTH, expand=True)
    
    def scrape_web(self):
        """抓取网页内容"""
        web_url = self.web_url_var.get().strip()
        if not web_url:
            messagebox.showwarning("警告", "请输入网页URL")
            return
        
        # 重置进度条
        self.web_progress_var.set(0)
        self.web_progress_label.config(text="开始抓取网页...")
        
        # 创建后台线程执行长时间操作
        def scrape_web_thread():
            try:
                # 更新进度条
                def update_progress(value, text):
                    self.root.after(0, lambda: self.web_progress_var.set(value))
                    self.root.after(0, lambda: self.web_progress_label.config(text=text))
                
                # 创建WebScraper实例
                web_scraper = WebScraper()
                
                # 抓取网页内容
                update_progress(50, "抓取网页内容...")
                result = web_scraper.run(web_url)
                
                memory_id = None
                if result and 'content' in result:
                    # 存储到工作记忆
                    update_progress(60, "存储到工作记忆...")
                    sys = self.get_system(use_real_llm=False)
                    if not sys:
                        raise Exception("系统初始化失败")
                    wm = sys['wm']
                    if wm:
                        memory_id = wm.capture(result['content'], emotion=None)
                    else:
                        print("[WARN] WorkingMemory not available")
                
                # 处理结果
                update_progress(80, "处理抓取结果...")
                
                # 更新结果显示 - 更友好的格式
                def update_display():
                    self.web_result.delete(1.0, tk.END)
                    if result:
                        self.web_result.insert(tk.END, "=== 网页抓取结果 ===\n")
                        self.web_result.insert(tk.END, f"URL: {result.get('url', '未知')}\n")
                        if memory_id:
                            self.web_result.insert(tk.END, f"记忆ID: {memory_id}\n")
                        self.web_result.insert(tk.END, "\n=== 内容预览 ===\n")
                        content = result.get('content', '')
                        # 显示前2000个字符
                        self.web_result.insert(tk.END, content[:2000])
                        if len(content) > 2000:
                            self.web_result.insert(tk.END, "\n... (内容已截断)")
                
                self.root.after(0, update_display)
                
                # 完成
                update_progress(100, "抓取完成!")
                
                # 显示成功消息
                success_message = f"网页抓取完成"
                if memory_id:
                    success_message += f"，记忆ID: {memory_id}"
                self.root.after(0, lambda: messagebox.showinfo("成功", success_message))
            except Exception as e:
                # 显示错误消息
                self.root.after(0, lambda: messagebox.showerror("错误", f"抓取失败: {str(e)}"))
                # 重置进度条
                self.root.after(0, lambda: self.web_progress_var.set(0))
                self.root.after(0, lambda: self.web_progress_label.config(text="准备中..."))
        
        # 启动后台线程
        thread = threading.Thread(target=scrape_web_thread)
        thread.daemon = True
        thread.start()
    
    def create_schedule_tab(self):
        """创建定时任务配置选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="定时任务")
        
        # 任务列表
        task_frame = ttk.LabelFrame(tab, text="定时任务", padding="10")
        task_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 任务列表树
        columns = ("id", "type", "time", "interval", "status")
        self.task_tree = ttk.Treeview(task_frame, columns=columns, show="headings")
        
        # 设置列标题
        self.task_tree.heading("id", text="ID")
        self.task_tree.heading("type", text="任务类型")
        self.task_tree.heading("time", text="执行时间")
        self.task_tree.heading("interval", text="执行间隔")
        self.task_tree.heading("status", text="状态")
        
        # 设置列宽
        self.task_tree.column("id", width=50)
        self.task_tree.column("type", width=100)
        self.task_tree.column("time", width=150)
        self.task_tree.column("interval", width=100)
        self.task_tree.column("status", width=80)
        
        self.task_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 任务操作按钮
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="添加任务", command=self.add_scheduled_task).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="删除任务", command=self.delete_scheduled_task).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="刷新任务", command=self.refresh_task_list).pack(side=tk.LEFT, padx=5)
        
        # 任务执行历史
        history_frame = ttk.LabelFrame(tab, text="任务执行历史", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 历史记录树
        history_columns = ("id", "type", "start_time", "status", "error")
        self.history_tree = ttk.Treeview(history_frame, columns=history_columns, show="headings")
        
        # 设置列标题
        self.history_tree.heading("id", text="ID")
        self.history_tree.heading("type", text="任务类型")
        self.history_tree.heading("start_time", text="执行时间")
        self.history_tree.heading("status", text="状态")
        self.history_tree.heading("error", text="错误信息")
        
        # 设置列宽
        self.history_tree.column("id", width=50)
        self.history_tree.column("type", width=100)
        self.history_tree.column("start_time", width=200)
        self.history_tree.column("status", width=80)
        self.history_tree.column("error", width=300)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 历史记录操作按钮
        history_button_frame = ttk.Frame(tab)
        history_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(history_button_frame, text="刷新历史", command=self.refresh_history_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(history_button_frame, text="清空历史", command=self.clear_history).pack(side=tk.LEFT, padx=5)
        
        # 刷新任务列表和历史记录
        self.refresh_task_list()
        self.refresh_history_list()
    
    def refresh_task_list(self):
        """刷新任务列表"""
        # 清空树
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        
        # 这里应该加载实际的任务列表
        # 暂时添加一些示例任务
        tasks = [
            {"id": 1, "type": "consolidate", "time": "02:00", "interval": "daily", "status": "active"},
            {"id": 2, "type": "deepen", "time": "03:00", "interval": "weekly", "status": "active"},
            {"id": 3, "type": "reading", "time": "08:00", "interval": "daily", "status": "active"}
        ]
        
        for task in tasks:
            self.task_tree.insert("", tk.END, values=(
                task["id"],
                task["type"],
                task["time"],
                task["interval"],
                task["status"]
            ))
    
    def refresh_history_list(self):
        """刷新任务执行历史列表"""
        # 清空树
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # 这里应该加载实际的历史记录
        # 暂时添加一些示例记录
        history = [
            {"id": 1, "type_name": "consolidate", "start_time": "2026-04-05 02:00:00", "status_name": "success", "error_message": ""},
            {"id": 2, "type_name": "deepen", "start_time": "2026-04-04 03:00:00", "status_name": "success", "error_message": ""},
            {"id": 3, "type_name": "reading", "start_time": "2026-04-05 08:00:00", "status_name": "success", "error_message": ""}
        ]
        
        # 添加历史记录，按时间倒序
        for record in reversed(history):
            # 限制错误信息长度
            error_message = record.get("error_message", "")
            if len(error_message) > 50:
                error_message = error_message[:50] + "..."
            
            self.history_tree.insert("", tk.END, values=(
                record["id"],
                record["type_name"],
                record["start_time"],
                record["status_name"],
                error_message
            ))
    
    def add_scheduled_task(self):
        """添加定时任务"""
        # 这里应该实现添加任务的逻辑
        messagebox.showinfo("信息", "添加任务功能待实现")
    
    def delete_scheduled_task(self):
        """删除定时任务"""
        # 这里应该实现删除任务的逻辑
        messagebox.showinfo("信息", "删除任务功能待实现")
    
    def clear_history(self):
        """清空任务执行历史"""
        # 这里应该实现清空历史的逻辑
        messagebox.showinfo("信息", "清空历史功能待实现")
    
    def create_cleanup_tab(self):
        """创建清除记忆选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="清除记忆")
        
        # 清除选项
        option_frame = ttk.LabelFrame(tab, text="清除选项", padding="10")
        option_frame.pack(fill=tk.X, pady=5)
        
        # 清除类型
        ttk.Label(option_frame, text="清除类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.cleanup_type_var = tk.StringVar(value="all")
        cleanup_type_frame = ttk.Frame(option_frame)
        cleanup_type_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(cleanup_type_frame, text="清除所有记忆", value="all", variable=self.cleanup_type_var).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(cleanup_type_frame, text="清除某一时段记忆", value="time_range", variable=self.cleanup_type_var).pack(side=tk.LEFT, padx=10)
        
        # 清除方式
        ttk.Label(option_frame, text="清除方式:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.cleanup_method_var = tk.StringVar(value="soft")
        cleanup_method_frame = ttk.Frame(option_frame)
        cleanup_method_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(cleanup_method_frame, text="软删除（标记为过期/删除）", value="soft", variable=self.cleanup_method_var).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(cleanup_method_frame, text="硬删除（彻底从数据库删除）", value="hard", variable=self.cleanup_method_var).pack(side=tk.LEFT, padx=10)
        
        # 时间范围选择（默认禁用）
        time_frame = ttk.LabelFrame(option_frame, text="时间范围", padding="10")
        time_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=5)
        
        ttk.Label(time_frame, text="开始时间:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.start_time_var = tk.StringVar(value="2026-01-01")
        ttk.Entry(time_frame, textvariable=self.start_time_var, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(time_frame, text="结束时间:").grid(row=0, column=2, sticky=tk.W, pady=5)
        self.end_time_var = tk.StringVar(value="2026-12-31")
        ttk.Entry(time_frame, textvariable=self.end_time_var, width=20).grid(row=0, column=3, sticky=tk.W, pady=5)
        
        # 记忆类型选择
        ttk.Label(option_frame, text="记忆类型:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.memory_type_cleanup_var = tk.StringVar(value="all")
        memory_type_frame = ttk.Frame(option_frame)
        memory_type_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(memory_type_frame, text="所有类型", value="all", variable=self.memory_type_cleanup_var).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(memory_type_frame, text="工作记忆", value="working", variable=self.memory_type_cleanup_var).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(memory_type_frame, text="长期记忆", value="long_term", variable=self.memory_type_cleanup_var).pack(side=tk.LEFT, padx=10)
        
        # 按钮区域
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="开始清除", command=self.cleanup_memory).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="恢复记忆", command=self.restore_memory).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.cleanup_progress_var = tk.DoubleVar(value=0)
        progress_frame = ttk.LabelFrame(tab, text="进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Progressbar(progress_frame, variable=self.cleanup_progress_var, maximum=100).pack(fill=tk.X)
        self.cleanup_progress_label = ttk.Label(progress_frame, text="准备中...")
        self.cleanup_progress_label.pack(fill=tk.X, pady=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="清除结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.cleanup_result = scrolledtext.ScrolledText(result_frame, height=10, width=80)
        self.cleanup_result.pack(fill=tk.BOTH, expand=True)
    
    def cleanup_memory(self):
        """清除记忆"""
        cleanup_type = self.cleanup_type_var.get()
        memory_type = self.memory_type_cleanup_var.get()
        cleanup_method = self.cleanup_method_var.get()
        
        # 重置进度条
        self.cleanup_progress_var.set(0)
        self.cleanup_progress_label.config(text="开始清除记忆...")
        
        # 创建后台线程执行长时间操作
        def cleanup_thread():
            try:
                # 更新进度条
                def update_progress(value, text):
                    self.root.after(0, lambda: self.cleanup_progress_var.set(value))
                    self.root.after(0, lambda: self.cleanup_progress_label.config(text=text))
                
                # 获取系统实例
                sys = self.get_system(use_real_llm=False)
                if not sys:
                    raise Exception("系统初始化失败")
                
                wm = sys.get('wm')
                ltm = sys.get('ltm')
                
                # 执行清除
                update_progress(30, "执行清除操作...")
                
                # 实现实际的清除逻辑
                deleted = 0
                
                if memory_type == "all" or memory_type == "working":
                    # 清除工作记忆
                    if wm:
                        if cleanup_method == "hard":
                            # 硬删除：从数据库中彻底删除
                            conn = wm.connect()
                            cursor = conn.cursor()
                            
                            if cleanup_type == "all":
                                # 清除所有工作记忆
                                cursor.execute("DELETE FROM working_memory")
                                deleted += cursor.rowcount
                            elif cleanup_type == "time_range":
                                # 清除指定时间范围内的工作记忆
                                start_time = self.start_time_var.get()
                                end_time = self.end_time_var.get()
                                cursor.execute("DELETE FROM working_memory WHERE timestamp BETWEEN ? AND ?", (start_time, end_time))
                                deleted += cursor.rowcount
                            
                            conn.commit()
                            wm.close()
                        else:
                            # 软删除：标记为已整理（这里简化处理）
                            # 由于WorkingMemory类没有提供批量软删除方法，我们需要手动实现
                            conn = wm.connect()
                            cursor = conn.cursor()
                            
                            if cleanup_type == "all":
                                # 标记所有工作记忆为已整理
                                cursor.execute("UPDATE working_memory SET is_consolidated = 1")
                                deleted += cursor.rowcount
                            elif cleanup_type == "time_range":
                                # 标记指定时间范围内的工作记忆为已整理
                                start_time = self.start_time_var.get()
                                end_time = self.end_time_var.get()
                                cursor.execute("UPDATE working_memory SET is_consolidated = 1 WHERE timestamp BETWEEN ? AND ?", (start_time, end_time))
                                deleted += cursor.rowcount
                            
                            conn.commit()
                            wm.close()
                
                if memory_type == "all" or memory_type == "long_term":
                    # 清除长期记忆
                    if ltm:
                        if cleanup_method == "hard":
                            # 硬删除：从数据库中彻底删除
                            # 由于LongTermMemory类没有提供批量删除方法，我们需要手动实现
                            conn = ltm.connect()
                            cursor = conn.cursor()
                            
                            # 先删除关联
                            cursor.execute("DELETE FROM ltm_links")
                            # 再删除节点
                            cursor.execute("DELETE FROM ltm_nodes")
                            deleted += cursor.rowcount
                            
                            conn.commit()
                            ltm.close()
                        else:
                            # 软删除：降低权重（这里简化处理）
                            # 由于LongTermMemory类没有提供批量软删除方法，我们需要手动实现
                            conn = ltm.connect()
                            cursor = conn.cursor()
                            
                            cursor.execute("UPDATE ltm_nodes SET weight = 0.01")
                            deleted += cursor.rowcount
                            
                            conn.commit()
                            ltm.close()
                
                result = {
                    "status": "success",
                    "message": "记忆清除完成",
                    "deleted": deleted
                }
                
                # 处理结果
                update_progress(80, "处理清除结果...")
                
                # 更新结果显示
                self.root.after(0, lambda: self.cleanup_result.delete(1.0, tk.END))
                self.root.after(0, lambda: self.cleanup_result.insert(tk.END, json.dumps(result, ensure_ascii=False, indent=2)))
                
                # 完成
                update_progress(100, "清除完成!")
                
                # 显示成功消息
                self.root.after(0, lambda: messagebox.showinfo("成功", f"记忆清除完成\n删除了 {result.get('deleted', 0)} 条记忆"))
            except Exception as e:
                # 显示错误消息
                self.root.after(0, lambda: messagebox.showerror("错误", f"清除失败: {str(e)}"))
                # 重置进度条
                self.root.after(0, lambda: self.cleanup_progress_var.set(0))
                self.root.after(0, lambda: self.cleanup_progress_label.config(text="准备中..."))
        
        # 启动后台线程
        thread = threading.Thread(target=cleanup_thread)
        thread.daemon = True
        thread.start()
    
    def restore_memory(self):
        """恢复记忆"""
        # 这里应该实现恢复记忆的逻辑
        messagebox.showinfo("信息", "恢复记忆功能待实现")
    
    def load_scheduled_tasks(self):
        """加载定时任务"""
        # 这里应该实现加载定时任务的逻辑
        pass
    
    def start_scheduler_thread(self):
        """启动调度器线程"""
        # 这里应该实现启动调度器线程的逻辑
        pass

    def create_scheduler_tab(self):
        """创建调度器选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="调度器")
        
        # 状态显示
        status_frame = ttk.LabelFrame(tab, text="调度器状态", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.scheduler_status_var = tk.StringVar(value="未启动")
        ttk.Label(status_frame, text="状态:", width=15).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.scheduler_status_var, width=20).grid(row=0, column=1, sticky=tk.W)
        
        # 调度配置
        config_frame = ttk.LabelFrame(tab, text="调度配置", padding="10")
        config_frame.pack(fill=tk.X, pady=5)
        
        # 整理进程配置
        ttk.Label(config_frame, text="整理进程 (秒):", width=20).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.consolidator_interval_var = tk.StringVar(value="3600")
        ttk.Entry(config_frame, textvariable=self.consolidator_interval_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 深化进程配置
        ttk.Label(config_frame, text="深化进程 (秒):", width=20).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.deepener_interval_var = tk.StringVar(value="86400")
        ttk.Entry(config_frame, textvariable=self.deepener_interval_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 重放引擎配置
        ttk.Label(config_frame, text="重放引擎 (秒):", width=20).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.replay_interval_var = tk.StringVar(value="10800")
        ttk.Entry(config_frame, textvariable=self.replay_interval_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="启动调度器", command=self.start_scheduler).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="停止调度器", command=self.stop_scheduler).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="更新配置", command=self.update_scheduler_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="获取状态", command=self.get_scheduler_status).pack(side=tk.LEFT, padx=5)
        
        # 立即运行按钮
        run_frame = ttk.LabelFrame(tab, text="立即运行", padding="10")
        run_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(run_frame, text="立即整理", command=lambda: self.run_scheduler_process('consolidator')).pack(side=tk.LEFT, padx=5)
        ttk.Button(run_frame, text="立即深化", command=lambda: self.run_scheduler_process('deepener')).pack(side=tk.LEFT, padx=5)
        ttk.Button(run_frame, text="立即重放", command=lambda: self.run_scheduler_process('replay_engine')).pack(side=tk.LEFT, padx=5)
        
        # 生理节律调整
        rhythm_frame = ttk.LabelFrame(tab, text="生理节律", padding="10")
        rhythm_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(rhythm_frame, text="按生理节律调整", command=self.adjust_by_rhythm).pack(side=tk.LEFT, padx=5)
        
        # 结果显示
        result_frame = ttk.LabelFrame(tab, text="调度器状态", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.scheduler_result = scrolledtext.ScrolledText(result_frame, height=15, width=80)
        self.scheduler_result.pack(fill=tk.BOTH, expand=True)

    def start_scheduler(self):
        """启动调度器"""
        try:
            sys = self.get_system(use_real_llm=True)
            scheduler = sys['scheduler']
            scheduler.start()
            self.scheduler_status_var.set("运行中")
            messagebox.showinfo("成功", "调度器已启动")
        except Exception as e:
            messagebox.showerror("错误", f"启动调度器失败: {e}")

    def stop_scheduler(self):
        """停止调度器"""
        try:
            sys = self.get_system(use_real_llm=True)
            scheduler = sys['scheduler']
            scheduler.stop()
            self.scheduler_status_var.set("已停止")
            messagebox.showinfo("成功", "调度器已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止调度器失败: {e}")

    def update_scheduler_config(self):
        """更新调度器配置"""
        try:
            consolidator_interval = int(self.consolidator_interval_var.get())
            deepener_interval = int(self.deepener_interval_var.get())
            replay_interval = int(self.replay_interval_var.get())
            
            sys = self.get_system(use_real_llm=True)
            scheduler = sys['scheduler']
            
            scheduler.update_schedule('consolidator', consolidator_interval)
            scheduler.update_schedule('deepener', deepener_interval)
            scheduler.update_schedule('replay_engine', replay_interval)
            
            messagebox.showinfo("成功", "调度器配置已更新")
        except Exception as e:
            messagebox.showerror("错误", f"更新配置失败: {e}")

    def get_scheduler_status(self):
        """获取调度器状态"""
        try:
            sys = self.get_system(use_real_llm=True)
            scheduler = sys['scheduler']
            status = scheduler.get_status()
            
            self.scheduler_result.delete(1.0, tk.END)
            self.scheduler_result.insert(tk.END, json.dumps(status, ensure_ascii=False, indent=2))
            
            # 更新状态显示
            if status['running']:
                self.scheduler_status_var.set("运行中")
            else:
                self.scheduler_status_var.set("已停止")
        except Exception as e:
            messagebox.showerror("错误", f"获取状态失败: {e}")

    def run_scheduler_process(self, process_name):
        """立即运行指定进程"""
        try:
            sys = self.get_system(use_real_llm=True)
            scheduler = sys['scheduler']
            success = scheduler.run_now(process_name)
            if success:
                messagebox.showinfo("成功", f"{process_name} 进程已启动")
            else:
                messagebox.showerror("错误", f"启动 {process_name} 进程失败")
        except Exception as e:
            messagebox.showerror("错误", f"运行进程失败: {e}")

    def adjust_by_rhythm(self):
        """按生理节律调整调度计划"""
        try:
            sys = self.get_system(use_real_llm=True)
            scheduler = sys['scheduler']
            scheduler.adjust_schedule_by_rhythm()
            
            # 更新界面显示
            status = scheduler.get_status()
            self.scheduler_result.delete(1.0, tk.END)
            self.scheduler_result.insert(tk.END, json.dumps(status, ensure_ascii=False, indent=2))
            
            messagebox.showinfo("成功", "已按生理节律调整调度计划")
        except Exception as e:
            messagebox.showerror("错误", f"调整失败: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NoesisIIGUI(root)
    root.mainloop()