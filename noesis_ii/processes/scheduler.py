"""
调度器（Scheduler）

P1 升级：实现设计文档 4.1 节的完整运行周期
- 清醒期（Wake）：整理频繁 + 记忆强化
- 整理期（Consolidation）：标准整理
- 睡眠/静止期（Sleep）：回放引擎（NREM → REM → 微觉醒）
- 深化期（Deepening）：人格生成 + 模式识别

时间节律自动化：根据当前时段自动选择合适的进程组合和频率。
"""

import sqlite3
import os
import datetime
import time
import threading
from typing import Dict, List, Optional
from itertools import chain
from noesis_ii.processes.consolidator import Consolidator
from noesis_ii.processes.deepener import Deepener
from noesis_ii.processes.replay_engine import ReplayEngine

# 旧架构模块（可选导入）
try:
    from noesis_ii.core.long_term_memory import LongTermMemory
except ImportError:
    LongTermMemory = None


# 时间节律定义（与设计文档 4.1 节对齐）
CIRCADIAN_PHASES = {
    'wake_morning': {
        'hours': set(range(6, 12)),
        'label': '清醒期（上午）',
        'processes': {
            'consolidator': {'interval': 1800, 'priority': 'high'},  # 30分钟
            'replay_engine': {'interval': 7200, 'priority': 'low'},  # 2小时
            'deepener': {'interval': 86400, 'priority': 'low'},      # 每天
        }
    },
    'wake_afternoon': {
        'hours': set(range(12, 18)),
        'label': '清醒期（下午）',
        'processes': {
            'consolidator': {'interval': 3600, 'priority': 'normal'},  # 1小时
            'replay_engine': {'interval': 10800, 'priority': 'normal'},  # 3小时
            'deepener': {'interval': 86400, 'priority': 'low'},
        }
    },
    'consolidation_evening': {
        'hours': set(range(18, 22)),
        'label': '整理期（傍晚）',
        'processes': {
            'consolidator': {'interval': 1800, 'priority': 'high'},  # 30分钟
            'replay_engine': {'interval': 3600, 'priority': 'high'},  # 1小时
            'deepener': {'interval': 43200, 'priority': 'normal'},  # 12小时
        }
    },
    'sleep_night': {
        'hours': set(chain(range(22, 24), range(0, 6))),
        'label': '睡眠/静止期',
        'processes': {
            'consolidator': {'interval': 7200, 'priority': 'low'},  # 2小时
            'replay_engine': {'interval': 1800, 'priority': 'high'},  # 30分钟（完整回放）
            'deepener': {'interval': 43200, 'priority': 'normal'},
        }
    }
}


class Scheduler:
    """
    调度器：时间节律自动化

    根据当前时段自动调整各进程的运行频率和优先级。
    """

    def __init__(self, db_path: str, llm_config: Dict = None):
        self.db_path = db_path
        self.llm_config = llm_config or {}

        self.consolidator = Consolidator(db_path, self.llm_config)
        self.deepener = Deepener(db_path, self.llm_config)
        self.replay_engine = ReplayEngine(db_path)
        self.long_term_memory = LongTermMemory(db_path) if LongTermMemory else None

        # 进程运行记录
        self.run_history = {
            'consolidator': {'last_run': None, 'run_count': 0},
            'deepener': {'last_run': None, 'run_count': 0},
            'replay_engine': {'last_run': None, 'run_count': 0}
        }

        self.running = False
        self.thread = None

    def start(self):
        """启动调度器"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        phase = self.get_current_phase()
        print(f"[SCHEDULER] Started in phase: {phase['label']}")

    def stop(self):
        """停止调度器"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("[SCHEDULER] Stopped")

    def _run(self):
        """主调度循环"""
        while self.running:
            phase = self.get_current_phase()
            processes = phase['processes']

            for process_name, config in processes.items():
                if self._should_run(process_name, config['interval']):
                    self._run_process(process_name, config)
                    self.run_history[process_name]['last_run'] = datetime.datetime.now()
                    self.run_history[process_name]['run_count'] += 1

            # 根据时段调整检查频率
            check_interval = 30 if phase['label'] == '睡眠/静止期' else 60
            time.sleep(check_interval)

    def _should_run(self, process_name: str, interval: int) -> bool:
        """检查进程是否应该运行"""
        history = self.run_history.get(process_name)
        if not history or history['last_run'] is None:
            return True

        elapsed = (datetime.datetime.now() - history['last_run']).total_seconds()
        return elapsed >= interval

    def _run_process(self, process_name: str, config: Dict):
        """运行进程"""
        priority = config.get('priority', 'normal')
        print(f"[SCHEDULER] Running {process_name} (priority={priority})")

        try:
            if process_name == 'consolidator':
                self.consolidator.run()
            elif process_name == 'deepener':
                self.deepener.run()
            elif process_name == 'replay_engine':
                # 睡眠期使用完整模式，其他时段使用快速模式
                phase = self.get_current_phase()
                if phase['label'] == '睡眠/静止期':
                    self.replay_engine.run(mode='full', limit=10)
                else:
                    self.replay_engine.run(mode='forward', limit=5)

            print(f"[SCHEDULER] {process_name} completed")
        except Exception as e:
            print(f"[SCHEDULER] {process_name} failed: {e}")

    def get_current_phase(self) -> Dict:
        """获取当前时间节律阶段"""
        current_hour = datetime.datetime.now().hour

        for phase_name, phase_config in CIRCADIAN_PHASES.items():
            if current_hour in phase_config['hours']:
                return {
                    'name': phase_name,
                    'label': phase_config['label'],
                    'hour': current_hour,
                    'processes': phase_config['processes']
                }

        # 默认
        return CIRCADIAN_PHASES['wake_afternoon']

    def get_schedule(self) -> Dict:
        """获取当前调度计划"""
        phase = self.get_current_phase()
        schedule = {
            'current_phase': phase['label'],
            'current_hour': phase['hour'],
            'processes': {}
        }

        for proc_name, config in phase['processes'].items():
            history = self.run_history.get(proc_name, {})
            last_run = history.get('last_run')

            next_run = "immediately"
            if last_run:
                interval = config['interval']
                next_time = last_run + datetime.timedelta(seconds=interval)
                next_run = next_time.isoformat()

            schedule['processes'][proc_name] = {
                'interval': config['interval'],
                'priority': config['priority'],
                'last_run': last_run.isoformat() if last_run else None,
                'next_run': next_run,
                'run_count': history.get('run_count', 0)
            }

        return schedule

    def run_now(self, process_name: str) -> bool:
        """立即运行指定进程"""
        if process_name not in self.run_history:
            print(f"[SCHEDULER] Unknown process: {process_name}")
            return False

        print(f"[SCHEDULER] Running '{process_name}' immediately")
        config = {'priority': 'manual', 'interval': 0}

        try:
            self._run_process(process_name, config)
            self.run_history[process_name]['last_run'] = datetime.datetime.now()
            self.run_history[process_name]['run_count'] += 1
            return True
        except Exception as e:
            print(f"[SCHEDULER] {process_name} failed: {e}")
            return False

    def run_full_cycle(self) -> Dict:
        """
        运行完整周期（手动触发）

        依次执行：整理 → 回放 → 深化
        """
        results = {}

        print("[SCHEDULER] === Full Cycle Start ===")

        # 1. 整理期
        print("[SCHEDULER] Phase 1: Consolidation")
        try:
            self.consolidator.run()
            results['consolidation'] = 'success'
        except Exception as e:
            results['consolidation'] = f'error: {e}'

        # 2. 睡眠期（回放）
        print("[SCHEDULER] Phase 2: Sleep/Replay")
        try:
            replay_results = self.replay_engine.run(mode='full', limit=10)
            results['replay'] = replay_results
        except Exception as e:
            results['replay'] = f'error: {e}'

        # 3. 深化期
        print("[SCHEDULER] Phase 3: Deepening")
        try:
            deep_results = self.deepener.run()
            results['deepening'] = deep_results
        except Exception as e:
            results['deepening'] = f'error: {e}'

        # 4. 遗忘
        print("[SCHEDULER] Phase 4: Forgetting")
        try:
            forgotten = self.long_term_memory.apply_forgetting()
            results['forgetting'] = f'{forgotten} nodes processed'
        except Exception as e:
            results['forgetting'] = f'error: {e}'

        print("[SCHEDULER] === Full Cycle Complete ===")
        return results

    def get_status(self) -> Dict:
        """获取调度器状态"""
        return {
            'running': self.running,
            'phase': self.get_current_phase()['label'],
            'run_history': {
                k: {
                    'last_run': v['last_run'].isoformat() if v['last_run'] else None,
                    'run_count': v['run_count']
                }
                for k, v in self.run_history.items()
            },
            'schedule': self.get_schedule()
        }

    # === 向后兼容方法 ===

    def update_schedule(self, process_name: str, interval: int) -> bool:
        """兼容旧接口：更新指定进程的调度间隔"""
        if process_name in self.run_history:
            phase = self.get_current_phase()
            if process_name in phase['processes']:
                phase['processes'][process_name]['interval'] = interval
                return True
        return False

    def get_next_run_times(self) -> Dict[str, str]:
        """兼容旧接口：获取下次运行时间"""
        schedule = self.get_schedule()
        return {
            proc_name: info.get('next_run', 'unknown')
            for proc_name, info in schedule.get('processes', {}).items()
        }

    def check_circadian_rhythm(self) -> Dict[str, bool]:
        """兼容旧接口：检查生理节律"""
        phase = self.get_current_phase()
        return {
            'wake_morning': phase['name'] == 'wake_morning',
            'wake_afternoon': phase['name'] == 'wake_afternoon',
            'consolidation_evening': phase['name'] == 'consolidation_evening',
            'sleep_night': phase['name'] == 'sleep_night'
        }
