"""
NOESIS-II 阶段7：CLI 交互测试
覆盖 --mode interactive / daemon / consolidate / deepen 模式
覆盖 help / status / retrieve / input / book / rss / web 命令
"""

import os
import sys
import subprocess
import time
import unittest

# 测试脚本路径: NOESIS-II v1.0/noesis_ii/test_scripts/test_cli.py
# __file__ = .../test_cli.py
# dirname = .../test_scripts/
# dirname = .../noesis_ii/
# dirname = .../NOESIS-II v1.0/  ← 项目根目录（main.py 中的 from noesis_ii.xxx 需要从此处运行）
TEST_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
NOESIS_PKG_DIR = os.path.dirname(TEST_SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(NOESIS_PKG_DIR)


class TestCLIModes(unittest.TestCase):
    """测试 CLI 运行模式"""

    def _run_cli(self, args, input_text=None, timeout=10, env_extra=None):
        """辅助方法：运行 CLI 命令并返回输出"""
        # main.py 使用 `from noesis_ii.xxx` 绝对导入，需要将项目根目录加入 PYTHONPATH
        main_py = os.path.join(NOESIS_PKG_DIR, 'main.py')
        cmd = [sys.executable, main_py] + args
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # 将项目根目录加入 PYTHONPATH
        existing_path = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + existing_path if existing_path else PROJECT_ROOT
        if env_extra:
            env.update(env_extra)

        try:
            if input_text:
                result = subprocess.run(
                    cmd, input=input_text,
                    capture_output=True, text=True,
                    timeout=timeout, cwd=PROJECT_ROOT, env=env
                )
            else:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout, cwd=PROJECT_ROOT, env=env
                )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, '', 'Timeout'

    def test_mode_consolidate(self):
        """测试 --mode consolidate 模式"""
        # Use a separate empty database to avoid processing production data
        test_db_dir = os.path.join(NOESIS_PKG_DIR, 'data')
        test_db = os.path.join(test_db_dir, 'test_cli_consolidate.db')
        if os.path.exists(test_db):
            os.remove(test_db)
        # Point to test DB via a temporary config override
        import tempfile, yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump({'storage': {'db_path': f'noesis_ii/data/test_cli_consolidate.db'}}, f)
            tmp_config = f.name
        try:
            rc, stdout, stderr = self._run_cli(
                ['--mode', 'consolidate', '--config', tmp_config],
                timeout=30
            )
            self.assertEqual(rc, 0, f"consolidate mode failed: {stderr}")
            self.assertIn('[INIT]', stdout, "Missing initialization message")
            self.assertIn('[PROCESS]', stdout, "Missing consolidation process message")
            self.assertIn('[DONE]', stdout, "Missing completion message")
        finally:
            if os.path.exists(tmp_config):
                os.remove(tmp_config)
            if os.path.exists(test_db):
                os.remove(test_db)

    def test_mode_deepen(self):
        """测试 --mode deepen 模式"""
        # Use a separate empty database to avoid processing production data
        test_db_dir = os.path.join(NOESIS_PKG_DIR, 'data')
        test_db = os.path.join(test_db_dir, 'test_cli_deepen.db')
        if os.path.exists(test_db):
            os.remove(test_db)
        import tempfile, yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump({'storage': {'db_path': f'noesis_ii/data/test_cli_deepen.db'}}, f)
            tmp_config = f.name
        try:
            rc, stdout, stderr = self._run_cli(
                ['--mode', 'deepen', '--config', tmp_config],
                timeout=30
            )
            self.assertEqual(rc, 0, f"deepen mode failed: {stderr}")
            self.assertIn('[INIT]', stdout, "Missing initialization message")
            self.assertIn('[PROCESS]', stdout, "Missing deepening process message")
            self.assertIn('[DONE]', stdout, "Missing completion message")
        finally:
            if os.path.exists(tmp_config):
                os.remove(tmp_config)
            if os.path.exists(test_db):
                os.remove(test_db)

    def test_mode_daemon(self):
        """测试 --mode daemon 模式（心跳后超时退出）"""
        main_py = os.path.join(NOESIS_PKG_DIR, 'main.py')
        cmd = [sys.executable, main_py, '--mode', 'daemon']
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        existing_path = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + existing_path if existing_path else PROJECT_ROOT

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=PROJECT_ROOT, env=env
            )
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate(timeout=3)
        except Exception as e:
            self.fail(f"daemon process failed: {e}")

        # 验证它确实启动了并输出了心跳
        self.assertIn('[DAEMON]', stdout,
            f"Missing daemon start message. stdout={stdout[:500] if stdout else '(empty)'} stderr={stderr[:200] if stderr else '(empty)'}")
        self.assertIn('Heartbeat', stdout, "Missing heartbeat output")

    def test_mode_interactive_help(self):
        """测试 --mode interactive 中的 help 命令"""
        input_commands = 'help\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive help failed: {stderr}\nstdout={stdout[:500]}")
        self.assertIn('[MODE]', stdout, "Missing interactive mode message")
        self.assertIn('help', stdout.lower(), "Missing help output")
        self.assertIn('retrieve', stdout.lower(), "Missing command listing")

    def test_mode_interactive_status(self):
        """测试 --mode interactive 中的 status 命令"""
        input_commands = 'status\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive status failed: {stderr}")
        self.assertIn('System Status', stdout, "Missing status output")
        self.assertIn('Config: Loaded', stdout, "Missing config status")
        self.assertIn('Database: Initialized', stdout, "Missing database status")

    def test_mode_interactive_retrieve(self):
        """测试 --mode interactive 中的 retrieve 命令"""
        input_commands = 'retrieve test\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive retrieve failed: {stderr}")
        self.assertIn('[RETRIEVE]', stdout, "Missing retrieve output")

    def test_mode_interactive_retrieve_empty(self):
        """测试 --mode interactive 中 retrieve 不带参数"""
        input_commands = 'retrieve\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive retrieve empty failed: {stderr}")
        self.assertIn('[ERROR]', stdout, "Missing error for empty retrieve")

    def test_mode_interactive_input(self):
        """测试 --mode interactive 中的 input 命令"""
        input_commands = 'input This is a CLI test input\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive input failed: {stderr}")
        self.assertIn('[OK]', stdout, "Missing input success message")
        self.assertIn('memory_id=', stdout, "Missing memory_id output")

    def test_mode_interactive_input_empty(self):
        """测试 --mode interactive 中 input 不带参数"""
        input_commands = 'input\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive input empty failed: {stderr}")
        self.assertIn('[ERROR]', stdout, "Missing error for empty input")

    def test_mode_interactive_unknown_command(self):
        """测试 --mode interactive 中未知命令"""
        input_commands = 'nonexistent_command\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive unknown command failed: {stderr}")
        self.assertIn('[ERROR]', stdout, "Missing error for unknown command")

    def test_mode_interactive_book_info_no_book(self):
        """测试 --mode interactive 中 book info 命令（无已加载书籍）"""
        input_commands = 'book info\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive book info failed: {stderr}")

    def test_mode_interactive_book_usage(self):
        """测试 --mode interactive 中 book 命令用法提示"""
        input_commands = 'book\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive book usage failed: {stderr}")
        self.assertIn('USAGE', stdout, "Missing book usage message")

    def test_mode_interactive_rss_usage(self):
        """测试 --mode interactive 中 rss 命令用法提示"""
        input_commands = 'rss\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive rss usage failed: {stderr}")
        self.assertIn('USAGE', stdout, "Missing rss usage message")

    def test_mode_interactive_web_usage(self):
        """测试 --mode interactive 中 web 命令用法提示"""
        input_commands = 'web\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"interactive web usage failed: {stderr}")
        self.assertIn('USAGE', stdout, "Missing web usage message")

    def test_default_mode(self):
        """测试默认模式（应为 interactive）"""
        input_commands = 'exit\n'
        rc, stdout, stderr = self._run_cli([], input_text=input_commands)
        self.assertEqual(rc, 0, f"default mode failed: {stderr}")
        self.assertIn('[MODE]', stdout, "Missing mode selection message")

    def test_full_interactive_flow(self):
        """测试完整交互流程：help -> status -> input -> retrieve -> exit"""
        input_commands = 'help\nstatus\ninput CLI full flow test\nretrieve CLI full flow\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"full interactive flow failed: {stderr}")
        self.assertIn('[MODE]', stdout)
        self.assertIn('Available commands', stdout)
        self.assertIn('System Status', stdout)
        self.assertIn('[OK]', stdout)
        self.assertIn('[RETRIEVE]', stdout)
        self.assertIn('[EXIT]', stdout)


class TestCLIMultiCommands(unittest.TestCase):
    """测试 CLI 多命令组合"""

    def _run_cli(self, args, input_text=None, timeout=10):
        main_py = os.path.join(NOESIS_PKG_DIR, 'main.py')
        cmd = [sys.executable, main_py] + args
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        existing_path = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + existing_path if existing_path else PROJECT_ROOT
        try:
            if input_text:
                result = subprocess.run(
                    cmd, input=input_text,
                    capture_output=True, text=True,
                    timeout=timeout, cwd=PROJECT_ROOT, env=env
                )
            else:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout, cwd=PROJECT_ROOT, env=env
                )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, '', 'Timeout'

    def test_multiple_inputs(self):
        """测试连续多次 input 命令"""
        input_commands = 'input First test input\ninput Second test input\ninput Third test input\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"multiple inputs failed: {stderr}")
        self.assertEqual(stdout.count('[OK]'), 3, "Expected 3 successful input operations")

    def test_input_then_retrieve(self):
        """测试 input 后立即 retrieve 验证"""
        unique_keyword = f'unique_test_{int(time.time())}'
        input_commands = f'input {unique_keyword}\nretrieve {unique_keyword}\nexit\n'
        rc, stdout, stderr = self._run_cli(['--mode', 'interactive'], input_text=input_commands)
        self.assertEqual(rc, 0, f"input+retrieve failed: {stderr}")
        self.assertIn('[OK]', stdout)
        self.assertIn('[RETRIEVE]', stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
