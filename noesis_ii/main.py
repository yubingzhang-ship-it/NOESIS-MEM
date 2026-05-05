import os
import sys
import argparse
import time

# Force UTF-8 on Windows (fixes GBK console garbling)
# Use surrogateescape to handle non-UTF-8 bytes gracefully (e.g., piped GBK input)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.stdin.reconfigure(encoding='utf-8', errors='replace')

from core.schema import Schema
from config_loader import ConfigLoader
from retrieval.retriever import Retriever
from input.input_manager import InputManager
from input.book_reader import BookReader
from input.rss_fetcher import RSSFetcher
from input.web_scraper import WebScraper
from processes.consolidator import Consolidator
from processes.deepener import Deepener
from core.hgm import HierarchicalGenerativeModel
from core.multi_criteria_retriever import MultiCriteriaRetriever
from core.persona_profile import PersonaProfile

class NoesisII:
    def __init__(self):
        self.config = None
        self.db_schema = None
        self.retriever = None
        self.input_manager = None
        self.book_reader = None
        self.rss_fetcher = None
        self.web_scraper = None

    def initialize(self, config_path):
        """Initialize the system"""
        # Load config
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.load()

        # Init database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              self.config.get('storage.db_path', 'data/noesis.db'))
        self.db_schema = Schema(db_path)
        self.db_schema.init_db()

        # Init retrieval system
        self.retriever = Retriever(db_path)

        # Init input system
        self.input_manager = InputManager(db_path)

        # Get MinerU API key
        mineru_api_key = self.config.get('mineru', {}).get('api_key', None)

        # Init active input modules
        self.book_reader = BookReader(mineru_api_key=mineru_api_key)
        self.rss_fetcher = RSSFetcher()
        self.web_scraper = WebScraper()

        # Register input sources
        self.input_manager.register_input_source('book_reader', self.book_reader)
        self.input_manager.register_input_source('rss_fetcher', self.rss_fetcher)
        self.input_manager.register_input_source('web_scraper', self.web_scraper)

        print("[INIT] PersonaMem system initialized")

    def run_interactive(self):
        """Run in interactive mode"""
        print("[MODE] Entering interactive mode...")
        print("Type 'help' for available commands")

        while True:
            try:
                command = input(">>> ").strip()
                if command == 'exit':
                    break
                elif command == 'help':
                    self._print_help()
                elif command == 'status':
                    self._print_status()
                elif command.startswith('retrieve '):
                    query = command[9:].strip()
                    if query:
                        results = self.retriever.retrieve(query)
                        print(f"[RETRIEVE] Found {len(results['integrated'])} result(s)")
                        for i, result in enumerate(results['integrated'][:5]):
                            print(f"  {i+1}. {result['content'][:100]}...")
                    else:
                        print("[ERROR] Please provide a search query")
                elif command.startswith('input '):
                    input_content = command[6:].strip()
                    if input_content:
                        result = self.input_manager.process_input(input_content)
                        print(f"[OK] Input processed, memory_id={result['memory_id']}")
                    else:
                        print("[ERROR] Please provide input content")
                elif command.startswith('book '):
                    book_cmd = command[5:].strip().split(' ')
                    if len(book_cmd) >= 2:
                        if book_cmd[0] == 'load':
                            book_path = ' '.join(book_cmd[1:])
                            self.book_reader.load_book(book_path)
                        elif book_cmd[0] == 'read':
                            result = self.book_reader.read()
                            if result and 'content' in result:
                                process_result = self.input_manager.process_input(
                                    result['content'],
                                    source='book_reader'
                                )
                                print(f"[OK] Book content processed, memory_id={process_result['memory_id']}")
                        elif book_cmd[0] == 'info':
                            info = self.book_reader.get_book_info()
                            if info:
                                print(f"[INFO] Book: {info['book']}")
                                print(f"[INFO] Progress: {info['read_percentage']:.2f}%")
                    else:
                        print("[USAGE] Book command: book load <path> | book read | book info")
                elif command == 'book':
                    print("[USAGE] Book command: book load <path> | book read | book info")
                elif command.startswith('rss '):
                    rss_cmd = command[4:].strip().split(' ')
                    if len(rss_cmd) >= 2:
                        if rss_cmd[0] == 'add':
                            rss_url = ' '.join(rss_cmd[1:])
                            self.rss_fetcher.add_feed(rss_url)
                        elif rss_cmd[0] == 'fetch':
                            results = self.rss_fetcher.run()
                            if results and 'content' in results:
                                process_result = self.input_manager.process_input(
                                    results['content'],
                                    source='rss_fetcher'
                                )
                                print(f"[OK] RSS content processed, memory_id={process_result['memory_id']}")
                    else:
                        print("[USAGE] RSS command: rss add <URL> | rss fetch")
                elif command == 'rss':
                    print("[USAGE] RSS command: rss add <URL> | rss fetch")
                elif command.startswith('web '):
                    web_cmd = command[4:].strip().split(' ')
                    if len(web_cmd) >= 2:
                        if web_cmd[0] == 'scrape':
                            url = ' '.join(web_cmd[1:])
                            result = self.web_scraper.run(url)
                            if result and 'content' in result:
                                process_result = self.input_manager.process_input(
                                    result['content'],
                                    source='web_scraper'
                                )
                                print(f"[OK] Web content scraped, memory_id={process_result['memory_id']}")
                    else:
                        print("[USAGE] Web command: web scrape <URL>")
                elif command == 'web':
                    print("[USAGE] Web command: web scrape <URL>")
                else:
                    print(f"[ERROR] Unknown command: {command}")
            except KeyboardInterrupt:
                break

        print("[EXIT] Interactive mode ended")

    def run_daemon(self):
        """Run in daemon mode"""
        print("[DAEMON] Starting daemon mode...", flush=True)
        heartbeat = 0
        try:
            while True:
                heartbeat += 1
                print(f"[DAEMON] Heartbeat tick #{heartbeat} - running, timestamp={time.strftime('%H:%M:%S')}", flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            print("[DAEMON] Shutting down...", flush=True)

    def run_consolidation(self):
        """Run a single consolidation task"""
        print("[PROCESS] Running consolidation task...")
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              self.config.get('storage.db_path', 'data/noesis.db'))
        llm_config = self.config.get('llm', {})
        consolidator = Consolidator(db_path, llm_config=llm_config)
        count = consolidator.run(limit=10)
        print(f"[DONE] Consolidation complete: {count} memories processed")

    def run_deepening(self):
        """Run a single deepening task"""
        print("[PROCESS] Running deepening task...")
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              self.config.get('storage.db_path', 'data/noesis.db'))
        llm_config = self.config.get('llm', {})
        deepener = Deepener(db_path, llm_config=llm_config)
        result = deepener.run()
        print(f"[DONE] Deepening complete: {result.get('high_weight_nodes_count', 0)} nodes analyzed")

    def _print_help(self):
        """Print help information"""
        print("Available commands:")
        print("  help     - Show this help message")
        print("  status   - Show system status")
        print("  retrieve - Retrieve info (e.g., retrieve keyword)")
        print("  input    - Process input (e.g., input content)")
        print("  book     - Book commands (e.g., book load <path>)")
        print("  rss      - RSS commands (e.g., rss add <URL>)")
        print("  web      - Web scrape (e.g., web scrape <URL>)")
        print("  exit     - Exit the system")

    def _print_status(self):
        """Print system status"""
        print("System Status:")
        print("  Config: Loaded")
        print("  Database: Initialized")
        print("  Mode: Interactive")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PersonaMem - Persona-Consistent Memory System')
    parser.add_argument('--config', default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', default='interactive',
                        choices=['interactive', 'daemon', 'consolidate', 'deepen'],
                        help='Run mode')

    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)

    noesis = NoesisII()
    noesis.initialize(config_path)

    if args.mode == 'interactive':
        noesis.run_interactive()
    elif args.mode == 'daemon':
        noesis.run_daemon()
    elif args.mode == 'consolidate':
        noesis.run_consolidation()
    elif args.mode == 'deepen':
        noesis.run_deepening()


if __name__ == '__main__':
    main()
