import yaml
import os
import os.path
import json

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
    
    def load(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            self._create_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"[WARN] 加载配置文件失败: {e}")
            self.config = {}
        
        # 应用环境变量覆盖
        self._apply_env_overrides()
        
        # 从 WorkBuddy models.json 加载 LongCat API Key
        self._load_workbuddy_api_key()
        
        return self.config
    
    def get(self, key, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            "seed_values": {
                "compassion": 0.8,
                "wisdom": 0.7,
                "courage": 0.6,
                "integrity": 0.9
            },
            "active_reading": {
                "enabled": True,
                "sources": [
                    "books",
                    "rss",
                    "web"
                ]
            },
            "circadian_rhythm": {
                "morning": "06:00",
                "evening": "22:00"
            },
            "forgetting_curve": {
                "initial_strength": 1.0,
                "decay_rate": 0.05
            },
            "retrieval": {
                "top_k": 10,
                "threshold": 0.5
            },
            "consciousness": {
                "phi_threshold": 0.6,
                "broadcast_interval": 0.1
            },
            "llm_backend": {
                "type": "openai",
                "api_key": "${OPENAI_API_KEY}",
                "model": "gpt-3.5-turbo"
            },
            "storage": {
                "db_path": "data/noesis.db",
                "log_path": "logs/"
            }
        }
        
        # 确保配置目录存在
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        # 递归处理配置字典
        def process_dict(d, prefix=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    process_dict(value, f"{prefix}{key}.")
                elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    if env_var in os.environ:
                        d[key] = os.environ[env_var]
        
        process_dict(self.config)
    
    def _load_workbuddy_api_key(self):
        """从 WorkBuddy models.json 加载 LongCat API Key"""
        try:
            # WorkBuddy 配置文件路径
            workbuddy_config = os.path.expanduser("~/.workbuddy/models.json")
            
            if os.path.exists(workbuddy_config):
                with open(workbuddy_config, 'r', encoding='utf-8') as f:
                    models_data = json.load(f)
                
                # 查找 LongCat 模型的 API Key
                for model in models_data.get('models', []):
                    if model.get('id') == 'LongCat' or model.get('name') == 'LongCat':
                        api_key = model.get('apiKey')
                        if api_key:
                            # 更新 llm 配置
                            if 'llm' not in self.config:
                                self.config['llm'] = {}
                            self.config['llm']['api_key'] = api_key
                            # 同时设置 api_base（如果未配置）
                            if 'api_base' not in self.config['llm']:
                                self.config['llm']['api_base'] = 'https://api.longcat.chat/openai/v1'
                            print(f"[CONFIG] Loaded LongCat API Key from WorkBuddy models.json")
                            return
        except Exception as e:
            print(f"[WARN] Failed to load API Key from WorkBuddy: {e}")