"""
Configuration loader for the SPR Analyzer system
"""

import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigLoader:
    """Loads and manages configuration from YAML files and environment variables"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to the configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Determine config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'config.yaml'
            )
            
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Replace environment variable placeholders
            config = self._replace_env_vars(config)
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
            
    def _replace_env_vars(self, obj: Any) -> Any:
        """Recursively replace environment variable placeholders in config"""
        if isinstance(obj, dict):
            return {key: self._replace_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # Extract environment variable name
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)  # Return original if env var not found
        else:
            return obj
            
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def update(self, key_path: str, value: Any):
        """
        Update configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration key
            value: New value to set
        """
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
            
        # Set the value
        config_section[keys[-1]] = value
