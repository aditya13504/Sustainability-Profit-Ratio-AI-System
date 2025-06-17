"""
Logging utility module for the SPR Analyzer

Provides centralized logging configuration and setup.
"""

import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(config_path=None, default_level=logging.INFO):
    """
    Setup logging based on YAML configuration
    
    Args:
        config_path: Path to logging configuration YAML file
        default_level: Default logging level if config file not found
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'logging.yaml')
    
    # Get the project root directory (two levels up from src/utils)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(project_root, 'logs')
    
    # Create logs directory if it doesn't exist
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update file paths in config to use absolute paths
        for handler_name, handler_config in config.get('handlers', {}).items():
            if 'filename' in handler_config and not os.path.isabs(handler_config['filename']):
                handler_config['filename'] = os.path.join(project_root, handler_config['filename'])
        
        logging.config.dictConfig(config)
        
    except (FileNotFoundError, yaml.YAMLError) as e:
        # Fallback to basic configuration
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logs_dir, 'spr_analyzer.log'))
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load logging config from {config_path}: {e}")
        logger.info("Using fallback logging configuration")

def get_logger(name):
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)

class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    """
    
    @property
    def logger(self):
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger
    
    def log_method_call(self, method_name, *args, **kwargs):
        """Log a method call with parameters"""
        self.logger.debug(f"Calling {method_name} with args={args}, kwargs={kwargs}")
    
    def log_error(self, error, context=""):
        """Log an error with context"""
        self.logger.error(f"{context}: {str(error)}", exc_info=True)
    
    def log_performance(self, operation, duration):
        """Log performance metrics"""
        self.logger.info(f"Performance - {operation}: {duration:.2f}s")

# Initialize logging when module is imported
setup_logging()
