import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'console_debug': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        'src': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False
        },
        'src.agents.trading_agent': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False
        },
        'src.models.client': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False
        },
        'src.analysis': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False
        },
        'langgraph': {
            'level': 'WARNING',
            'handlers': ['console'],
            'propagate': False
        },
        'langchain': {
            'level': 'WARNING',
            'handlers': ['console'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}


def setup_logging(enable_debug: bool = False):
    """
    Setup logging to terminal only.
    
    Args:
        enable_debug: If True, enable DEBUG level logging to console
    """
    config = LOGGING_CONFIG.copy()
    
    # If debug is enabled, use console_debug instead of console
    if enable_debug:
        config['loggers']['src']['handlers'] = ['console_debug']
        config['loggers']['src.agents.trading_agent']['handlers'] = ['console_debug']
        config['loggers']['src.models.client']['handlers'] = ['console_debug']
        config['root']['handlers'] = ['console_debug']
    
    logging.config.dictConfig(config)
    
    # Return logger for main module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialised")
    if enable_debug:
        logger.info("DEBUG MODE ENABLED")
    
    return logger