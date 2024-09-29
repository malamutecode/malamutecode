"""Logger configuration file."""
import logging
import os

from colorlog import ColoredFormatter

logging.getLogger('flake8').propagate = False
logging.getLogger('pydocstyle').propagate = False
log = logging.getLogger()


def config_logging(logging_level: int = logging.INFO) -> None:
    """Create logging configuration."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging_level, format='%(asctime)s %(levelname)-8s %(name)s %(funcName)s   %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S', filename=os.devnull, filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging_level)
    logs_colors = {'DEBUG': 'cyan', 'INFO': 'blue', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}
    formatter = ColoredFormatter("%(log_color)s%(asctime)s %(levelname)-8s %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S', reset=True, log_colors=logs_colors,
                                 secondary_log_colors={}, style='%')
    console.setFormatter(formatter)
    log.addHandler(console)
    log.info(f'Configured logger. Logging level: {logging.getLevelName(logging_level)}')


config_logging()
