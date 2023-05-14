"""
Logger's setup file
"""

import logging
import os.path
import sys

# base level of logging
LOGGING_LEVEL = logging.INFO

# name of the file where logs will be store in
# 'logs.log' by default
LOG_FILE_NAME = 'logs.log'

# path to the folder where logs file stored
# ./recsys/logs by default
LOG_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), './../logs')

# full path to logs file
LOG_FILE_PATH = os.path.join(LOG_FOLDER_PATH, LOG_FILE_NAME)

# if logs` folder is not exit - create it
if not os.path.exists(LOG_FOLDER_PATH):
    os.makedirs(LOG_FOLDER_PATH)

# logger stream hadler to write info in console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOGGING_LEVEL)

# logger BasicConfig that uses two handlers
# FileHandler - to write log-unit in file
# and StreamHadler to write in console
logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(levelname)s::%(asctime)s::%(module)s::%(funcName)s::%(filename)s::%(lineno)d %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='a'), stream_handler],
                    datefmt='%d-%b-%y %H:%M:%S'
                    )

# logger instance to import to another modules
logger = logging.getLogger(__name__)


def log_import_error(type, value, traceback):
    logger.error(f"Import Error: {value}", exc_info=(type, value, traceback))


sys.excepthook = log_import_error

