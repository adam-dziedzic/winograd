import logging
from logging.handlers import RotatingFileHandler

log_format = '%(asctime)s - %(name)s - %(levelname)s - ' \
             '%(funcName)s(%(lineno)d)- %(message)s'

"""
author: Adam Dziedzic ady@uchicago.edu
"""


def set_up_logging(log_file, is_debug=False):
    handlers = [get_console_handler(), get_log_file_handler(log_file)]
    level = logging.INFO
    if is_debug:
        level = logging.DEBUG
    logging.basicConfig(level=level, format=log_format,
                        handlers=handlers)
    logging.info("started logging to: " + log_file)


def get_log_file_handler(log_file):
    """
    Log into a file.

    :param log_file: the name of the file for logging
    :return: the handler to log into file
    """
    # https://goo.gl/FxA4Mh
    fh = RotatingFileHandler(log_file, mode='a', maxBytes=5 * 1024 * 1024,
                             backupCount=3, encoding=None, delay=0)
    # create formatter
    formatter = logging.Formatter(log_format)
    # add formatter
    fh.setFormatter(formatter)
    return fh


def get_console_handler():
    """
    Log into the console.

    :return: console log handler
    """
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter(log_format)
    # add formatter
    ch.setFormatter(formatter)
    return ch


def get_logger(name=__name__):
    # create logger
    logger = logging.getLogger(name)
    return logger
