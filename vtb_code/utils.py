import logging

logger = logging.getLogger(__name__)


def logging_init(handlers=None):
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s',
                        handlers=handlers)
