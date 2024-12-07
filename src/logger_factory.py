import logging
import sys


class LoggerFactory:
    def __init__(self):
        self.default_handler = logging.StreamHandler(stream=sys.stdout)
        self.default_level = logging.INFO
        self.default_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def create_logger(self, name, log_handler=None, log_level=None, log_format=None):
        # 创建一个logger
        logger = logging.getLogger(name)
        if log_level:
            logger.setLevel(log_level)
        else:
            logger.setLevel(self.default_level)

        # 创建一个formatter，用于控制日志信息的输出格式
        if log_format:
            formatter = logging.Formatter(log_format)
        else:
            formatter = self.default_formatter

        # 将formatter添加到handler
        if log_handler:
            log_handler.setFormatter(formatter)
            logger.addHandler(log_handler)
        else:
            self.default_handler.setFormatter(formatter)
            logger.addHandler(self.default_handler)

        return logger


logger_factory = LoggerFactory()
