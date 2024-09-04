import logging
import os
from sys import stdout

class CustomFormatter(logging.Formatter):
    """
    Custom formatter for logging messages with different colors based on log levels.

    Attributes:
    -----------
    grey : str
        ANSI escape sequence for grey color, used for INFO level logs.
    yellow : str
        ANSI escape sequence for yellow color, used for WARNING level logs.
    blue : str
        ANSI escape sequence for blue color, used for DEBUG level logs.
    pink : str
        ANSI escape sequence for pink color (not currently used).
    red : str
        ANSI escape sequence for red color, used for ERROR level logs.
    bold_red : str
        ANSI escape sequence for bold red color, used for CRITICAL level logs.
    reset : str
        ANSI escape sequence to reset the color formatting.
    format : str
        Default log message format string.

    Methods:
    --------
    format(record):
        Formats the log record according to its level with the appropriate color.
    
    configure_logging(analysisPrefix):
        Configures logging for the application with a stream handler and file handler.
    """
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    pink = "\x1b[35;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "###\t[%(asctime)s] %(filename)s:%(lineno)d: %(name)s %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        """
        Format the log record using the appropriate format string based on the log level.

        Parameters:
        -----------
        record : logging.LogRecord
            The log record to be formatted.

        Returns:
        --------
        str
            The formatted log message with appropriate color coding.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    @classmethod
    def configure_logging(cls, analysisPrefix=None):
        """
        Configures logging for the application by setting up stream and file handlers.

        Parameters:
        -----------
        analysisPrefix : str, optional
            The prefix to be used in the log file name. If not provided, it defaults to the name of the current script file.

        Returns:
        --------
        None
        """
        if analysisPrefix is None:
            analysisPrefix = "grumpy"
        
        logger = logging.getLogger()
        logger.disabled = True
        logger.setLevel(logging.INFO)
        
        # Create handlers for streaming and file output
        streamhdlr = logging.StreamHandler(stdout)
        filehdlr = logging.FileHandler(f".{analysisPrefix}.log")
        
        logger.addHandler(streamhdlr)
        logger.addHandler(filehdlr)
        
        # Set logging level for handlers
        streamhdlr.setLevel(logging.INFO)
        filehdlr.setLevel(logging.INFO)
        
        # Set the formatters for the handlers
        lgrPlainFormat = logging.Formatter('###\t[%(asctime)s] %(filename)s:%(lineno)d: %(name)s %(levelname)s: %(message)s')
        filehdlr.setFormatter(lgrPlainFormat)
        streamhdlr.setFormatter(cls())