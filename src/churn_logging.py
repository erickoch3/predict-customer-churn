"""module to configure logging for the library"""
import logging
import sys


def logging_init():
    """Initialize the logging package to store in a local file and output to stdout"""
    logging_filename = './logs/churn_library.log'
    try:
        logging.basicConfig(
            filename=logging_filename,
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s'
        )
    except FileNotFoundError:
        logging.error("Failed to find logging file '%s'", logging_filename)
    # Add a handler to output the logging to the console as well.
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
