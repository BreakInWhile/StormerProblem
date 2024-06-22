import logging

def create_logger(name,level = None,format = None):

    # Create a custom logger
    logger = logging.getLogger(name)

    if level:
        logger.setLevel(level)
    else:
        logger.setLevel(logging.DEBUG)

    # Create a formatter
    if format:
        formatter = logging.Formatter(format)

    else:
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Create a handler and set the formatter
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
