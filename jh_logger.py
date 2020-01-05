import logging

def get_logger(logger_name, level):
    """ standard boilerplate code to get a logger
    use for logging at a level that will not change log 
    level for other loggers in library code
    @param logger_name: name of logger
    @ param level: logging level, e.g. INFO, DEBUG...
    @return logger, configured to log includes date and log level for messages
    """
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)    
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # add formatter to ch
    ch.setFormatter(formatter)
    
    # add ch to logger
    logger.addHandler(ch)
    return logger
