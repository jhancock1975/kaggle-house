import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from jh_logger import get_logger
import logging

logger = get_logger('keras_example', logging.DEBUG)

def load_data(train_file_name, test_file_name):
    test_df = pd.read_csv(test_file_name, header = 1)
    train_df = pd.read_csv(train_file_name, header = 1)
    return (train_df.iloc[:,1:-1], train_df.iloc[:,-1] ),(test_df.iloc[:,1:-1], test_df.iloc[:, -1])
    
if __name__ == "__main__":
    (train_data, train_targets), (test_data, test_targets) = load_data('/git/data/train.csv', '/git/data/test.csv')
    logger.debug('train_data {}'.format(train_data))
    logger.debug('train_targets {}'.format(train_targets))
    logger.debug('test_data {}'.format(test_data))
    logger.debug('test_targets {}'.format(test_targets))
