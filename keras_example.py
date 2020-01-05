import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from jh_logger import get_logger
import logging
from keras import models
from keras import layers

# this is some pratice code based on
# https://www.kaggle.com/shanekonaung/boston-housing-price-dataset-with-keras

logger = get_logger('keras_example', logging.DEBUG)

def load_data(train_file_name, test_file_name):
    """ load training data from kaggle dataset
    @param train_file_name: fully qualified file name of train.csv from kaggle
    housing practice competition
    @param test_file_name: fully qualified file name of test.csv from kaggle
    housing practice competition
    @return: pair of tuples of observations and responses
    """
    test_df = pd.read_csv(test_file_name, header = 1)
    train_df = pd.read_csv(train_file_name, header = 1)
    return (train_df.iloc[:,1:-1], train_df.iloc[:,-1] ),(test_df.iloc[:,1:-1], test_df.iloc[:, -1])

def build_model():
    """ build keras model
    @return: keras model
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
    return model

if __name__ == "__main__":
    (train_data, train_targets), (test_data, test_targets) = load_data('/git/data/house/train.csv', '/git/data/house/test.csv')

    logger.debug(f'train_data {train_data}')
    logger.debug(f'train_targets {train_targets}')
    logger.debug(f'test_data {test_data}')
    logger.debug(f'test_targets {test_targets}')

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []

    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i+1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
            axis=0)
    model = build_model()
    model.fit(partial_train_data,
              partial_train_targets,
            epochs=num_epochs,
              batch_size=1,
              verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

    logger.debug(f'all_scores : {all_scores}')
    logger.debug(f'mean all scores : {np.mean(all_scores)}')

    model = build_model()
    model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

    logger.debug(f'test_mae_score {test_mae_score}')
