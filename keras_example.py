import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from jh_logger import get_logger
import logging
from keras import models
from keras import layers
import util

util.seed_random_number_generators()

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
    test_df = pd.read_csv(test_file_name, index_col='Id')    
    train_df = pd.read_csv(train_file_name, index_col='Id')

    def encode_cats(df):
        """ one-hot encode all categorical values in data frame
        @param df: a dataframe
        @return: one hot encoded equivalents of non-numerical
        columns in data frame
        """
        df_numeric = df.select_dtypes(include = np.number)
        non_numeric_columns = df.select_dtypes(include=np.object).columns
        df_one_hot =  pd.get_dummies(df[non_numeric_columns])
        return pd.concat([df_numeric, df_one_hot], axis  = 1)
    
    return (encode_cats(train_df.iloc[:,1:-1]), train_df.iloc[:,-1] ), \
        encode_cats(test_df)

def build_model():
    """ build keras model
    @return: keras model
    """
    model = models.Sequential()
    model.add(layers.Dropout(0.2, input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
    return model

if __name__ == "__main__":
    (train_data, train_targets), test_data = load_data('/git/data/house/train.csv', '/git/data/house/test.csv')


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
    verbose_mode = 1
    

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
                  verbose=verbose_mode)

        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
    
    logger.debug(f'all_scores : {all_scores}')
    logger.debug(f'mean all scores : {np.mean(all_scores)}')


    predictions = model.predict(test_data)

    logger.debug(f'predictions {predictions}')
