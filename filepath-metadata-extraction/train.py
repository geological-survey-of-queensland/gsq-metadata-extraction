import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import preprocessing as pp

data_file = 'Processed 2D Files Training Data.csv'

def main():
    """Load training data, run training and model saving process"""
    
    data = pp.load('training_data.p')
    train, test = pp.train_test_split(data, test_frac=0.2, shuffle_before=False, shuffle_after=True)

    train_x, train_y = train['LookupDOSFilePath'].values, train['SurveyType'].values
    test_x, test_y = train['LookupDOSFilePath'].values, train['SurveyType'].values

    print(train_x)
    print(train_y)

    voc_size = pp.char_count

    model = karas.Sequential()
    model.add(keras.Embedding(voc_size, voc_size))
    model.add(keras.LSTM(voc_size, return_sequence=True))
    model.add(keras.LSTM(voc_size, return_sequence=True))
    model.add(keras.Dropout(0.5))
    model.add(keras.TimeDistributed(Dense(voc_size)))
    model.add(keras.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=2)
    model.evaluate(x_test, y_test)


class Batch():
    """
    Generate batches for a given dataset of given size.
    """
    
    def __init__(self, data_x, data_y, batch_size):
        """Initialise the generator with the dataset and batch size"""

        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size

        self.index = 0

    def generate(self):
        """Generate a batch randomly sampled from the dataset at run time"""

        lower = self.index
        upper = self.index + self.batch_size

        # if the end is reached, reset
        if upper >= len(self.data_x):
            upper = len(self.data_x)
            self.index = 0

        while True:
            yield data_x[lower:upper], data_y[lower:upper]



if __name__ == "__main__": main()