import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import preprocessing as pp

data_file = 'Processed 2D Files Training Data.csv'

def main():
    """Load training data, run training and model saving process"""
    
    voc_size = pp.char_count
    data = pp.load('training_data.p')

    # spli data into x and y as well as training and test set
    (train_x, train_y), (test_x, test_y) = pp.train_test_split(data['LookupDOSFilePath'], data['LineName'], test_frac=0.2, shuffle_before=True, shuffle_after=True)

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # one hot encode output because the model cant do that for some reason
    train_x = keras.utils.to_categorical(train_x, voc_size)
    test_x = keras.utils.to_categorical(test_x, voc_size)
    train_y = keras.utils.to_categorical(train_y, voc_size)
    test_y = keras.utils.to_categorical(test_y, voc_size)

    x_shape = [*train_x.shape]
    x_shape[0] = None
    y_shape = [*train_y.shape]
    y_shape[0] = None

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1] * train_y.shape[2])
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1] * test_y.shape[2])

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


    # create model
    model = keras.Sequential()
    model.add(keras.layers.Dense(train_x.shape[1], input_shape=[train_x.shape[1]], activation='relu'))
    model.add(keras.layers.Dense(1000, activation='relu'))
    #model.add(keras.layers.Embedding(voc_size, voc_size*train_x.shape[1], input_length=train_x.shape[1]))
    #model.add(keras.layers.LSTM(voc_size, activation='relu', return_sequences=True, input_shape=train_x.shape[:]))
    #model.add(keras.layers.LSTM(voc_size, activation='relu', return_sequences=False))
    #model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.TimeDistributed(keras.layers.Dense(voc_size)))
    model.add(keras.layers.Dense(train_y.shape[1], activation='relu'))
    model.add(keras.layers.Activation('softmax'))

    print(model.summary())

    # compile, run and evaluate
    optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.predict(test_x[:1]).shape)

    model.fit(train_x, train_y, epochs=1)
    loss_metric = model.evaluate(test_x, test_y)
    print(loss_metric)


    while True:
        
        query = input("Input (q to exit): ")
        if query == 'q': break

        x_strings = [query]
        x_vector = pp.vectorize_data(x_strings)
        x_padded = pp.pad_vector_data(x_vector, pp.char_to_int['<Padding>'], width=x_shape[1])
        x_one_hot = keras.utils.to_categorical(x_padded, voc_size)
        x_one_hot_flat = x_one_hot.reshape(-1, x_one_hot.shape[1] * x_one_hot.shape[2])

        y_one_hot_flat = model.predict(x_one_hot_flat)

        y_one_hot = y_one_hot_flat.reshape((-1, *y_shape[1:]))
        y_vector = np.argmax(y_one_hot, 2)
        y_strings = pp.decode_data(y_vector)

        print(y_strings)


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