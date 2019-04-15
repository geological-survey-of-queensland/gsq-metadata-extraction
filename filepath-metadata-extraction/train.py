import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import preprocessing as pp

data_file = 'Processed 2D Files Training Data.csv'

x_cut = (slice(None), slice(0,6))
y_cut = (slice(None), slice(0,6))
epochs = 5


def main():
    """Load training data, run training and model saving process"""
    
    voc_size = pp.char_count
    data = pp.load('training_data.p')

    # spli data into x and y as well as training and test set
    (train_x, train_y), (test_x, test_y) = pp.train_test_split(data['LineName'], data['LineName'], test_frac=0.2, shuffle_before=True, shuffle_after=True)

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # one hot encode output because the model cant do that for some reason
    train_x = train_x[x_cut]
    test_x = test_x[x_cut]
    train_y = train_y[y_cut]
    test_y = test_y[y_cut]

    # train_x = np.random.uniform(low=0, high=pp.char_count, size=train_x.shape).astype(int)

    #train_x = keras.utils.to_categorical(train_x, voc_size)
    #test_x = keras.utils.to_categorical(test_x, voc_size)
    train_y = keras.utils.to_categorical(train_y, voc_size)
    test_y = keras.utils.to_categorical(test_y, voc_size)

    x_shape = [*train_x.shape]
    x_shape[0] = None
    y_shape = [*train_y.shape]
    y_shape[0] = None

    [x_shape_char] = x_shape[1:]
    #[x_shape_char, x_shape_ones] = x_shape[1:]
    [y_shape_char, y_shape_ones] = y_shape[1:]

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    #train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    #test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1] * train_y.shape[2])
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1] * test_y.shape[2])

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # create model
    model = keras.Sequential()
    model.add(keras.layers.Embedding(y_shape_ones, 50, name='le', input_length=x_shape_char))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(x_shape_char*10, activation='softmax', name='li'))
    #model.add(keras.layers.Dense(x_shape_char*10, activation='softmax', name='lh1'))
    # model.add(keras.layers.Dense(y_shape_char*y_shape_ones, activation='softmax', name='lh2'))
    model.add(keras.layers.Dense(y_shape_char*y_shape_ones, activation='relu', name='lo'))#, input_shape=(x_shape_char*y_shape_ones,)))

    # top_k_accuracy = lambda y_true, y_pred: keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=y_shape_char)
    def top_k_accuracy(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=y_shape_char)

    model.compile(optimizer='adam', loss='poisson', metrics=[top_k_accuracy, 'binary_accuracy', 'mean_squared_error'])
    print(model.summary())
    
    model.fit(train_x, train_y, epochs=epochs)


    while True:
        
        # query
        query = input("Input (q to exit): ")
        if query == 'q': break
        query = query.split()

        # preprocess x
        x_strings = [query[0]]
        x_vector = pp.vectorize_data(x_strings)
        x_padded = pp.pad_vector_data(x_vector, pp.char_to_int['<Padding>'], width=x_shape[1])[x_cut]
        x_one_hot = keras.utils.to_categorical(x_padded, voc_size)
        x_one_hot_flat = x_one_hot.reshape(-1, x_one_hot.shape[1] * x_one_hot.shape[2])

        if len(query) > 1:
            # preprocess y
            y_strings = [query[1]]
            y_vector = pp.vectorize_data(y_strings)
            y_padded = pp.pad_vector_data(y_vector, pp.char_to_int['<Padding>'], width=y_shape[1])[y_cut]
            y_one_hot = keras.utils.to_categorical(y_padded, voc_size)
            y_one_hot_flat = y_one_hot.reshape(-1, y_one_hot.shape[1] * y_one_hot.shape[2])

        # run
        p_one_hot_flat = model.predict(x_padded)
        if len(query) > 1:
            accuracy = model.evaluate(x_padded, y_one_hot_flat)

        # decode
        p_one_hot = p_one_hot_flat.reshape((-1, *y_shape[1:]))
        p_vector = np.argmax(p_one_hot, 2)
        p_strings = pp.decode_data(p_vector)

        # print
        print(p_strings)
        if len(query) > 1:
            print(*accuracy)



if __name__ == "__main__": main()