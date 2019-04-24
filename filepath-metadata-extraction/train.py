import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import preprocessing as pp
import sys, inspect

def accuracy(y_true, y_pred):
    argmax_true = tf.math.argmax(y_true)
    argmax_pred = tf.math.argmax(y_pred)
    match = tf.equal(argmax_true, argmax_pred)
    match = tf.cast(match, tf.int32)
    return tf.reduce_mean(match)

#######################################################################################################################
# Configuration
#######################################################################################################################

data_file = 'Processed 2D Files Training Data.csv'
model_file = 'model.h5'

reuse_model = len(sys.argv) > 1 and sys.argv[1] == 'reuse'
x_cut = (slice(None), slice(0,12))
y_cut = (slice(None), slice(0,12))
epochs = 100
embedding_size = 20
metrics = ['mean_absolute_error', 'categorical_accuracy', 'binary_accuracy', accuracy]
loss = 'poisson'


def main():
    """Load training data, run training and model saving process"""
    
    voc_size = pp.char_count
    data = pp.load('training_data.p')

    # spli data into x and y as well as training and test set
    # Unique Record ID	FileName	Original_FileName	SurveyNum	SurveyName	LineName	SurveyType	PrimaryDataType	SecondaryDataType	TertiaryDataType	Quaternary	File_Range	First_SP_CDP	Last_SP_CDP	CompletionYear	TenureType	Operator Name	GSQBarcode	EnergySource	LookupDOSFilePath

    (train_x, train_y), (test_x, test_y) = pp.train_test_split(data['LineName'], data['LineName'], test_frac=0.2, shuffle_before=True, shuffle_after=True)

    print('train_x', train_x.shape, 'train_y', train_y.shape, 'test_x', test_x.shape, 'test_y', test_y.shape, sep='\t')

    # original size
    x_org_shape = [*train_x.shape]
    x_org_shape[0] = None
    y_org_shape = [*train_y.shape]
    y_org_shape[0] = None

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

    #[x_shape_char] = x_shape[1:]
    x_shape_char, x_shape_ones, *_ = x_shape[1:] + [None]
    y_shape_char, y_shape_ones, *_ = y_shape[1:] + [None]

    print('train_x', train_x.shape, 'train_y', train_y.shape, 'test_x', test_x.shape, 'test_y', test_y.shape, sep='\t')

    #train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    #test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
    #train_y = train_y.reshape(train_y.shape[0], train_y.shape[1] * train_y.shape[2])
    #test_y = test_y.reshape(test_y.shape[0], test_y.shape[1] * test_y.shape[2])

    print('train_x', train_x.shape, 'train_y', train_y.shape, 'test_x', test_x.shape, 'test_y', test_y.shape, sep='\t')

    if not reuse_model:

        # create model
        model = keras.Sequential()

        # embed characters into dense embedded space
        model.add(keras.layers.Embedding(y_shape_ones, embedding_size, name='le', input_length=x_shape_char))
        model.add(keras.layers.Flatten()) # for dense DNN

        #model.add(keras.layers.LSTM(y_shape_ones*x_shape_char, activation='exponential', return_sequences=True, name='lr1'))
        #model.add(keras.layers.Dense(x_shape_char*10, activation='softmax', name='li'))
        #model.add(keras.layers.Dense(2000, activation='hard_sigmoid', name='lh1'))
        #model.add(keras.layers.Dense(y_shape_char*y_shape_ones, activation='softmax', name='lh2'))

        # output layer
        #model.add(keras.layers.TimeDistributed(keras.layers.Dense(y_shape_char*y_shape_ones, activation='exponential', name='lo')))
        model.add(keras.layers.Dense(y_shape_char*y_shape_ones, activation='exponential', name='lo'))
        
        # reshape to one char per output
        model.add(keras.layers.Reshape((y_shape_char, y_shape_ones)))

        # top k accuracy
        def top_k_accuracy(y_true, y_pred):
            return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=y_shape_char)

        # loss poisson mean_squared_logarithmic_error categorical_crossentropy
        # metrics 
        model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    else:
        
        # reload model from file
        model = keras.models.load_model(model_file)

    

    print(model.summary())
    model.fit(train_x, train_y, epochs=epochs)
    print(*list(zip([loss]+metrics, model.evaluate(test_x, test_y))), sep='\n', end='\n\n')



    # save model to file
    model.save(model_file)

    while True:
        
        # query
        query = input("Input (q to exit): ")
        if query == 'q': break
        query = query.split()

        # preprocess x
        x_strings = [query[0]]
        x_vector = pp.vectorize_data(x_strings)
        x_padded = pp.pad_vector_data(x_vector, pp.char_to_int['<Padding>'], width=x_org_shape[1])[x_cut]
        x_one_hot = keras.utils.to_categorical(x_padded, voc_size)
        x_one_hot_flat = x_one_hot.reshape(-1, x_one_hot.shape[1] * x_one_hot.shape[2])

        if len(query) > 1:
            # preprocess y
            y_strings = [query[1]]
            y_vector = pp.vectorize_data(y_strings)
            y_padded = pp.pad_vector_data(y_vector, pp.char_to_int['<Padding>'], width=y_org_shape[1])[y_cut]
            y_one_hot = keras.utils.to_categorical(y_padded, voc_size)
            y_one_hot_flat = y_one_hot.reshape(-1, y_one_hot.shape[1] * y_one_hot.shape[2])

        # run
        output = []
        p_one_hot = model.predict(x_padded)
        if len(query) > 1:
            output += model.evaluate(x_padded, y_one_hot)
            output += y_strings


        # decode
        #p_one_hot = p_one_hot_flat.reshape((-1, *y_shape[1:]))
        p_vector = np.argmax(p_one_hot, 2)
        p_strings = pp.decode_data(p_vector)

        # print
        output += [f"'{s}'" for s in p_strings]
        print(*output, sep='\t')



if __name__ == "__main__": main()