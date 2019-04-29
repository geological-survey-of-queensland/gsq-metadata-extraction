import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import preprocessing as pp
import sys, inspect

# percentage of samples that exactly match
def exact_match_accuracy(y_true, y_pred):
    argmax_true = tf.math.argmax(y_true, axis=2)            # onehot to index               (batch, width, onehot:int) -> (batch, width:int)
    argmax_pred = tf.math.argmax(y_pred, axis=2)            # onehot to index               (batch, width, onehot:int) -> (batch, width:int)
    match_char = tf.math.equal(argmax_true, argmax_pred)    # match characters              (batch, width:int) -> (batch, width:bool)
    match_word = tf.math.reduce_all(match_char, axis=1)     # require all character in sample to match      (batch, width:bool) -> (batch:bool)
    match_int = tf.cast(match_word, tf.float32)             # bool to int                                   (batch:bool) -> (batch:int)
    return tf.reduce_mean(match_int)                        # percentage of samples that are an exact match (batch:int) -> int

# top k accuracy
def top_k_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=y_shape_char)

#######################################################################################################################
# Configuration
#######################################################################################################################

# Unique Record ID	FileName	Original_FileName	SurveyNum	SurveyName	LineName	SurveyType	PrimaryDataType	SecondaryDataType	TertiaryDataType	Quaternary	File_Range	First_SP_CDP	Last_SP_CDP	CompletionYear	TenureType	Operator Name	GSQBarcode	EnergySource	LookupDOSFilePath

# file, x, y, new|reuse, epoch
defaults = ['model.h5', 'LineName', 'LineName', 'new', '1']
parameters = sys.argv[1:]
parameters += defaults[len(parameters):] # fill with defaults

data_file = 'Processed 2D Files Training Data.csv'
model_file = parameters[0]

reuse_model = parameters[3] == 'reuse'
x_cut = (slice(None), slice(None))
y_cut = (slice(None), slice(0,15))
epochs = int(parameters[4])
embedding_size = 20
metrics = ['mean_absolute_error', 'categorical_accuracy', 'binary_accuracy', exact_match_accuracy]
loss = 'mean_squared_logarithmic_error'

x_name, y_name = parameters[1:3]

def main():
    """Load training data, run training and model saving process"""
    
    # LOAD DATA
    # -------------------------------------------------------------------------------------------------------------------------------------
    voc_size = pp.char_count
    data = pp.load('training_data.p')

    # spli data into x and y as well as training and test set

    (train_x, train_y), (test_x, test_y) = pp.train_test_split(data[x_name], data[y_name], test_frac=0.2, shuffle_before=False, shuffle_after=True) # split training and test
    (_, _), (showcase_x, showcase_y) = pp.train_test_split(test_x, test_y, test_frac=0.005, shuffle_before=True, shuffle_after=True) # extract small showcase subset of test

    print('train_x', train_x.shape, 'train_y', train_y.shape, test_y.shape, sep='\t')

    # PREPARE DATA
    # -------------------------------------------------------------------------------------------------------------------------------------
    # original size
    x_org_shape = [*train_x.shape]
    x_org_shape[0] = None
    y_org_shape = [*train_y.shape]
    y_org_shape[0] = None

    # one hot encode output because the model cant do that for some reason
    train_x = train_x[x_cut]
    test_x = test_x[x_cut]
    showcase_x = showcase_x[x_cut]
    train_y = train_y[y_cut]
    test_y = test_y[y_cut]
    showcase_y = showcase_y[y_cut]

    # train_x = np.random.uniform(low=0, high=pp.char_count, size=train_x.shape).astype(int)

    #train_x = keras.utils.to_categorical(train_x, voc_size)
    #test_x = keras.utils.to_categorical(test_x, voc_size)
    #showcase_x = keras.utils.to_categorical(showcase_x, voc_size)
    train_y = keras.utils.to_categorical(train_y, voc_size)
    test_y = keras.utils.to_categorical(test_y, voc_size)
    showcase_y = keras.utils.to_categorical(showcase_y, voc_size)

    x_shape = [*train_x.shape]
    x_shape[0] = None
    y_shape = [*train_y.shape]
    y_shape[0] = None

    #[x_shape_char] = x_shape[1:]
    x_shape_char, x_shape_ones, *_ = x_shape[1:] + [None]
    y_shape_char, y_shape_ones, *_ = y_shape[1:] + [None]

    print('train_x', train_x.shape, 'train_y', train_y.shape, sep='\t')

    #train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    #test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
    #train_y = train_y.reshape(train_y.shape[0], train_y.shape[1] * train_y.shape[2])
    #test_y = test_y.reshape(test_y.shape[0], test_y.shape[1] * test_y.shape[2])

    print('train_x', train_x.shape, 'train_y', train_y.shape, sep='\t')

    # DEFINE MODEL ARCHITECTURE
    # -------------------------------------------------------------------------------------------------------------------------------------

    # create model
    model = keras.Sequential()

    # DNN
    model.add(keras.layers.Embedding(y_shape_ones, embedding_size, name='le', input_length=x_shape_char))   # embed characters into dense embedded space
    model.add(keras.layers.Flatten())                                                                       # flatten to 1D per sample
    #model.add(keras.layers.Dense(1400, activation='exponential', name='lh'))           # dense layer
    model.add(keras.layers.Dense(y_shape_char*y_shape_ones, activation='exponential', name='lo'))           # dense layer
    model.add(keras.layers.Reshape((y_shape_char, y_shape_ones)))                                           # un flatten

    
    #model.add(keras.layers.Flatten(input_shape=(x_shape_char, x_shape_ones))) # for dense DNN

    #model.add(keras.layers.LSTM(y_shape_ones*x_shape_char, activation='exponential', return_sequences=True, name='lr1'))
    #model.add(keras.layers.Dense(1000, activation='exponential', name='li', input_shape=(x_shape_char, x_shape_ones)))
    #model.add(keras.layers.Dense(2000, activation='hard_sigmoid', name='lh1'))
    #model.add(keras.layers.Dense(y_shape_char*y_shape_ones*2, activation='softmax', name='lh2'))

    # output layer
    #model.add(keras.layers.TimeDistributed(keras.layers.Dense(y_shape_char*y_shape_ones, activation='exponential', name='lo')))
    
    # reshape to one char per output

    # loss poisson mean_squared_logarithmic_error categorical_crossentropy
    # metrics 
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    # if reuse is specified, the saved model is used andthe weights are applyed
    if reuse_model:
        model.load_weights(model_file)


    # COMPILE, RUN, EVALUATE AND SAVE
    # -------------------------------------------------------------------------------------------------------------------------------------
    print(model.summary())
    model.fit(train_x, train_y, epochs=epochs)
    model.save_weights(model_file) # save model to file


    # AUTEOMATED TESTING
    # -------------------------------------------------------------------------------------------------------------------------------------
    p_one_hot = model.predict(showcase_x)
    p_vector = np.argmax(p_one_hot, 2)
    p_strings = pp.decode_data(p_vector)

    y_vector = np.argmax(showcase_y, 2)
    y_strings = pp.decode_data(y_vector)

    #x_vector = np.argmax(showcase_x, 2)
    x_strings = pp.decode_data(showcase_x)

    x_strings = [s.replace('<Padding>', '') for s in x_strings]
    y_strings = [s.replace('<Padding>', '') for s in y_strings]
    p_strings = [s.replace('<Padding>', '') for s in p_strings]
    x_w, y_w, p_w = max([len(s) for s in x_strings]), max([len(s) for s in y_strings]), max([len(s) for s in p_strings])
    y_p_strings = ['  '.join([x.ljust(x_w), y.ljust(y_w), p.ljust(p_w), str(y==p)]) for x, y, p in zip(x_strings, y_strings, p_strings)]

    print(*y_p_strings, sep='\n', end='\n\n')

    # accuracy on entire training set
    print(*list(zip([loss]+metrics, model.evaluate(test_x, test_y))), sep='\n', end='\n\n') # evaluate and list loss and each metric

    # MANUAL TESTING
    # -------------------------------------------------------------------------------------------------------------------------------------
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
        #x_one_hot_flat = x_one_hot.reshape(-1, x_one_hot.shape[1] * x_one_hot.shape[2])

        if len(query) > 1:
            # preprocess y
            y_strings = [query[1]]
            y_vector = pp.vectorize_data(y_strings)
            y_padded = pp.pad_vector_data(y_vector, pp.char_to_int['<Padding>'], width=y_org_shape[1])[y_cut]
            y_one_hot = keras.utils.to_categorical(y_padded, voc_size)
            #y_one_hot_flat = y_one_hot.reshape(-1, y_one_hot.shape[1] * y_one_hot.shape[2])

        # run
        output = []
        p_one_hot = model.predict(x_padded)
        if len(query) > 1:
            output += [format(x, '.5e') for x in model.evaluate(x_padded, y_one_hot)]
            output += y_strings


        # decode
        #p_one_hot = p_one_hot_flat.reshape((-1, *y_shape[1:]))
        p_vector = np.argmax(p_one_hot, 2)
        p_strings = pp.decode_data(p_vector)

        # print
        output += [f"'{s}'" for s in p_strings]
        print(*output, sep='\t')



if __name__ == "__main__": main()