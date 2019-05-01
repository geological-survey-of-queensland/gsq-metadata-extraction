import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import preprocessing as pp
import sys, inspect, argparse

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

parser = argparse.ArgumentParser(description='Metadata extractor ML training')
parser.add_argument('-r', '--resume',       action='store_true', default=False,        help='continue from the specified files model')
parser.add_argument('-f', '--file',         action='store',      default='model.h5',   help='file to read/write model', metavar='File')
parser.add_argument('-x',                   action='store',      default='LineName',   help='input parameter for model')
parser.add_argument('-y',                   action='store',      default='LineName',   help='output parameter for model')
parser.add_argument('-e', '--epochs',       action='store',      default=1, type=int,  help='number of epochs', metavar='Epochs')
parser.add_argument('-b', '--batch',        action='store',      default=20, type=int, help='batch size', metavar='Batch')
parser.add_argument('-a', '--architecture', action='store',      default='P-NN',       help='select architecture, view source code', metavar='Architecture')
parser.add_argument('-s', '--shuffle',      action='store',      default=[False, True], nargs=2, type=bool, help='suffle before and/or after split', metavar=('Before','After'))
parser.add_argument('-v', '--verbose',      action='store_true', default=False,        help='output debugging data')

args = parser.parse_args()

verbose = args.verbose
def log(*l, **d): 
    if verbose: print(*l, **d)
    
log(args)


data_file = 'Processed 2D Files Training Data.csv'
subset = slice(None) # only use subset of the dataset
x_cut = (subset, slice(None))
y_cut = (subset, slice(0,12))

model_file = args.file
reuse_model = args.resume

architecture = args.architecture
embedding_size = 20
lstm_hidden_size = embedding_size * 15

epochs = args.epochs
batch_size = args.batch
metrics = ['mean_absolute_error', 'categorical_accuracy', 'binary_accuracy', exact_match_accuracy]
loss = 'mean_squared_logarithmic_error' # poisson mean_squared_logarithmic_error categorical_crossentropy

x_name, y_name = args.x, args.y
shuffle_before, shuffle_after = args.shuffle

# LOAD DATA
# ------------------------------------------------------------------------------------------------------------------------------------1
def load_data():
    
    data = pp.load('training_data.p')

    # spli data into x and y as well as training and test set

    (train_x, train_y), (test_x, test_y) = pp.train_test_split(data[x_name], data[y_name], test_frac=0.2, shuffle_before=shuffle_before, shuffle_after=shuffle_after) # split training and test
    (_, _), (showcase_x, showcase_y) = pp.train_test_split(test_x, test_y, test_frac=0.005, shuffle_before=False, shuffle_after=False) # extract small showcase subset of test

    log('train_x', train_x.shape, 'train_y', train_y.shape, test_y.shape, sep='\t')

    return train_x, train_y, test_x, test_y, showcase_x, showcase_y

#def repare_data(train_x, train_y, test_x, test_y, showcase_x, showcase_y):

def main():
    """Load training data, run training and model saving process"""

    train_x, train_y, test_x, test_y, showcase_x, showcase_y = load_data()

    # PREPARE DATA
    # ------------------------------------------------------------------------------------------------------------------------------------2
    voc_size = pp.char_count

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

    # output to onehot categorical encoding
    train_y = keras.utils.to_categorical(train_y, voc_size)
    test_y = keras.utils.to_categorical(test_y, voc_size)
    showcase_y = keras.utils.to_categorical(showcase_y, voc_size)

    # store input and output shape
    x_shape = [*train_x.shape]
    x_shape[0] = None
    y_shape = [*train_y.shape]
    y_shape[0] = None

    # named shape attributes
    x_shape_char, x_shape_ones, *_ = x_shape[1:] + [None]
    y_shape_char, y_shape_ones, *_ = y_shape[1:] + [None]

    log('train_x', train_x.shape, 'train_y', train_y.shape, sep='\t')

    # DEFINE MODEL ARCHITECTURE
    # ------------------------------------------------------------------------------------------------------------------------------------3

    # create model
    model = keras.Sequential()

    # P NN (Perceptron Neural Network)
    if architecture == 'P-NN':
        model.add(keras.layers.Embedding(y_shape_ones, embedding_size, name='le', input_length=x_shape_char))   # embed characters into dense embedded space
        model.add(keras.layers.Flatten())                                                                       # flatten to 1D per sample
        model.add(keras.layers.Dense(y_shape_char*y_shape_ones, activation='exponential', name='lo'))           # dense layer
        model.add(keras.layers.Dropout(0.001))                                                                  # dropout to prevent overfitting
        model.add(keras.layers.Reshape((y_shape_char, y_shape_ones)))                                           # un flatten

    # FF NN (Feed Forward Neural Network)
    if architecture == 'FF-NN':
        hidden_size = (y_shape_ones*embedding_size + y_shape_char*y_shape_ones) // 2
        model.add(keras.layers.Embedding(y_shape_ones, embedding_size, name='le', input_length=x_shape_char))   # embed characters into dense embedded space
        model.add(keras.layers.Flatten())                                                                       # flatten to 1D per sample
        model.add(keras.layers.Dense(hidden_size, activation='exponential', name='lh'))                         # dense layer
        model.add(keras.layers.Dropout(0.2))                                                                    # dropout to prevent overfitting
        model.add(keras.layers.Dense(y_shape_char*y_shape_ones, activation='exponential', name='lo'))           # dense layer
        model.add(keras.layers.Reshape((y_shape_char, y_shape_ones)))                                           # un flatten

    # LSTM RNN (Long-Short Term Memory Recurrent Neural Network)
    if architecture == 'LSTM-RNN1':
        model.add(keras.layers.Embedding(y_shape_ones, embedding_size, name='le', input_length=x_shape_char))   # embed characters into dense embedded space
        model.add(keras.layers.Dropout(0.2))                                                                    # dropout to prevent overfitting
        model.add(keras.layers.LSTM(y_shape_char * y_shape_ones, implementation=2, unroll=True))                # lstm recurrent cell
        model.add(keras.layers.Reshape((y_shape_char, y_shape_ones)))                                           # un flatten

    # LSTM RNN (Long-Short Term Memory Recurrent Neural Network)
    if architecture == 'LSTM-RNN2':
        model.add(keras.layers.Embedding(y_shape_ones, embedding_size, name='le', input_length=x_shape_char))   # embed characters into dense embedded space
        model.add(keras.layers.Dropout(0.2))                                                                    # dropout to prevent overfitting
        model.add(keras.layers.LSTM(lstm_hidden_size, return_sequences=True, return_state=True))                # lstm recurrent cell
        model.add(keras.layers.Dense(y_shape_char * y_shape_ones))                                              # dense combine time series into single output
        model.add(keras.layers.Reshape((y_shape_char, y_shape_ones)))                                           # un flatten
    
    
    # COMPILE AND COMBINE WITH SAVED
    # ------------------------------------------------------------------------------------------------------------------------------------4
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    # if reuse is specified, the saved model is used andthe weights are applyed
    if reuse_model:
        model.load_weights(model_file)


    # RUN, EVALUATE AND SAVE
    # ------------------------------------------------------------------------------------------------------------------------------------5
    print(model.summary())
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
    model.save_weights(model_file) # save model to file


    # AUTEOMATED TESTING
    # ------------------------------------------------------------------------------------------------------------------------------------6
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
    # ------------------------------------------------------------------------------------------------------------------------------------7
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