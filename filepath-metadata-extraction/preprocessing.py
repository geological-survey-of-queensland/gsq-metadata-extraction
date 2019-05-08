import pandas as pd
import numpy as np
import pickle
import math
import re
import keras

# tokens used to communicate non character entities
# tokens = ['<Padding>', '<Go>', '<EndOfString>', '<UnknownChar>', '<SurveyNum>', '<SurveyName>', '<LineName>', '<SurveyType>', '<PrimaryDataType>', '<SecondaryDataType>', '<TertiaryDataType>', '<Quaternary>', '<File_Range>', '<First_SP_CDP>', '<Last_SP_CDP>', '<CompletionYear>', '<TenureType>', '<Operator Name>', '<GSQBarcode>', '<EnergySource>', '<LookupDOSFilePath>', '<Source Of Data>']
tokens = ['<Padding>', '<Go>', '<EndOfString>', '<UnknownChar>']

# get set of characters to be used, use static preset list of characters
# TODO use all characters from entire training set?
available_chars = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890-_().,\\/\"':&")

# generate character to int and int to character maps
char_to_int = {c: i for i, c in enumerate(tokens + available_chars)}
int_to_char = {i: c for c, i in char_to_int.items()}
char_count = len(char_to_int) # number of character available



def main():
    """main program to perform all of preprocessing.
    reads data, vectorizes, pads it and saves it to file"""

    data = process_raw_data('SHUP 2D Files Training Data.csv')
    vectorized_data = {f: vectorize_data(data[f]) for f in data}
    padded_data = {f: pad_vector_data(vectorized_data[f], char_to_int['<Padding>']) for f in vectorized_data}

    pickle.dump((padded_data), open('training_data.p', 'wb'))



def process_raw_data(raw_source_file):
    """clean up data to remove and change some key sapects.
    
    # Arguments
        raw_source_file: the path of the raw unprocessed source data
    
    # Returns
        A dictionary with a key for each column/feature of the data and corresponding numpy matrix for the values
    """

    # read raw training data
    data = pd.read_csv(raw_source_file, dtype=str)
    print("Columns", data.columns.values)


    # 
    # modify the data as required
    # 

    # remove start of 'LookupDOSFilePath', ie:
    # '\SHUP\2D_Surveys\Processed_And_Support_Data\1980\KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy' -> 
    # '\2D_Surveys\Processed_And_Support_Data\1980\KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy'
    start_re = re.compile("^\\\\SHUP")
    data['LookupDOSFilePath'] = data['LookupDOSFilePath'].str.replace(start_re, '')

    # inster 'SurveyNum' into filepath since its foudn that way, ie:
    # '\2D_Surveys\Processed_And_Support_Data\1980\80022_KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy' -> 
    # '\2D_Surveys\Processed_And_Support_Data\1980\KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy'
    data['LookupDOSFilePath'] = data['LookupDOSFilePath'].str.slice(0, 44) + data['SurveyNum'].map(str) + '_' + data['LookupDOSFilePath'].str.slice(44)

    # drop irrelevant columns, 'FileName' 'Original_FileName'
    #data.drop('FileName', 1, inplace=True)
    #data.drop('Original_FileName', 1, inplace=True)

    # show final structure
    print('Final structure')
    print(data.head())

    # convert dataframe to to dictionary of numpy arrays
    dictionary = {feature:data[feature].values for feature in data.columns.values}

    return dictionary



def vectorize_data(data):
    """vectorize all data using global encoding dictionaries/lookup tables and store variable length vectors.
    Applys vectorize_string() on all elements on the given list.
    
    # Arguments
        data: a list of string to be vectorized

    # Returns
        a list of arrays of the vectorised data
    """

    return [vectorize_string(s) if type(s)==str else [] for s in data]



def vectorize_string(string):
    """convert a string into a vector.
    
    # Argument
        string: a string to be vectorized
    
    # Returns
        a list of integers that is the vector representation of the string
    """

    try:
        return [char_to_int[char] for char in string]
    except:
        print('error:', string, type(string))
        return []



def decode_data(data):
    """convert a list of vectors into a list of strings.
    Apply decode to each vector
    
    # Arguments
        data: list of vectors, or 2D matrix where each row is a vector

    # Returns
        a list of strings
    """

    return [decode(v) for v in data]



def decode(vector):
    """convert a vector into a string.
    
    # Arguments
        vector: a list of integers that represent a string

    # Returns
        the string that the vector represents
    """

    return ''.join([int_to_char[int(i)] for i in vector])



def load(filename):
    """load a processed pickle file.
    This function is provided to applications that use data processed by this program.
    
    # Arguments
        filename: the filename/path of the processed file to load
    
    # Returns 
        a the data object saved by this program, a dictionary that maps feature names 
        onto padded matrices containing the data
    """

    with open(filename, 'rb') as f:
        return pickle.load(f)

    return None



def train_test_split(*data, test_frac=0.2, shuffle_before=False, shuffle_after=False):
    """split dataset into train and test and optionally shuffle.
    
    # Arguments
        *data: numpy arrays to split
        test_frac: the fraction of the data to become test data
        shuffle_before: whether to shuffle the data before the split
        shuffle_after: whether to shuffle each set after the split

    # Returns
        train: the training data
        test:  the test data
    """

    # the mid point computed from test_frac
    middle = int(len(data[0]) * test_frac)

    # if applicable, shuffle the netire dataset
    if shuffle_before:
        order = np.arange(len(data[0]))         # default order of elements
        np.random.shuffle(order)                # randomise order
        data = [d[order] for d in data]         # new array with items in the randimised order

    test = [d[:middle] for d in data]
    train = [d[middle:] for d in data]

    # if applicable shuffle data after split
    if shuffle_after:
        test_order = np.arange(len(test[0]))    # (test set) default order of elements
        np.random.shuffle(test_order)           # (test set) randomise order
        test = [d[test_order] for d in test]    # (test set) new array with items in the randimised order
        train_order = np.arange(len(train[0]))  # (training set) default order of elements
        np.random.shuffle(train_order)          # (training set) randomise order
        train = [d[train_order] for d in train] # (training set) new array with items in the randimised order

    return train, test



def pad_vector_data(data, pad_token, width=None):
    """create a matrix wide enough to fit all samples and fill remaining space with the padding token.

    # Arguments
        data: list of vectors
        pad_token: token to pad with
        width: if defined, the width of the matrix and final vectors

    # Returns
        a numpy 2D matrix that contains all vectors of equal length with padding
    """

    # dimensions of the matrix
    height = len(data)
    width = width or max([len(v) for v in data])

    # empty matrix
    matrix = np.full((height, width), pad_token, np.int32)

    # insert vectors into matrix
    for i, v in enumerate(data):
        matrix[i, :len(v)] = v[:matrix.shape[1]]

    return matrix
        


def preprocess(data, x_name='LookupDOSFilePath', y_name='LineName', x_categorical=False, y_categorical=False, epoch_size=None, x_cut_s=None, x_cut_e=None, y_cut_s=None, y_cut_e=None):
    
    subset = slice(None, epoch_size) # only use subset of the dataset
    x_cut = (subset, slice(x_cut_s, x_cut_e))
    y_cut = (subset, slice(y_cut_s, y_cut_e))

    shuffle_before, shuffle_after = False, True
    test_split_frac, showcase_split_frac = 0.2, 0.005

    # spli data into x and y as well as training and test set
    (train_x, train_y), (test_x, test_y) = train_test_split(data[x_name][subset], data[y_name][subset], test_frac=test_split_frac, shuffle_before=shuffle_before, shuffle_after=shuffle_after) # split training and test
    (_, _), (showcase_x, showcase_y) = train_test_split(test_x, test_y, test_frac=showcase_split_frac, shuffle_before=True, shuffle_after=False) # extract small showcase subset of test

    print('train_x', train_x.shape, 'train_y', train_y.shape, sep='\t')

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

    # input and or output to onehot categorical encoding
    if x_categorical:
        train_x = keras.utils.to_categorical(train_x, char_count)
        test_x = keras.utils.to_categorical(test_x, char_count)
        showcase_x = keras.utils.to_categorical(showcase_x, char_count)
    if y_categorical:
        train_y = keras.utils.to_categorical(train_y, char_count)
        test_y = keras.utils.to_categorical(test_y, char_count)
        showcase_y = keras.utils.to_categorical(showcase_y, char_count)

    # store input and output shape
    x_shape = [*train_x.shape]
    x_shape[0] = None
    y_shape = [*train_y.shape]
    y_shape[0] = None

    # named shape attributes
    x_shape_char, x_shape_ones, *_ = x_shape[1:] + [None]
    y_shape_char, y_shape_ones, *_ = y_shape[1:] + [None]

    print('train_x', train_x.shape, 'train_y', train_y.shape, 'test_x', test_x.shape, 'test_y', test_y.shape, 'showcase_x', showcase_x.shape, 'showcase_y', showcase_y.shape, sep='\t')

    return train_x, train_y, test_x, test_y, showcase_x, showcase_y, x_shape_char, x_shape_ones, y_shape_char, y_shape_ones, x_name, y_name



if __name__ == "__main__": main()