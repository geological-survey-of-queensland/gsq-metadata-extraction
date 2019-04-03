import pandas as pd
import numpy as np
import pickle
import math
import re

# tokens used to communicate non character entities
tokens = ['<Padding>', '<Go>', '<EndOfString>', '<UnknownChar>', '<SurveyNum>', '<SurveyName>', '<LineName>', '<SurveyType>', '<PrimaryDataType>', '<SecondaryDataType>', '<TertiaryDataType>', '<Quaternary>', '<File_Range>', '<First_SP_CDP>', '<Last_SP_CDP>', '<CompletionYear>', '<TenureType>', '<Operator Name>', '<GSQBarcode>', '<EnergySource>', '<LookupDOSFilePath>', '<Source Of Data>']

# get set of characters to be used, use static preset list of characters
# TODO use all characters from entire training set?
available_chars = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890-_().,\\/\"':&")

# generate character to int and int to character maps
char_to_int = {c: i for i, c in enumerate(tokens + available_chars)}
int_to_char = {i: c for c, i in char_to_int.items()}
char_count = len(char_to_int)



def main():
    """main program to perform all of preprocessing"""

    data = process_training_data('SHUP 2D Files Training Data.csv')
    vectorized_data = vectorize_training_data(data)
    padded_data = pad_vector_data(vectorized_data, char_to_int['<Padding>'])

    pickle.dump((padded_data), open('training_data.p', 'wb'))



def process_training_data(raw_source_file):
    """clean up data to remove and change some key sapects"""

    # read raw training data
    data = pd.read_csv(raw_source_file, dtype=str)
    print("Columns", data.columns.values)


    # 
    # modify the data as required
    # 

    # remove start of 'LookupDOSFilePath'
    # '\SHUP\2D_Surveys\Processed_And_Support_Data\1980\KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy' -> 
    # '\2D_Surveys\Processed_And_Support_Data\1980\KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy' -> 
    start_re = re.compile("^\\\\SHUP")
    data['LookupDOSFilePath'] = data['LookupDOSFilePath'].str.replace(start_re, '')

    # inster 'SurveyNum' into filepath since its foudn that way
    # '\2D_Surveys\Processed_And_Support_Data\1980\80022_KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy' -> 
    # '\2D_Surveys\Processed_And_Support_Data\1980\KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy'
    data['LookupDOSFilePath'] = data['LookupDOSFilePath'].str.slice(0, 44) + data['SurveyNum'].map(str) + '_' + data['LookupDOSFilePath'].str.slice(44)

    # drop irrelevant columns, 'FileName' 'Original_FileName'
    data.drop('FileName', 1, inplace=True)
    data.drop('Original_FileName', 1, inplace=True)

    # show final structure
    print('Final structure')
    print(data.head())

    return data



def vectorize_training_data(data):
    """vectorize all data using global encoding dictionaries/lookup tables and store variable length vectors"""

    vectorized_data = {}

    # take each column from old data and create a new column with content where each string entry is encoded
    for feature in data.columns.values:
        vectorized_data[feature] = [vectorize_string(s) if type(s)==str else [] for s in data[feature]]

    return vectorized_data



def vectorize_string(string):
    """convert a string in to a vector"""

    try:
        return [char_to_int[char] for char in string]
    except:
        print('error:', string, type(string))
        return []



def decode(vector):
    """convert a vector into a string"""

    ''.join(int_to_char[i] for i in vector)



def load(filename):
    """load a processed pickle file"""

    with open(filename, 'rb') as f:
        return pickle.load(f)

    return None



def train_test_split(*data, test_frac=0.2, shuffle_before=False, shuffle_after=False):
    """split dataset into train and test and optionally shuffle"""

    middle = int(len(data[0]) * test_frac)

    if shuffle_before:
        order = np.arange(len(data[0]))
        np.random.shuffle(order)
        data = [d[order] for d in data]

    test = [d[:middle] for d in data]
    train = [d[middle:] for d in data]

    if shuffle_after:
        test_order = np.arange(len(test[0]))
        np.random.shuffle(test_order)
        test = [d[test_order] for d in test]
        train_order = np.arange(len(train[0]))
        np.random.shuffle(train_order)
        train = [d[train_order] for d in train]

    return train, test
        

def pad_vector_data(data, pad_token):
    """for each feature create a matrix wide enough to fit all samples and fill with the padding token"""

    padded_data = {}

    for feature in data:
        vector_list = data[feature]

        height = len(vector_list)
        width = max([len(v) for v in vector_list])

        matrix = np.full((height, width), pad_token, np.int32)

        for i, v in enumerate(vector_list):
            matrix[i, :len(v)] = v

        padded_data[feature] = matrix

    return padded_data
        


if __name__ == "__main__": main()