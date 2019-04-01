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



# main program to perform all of preprocessing
def main():
    data = process_training_data('SHUP 2D Files Training Data.csv')
    encoded_data = encode_training_data(data)

    encoded_data.to_csv('training_data.csv', index=False)
    pickle.dump((encoded_data), open('training_data.p', 'wb'))



# clean up data to remove and change some key sapects
def process_training_data(raw_source_file):

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



# encode all data using global encoding dictionaries/lookup tables
def encode_training_data(data):
    encoded_data = pd.DataFrame()

    # take each column from old data and create a new column with content where each string entry is encoded
    for column in data.columns.values:
        encoded = [encode(s) if type(s)==str else [] for s in data[column]]
        encoded_data[column] = pd.Series(encoded)
    
    print('encoded structure')
    print(encoded_data.head())

    return encoded_data



# convert a string in to a list of ints
def encode(string):
    try:
        return [char_to_int[char] for char in string]
    except:
        print('error:', string, type(string))
        return []



# convert a list if ints into a string
def decode(vector):
    ''.join(int_to_char[i] for i in vector)



# load a processed pickle file
def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

    return None


if __name__ == "__main__": main()