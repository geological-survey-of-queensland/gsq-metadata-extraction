import pandas as pd
import re

# tokens used to communicate non character entities
tokens = ['<Padding>', '<Go>', '<EndOfString>', '<UnknownChar>', '<SurveyNum>', '<SurveyName>', '<LineName>', '<SurveyType>', '<PrimaryDataType>', '<SecondaryDataType>', '<TertiaryDataType>', '<Quaternary>', '<File_Range>', '<First_SP_CDP>', '<Last_SP_CDP>', '<CompletionYear>', '<TenureType>', '<Operator Name>', '<GSQBarcode>', '<EnergySource>', '<LookupDOSFilePath>', '<Source Of Data>']

# get set of characters to be used, use static preset list of characters
# TODO use all characters from entire training set?
available_chars = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-_().,\\/\"':&")

# generate character to int and int to character maps
char_to_int = {c: i for i, c in enumerate(tokens + available_chars)}
int_to_char = {i: c for c, i in char_to_int.items()}

def process_training_data():

    raw_source_file = 'SHUP 2D Files Training Data.csv'
    modified_source_file = 'Processed 2D Files Training Data.csv'


    # 
    # read raw training data
    # 

    data = pd.read_csv(raw_source_file)
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

    # save and show final structure
    csv = data.to_csv(index=False).upper()
    data.to_csv(modified_source_file, index=False)
    print("Final structure")
    print(data.head())


# convert a string in to a list of ints
def encode(string):
    return [char_to_int[char] for char in string]

# convert a list if ints into a string
def decode(vector):
    ''.join(int_to_char[i] for i in vector)

# if __name__ == "__main__": process_training_data()