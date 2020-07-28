# Filepath metadata extraction

## Problem Statement

Extract metadata from filepaths as relevant information is often found in the file and folder names.



## Resources and relevant files

| Resource | Location | Description |
|----------|----------|-------------|
| Data | SHUP 2D Files Training Data.csv | training data for this stage |
| Performance measurements | Performance.xlsx | accuracy over training readings |
| Source code | train.ipynb | contains all source code for this stage |
| Data sampler | sample.py | loads data and draws a random sample (for testing in earlier stages) |



## Abbreviations and Definitions

| Abbreviation | Meaning |
|-------------|---------|
| ML | Machine learning |
| TF | Tensorflow |
| NN | Neural network |
| RNN | Recurrent Neural network |
| DNN | Deep Neural Network | 
| CNN | Convolutional Neural Network |
| DRNN | Deep Recurrent Neural Networks |
| LSTM | Long short term memory |
| GRU | Gated recurent unit |

| Word / Phrase | Meaning |
|---------------|---------|
| Sequence| a sequence of objects, ie text |
| Temporal inputs / data | sequencial data, often of varying size |
| Latent | Hidden or internal |



## Problem Definition and Algorithm


### Task Definition

3 main approaches for extracting data have been proposed thus far with the option of more in the future.
1. |    File path metadata extraction
2. |    Document metadata extraction – extract data from file contents
3. |    Hierarchical metadata distribution – distribute metadata across relevant files based on their hierarchical position in the current file structure.

Stage 1 only addresses the first approach; file path metadata extraction. The goal is to develop a machine learned model that is able to extract at least some useful metadata about a file from its file path (complete file path not just the filename) and demonstrate the viability of using machine learning for this purpose. This stage does not concern itself with creating an automated process that actually processes live data and work into the new system.
The training data consists of a csv file containing about 23900 samples. The file contains the following columns: 

| Used | Column name | Data type | example |
|------|-------------|-----------|---------|
| | Unique Record ID | Numeric: positive integer | 132395 |
| Yes | FileName |    String: file name only |    GRAY_83-NJX_RAW_MIGRATED_QR007932_132395.SGY |
| No |    Original_FileName |    String: original file name |    83-NJX_QR007932_RAW_MIGRATION.SGY |
| No |    SurveyNum |    Numeric: positive integer |    84022 |
| Yes |   SurveyName |    String |    GRAY |
| Yes |   LineName |    String |    83-NJX |
| No |    SurveyType |    Categorical (1 type present) |    LAND2D |
| No |    PrimaryDataType |    Categorical (1 type present) |    PROCESSED_SEGY |
| Yes |   SecondaryDataType |    Categorical (68 types) |    RAW_MIGRATED |
| No |    TertiaryDataType |    Categorical (7 types) |    RAW_MIGRATED |
| No |    Quaternary |    Categorical (2 types) |    MIGRATED |
| No |    File_Range |    String: Numeric range |    5001-5064 |
| No |    First_SP_CDP |    String: Numeric range |    100 |
| No |    Last_SP_CDP |    String: Numeric range |    350 |
| No |    CompletionYear |    Numeric: year |    1984 |
| No |    TenureType |    Categorical (4 types present) |    EPP |
| Yes |   Operator Name |    String |    DELHI PETROLEUM PTY LTD |
| No |    GSQBarcode |    String |    QR007932 |
| No |    EnergySource |    Categorical (7 types present) |    DYNAMITE SOURCE |
| Yes |   LookupDOSFilePath |    String: new complete file path |    \SHUP\2D_Surveys\Processed_And_Support_Data\1980\GRAY\SEGY\GRAY_83-NJX_RAW_MIGRATED_QR007932_132395.SGY |
| Yes |   Source Of Data |    Categorical (2 types) |    GSQ |

Some data is not used at this stage either because it is not relevant or it usually won’t be found in the file path.
One example of relevant data that can be obtained from the file path is LineName. In this case the model will need to learn extract for example ‘80H-60’ from the file path ‘\SHUP\2D_Surveys\Processed_And_Support_Data\1980\KINNOUL_BONNIE_DOON\SEGY\KINNOUL_BONNIE_DOON_80H-60_STACK_SDU10912TA_129204.sgy’.
The type of processing is formally categorized as a sequence to sequence, meaning the input and output is a variably sized sequence of objects (characters). For this problem recurrent neural networks are best suited.


### Algorithm Definition

#### Process overview

The process of can be broken down into 3 distinct stages: pre-processing the data, embedding and the core model. 

The pre-processing is responsible for loading in the training data and processing it into a machine learning compatible format. This includes converting the textual data to vector representations, shuffling the dataset, and splitting the set into various desired subsets. In this step each character is converted into a vector and the entire dimensionality of the data set is normalized. The resulting vectors need to be of equal dimensionality regardless of sequence length. This is important as NNs are only able to train from equally sized numerical vector data.

In the next stage the training data is embedded into a denser vector space. As mentioned above the textual data is converted into a vector format on a character basis. Embedding is a form of dimensionality reduction. This process learns a transformation of the original vector representation into dense vector space. The embedding process learns both the transformation to encode the vector into the dense space as well as the transformation to decode it again. Embedding is applied twice, first to transform characters into a dense vector space and then to transforms sequences of characters or ‘words’ into even denser vector representations. After the model has processed the embedded training data its result will be decoded back into the standard vector representation on a character level using the decoder learned in this step. Embedding the data significantly increases the models accuracy and training speed.

The final stage is the core model of the processing pipeline. It is given embedded training data in embedded form, performs processing and returns a result in embedded form. It is comprised of an encoder and a decoder. Because input is a sequence, the encoder is a RNN that uses a single LSTM cell to encode the sequences or fords into an internal vector representation. The LSTM cell receives one embedded ‘word’ at a time and updates its internal state after each step. The output of the LSTM cell is a fixed sized vector after every word has been processed. A dense layer then decodes this output into the embedded word vector representation. Note that the decoder produces a fixed size vector and not a sequence, the program then simply removes training padding tokens. 

#### Data Representation

During the steps of processing the data in transformed into a number of different representations.

The raw training data is provided in the form of a csv file with a column for each of the feature of the data as described in Task Definition. During research the ‘LookupDOSFilePath’ and ‘FileName’ columns are used for inputs, the former is very similar to the real input that this model might operate on, the latter is a shorter substring of the former that is useful in testing models with shorter training times. Each field in the csv file is a variable length string or empty. 

After pre-processing the data is contained in a dictionary that maps column names onto multidimensional arrays of onehot encoded vector representations of the characters:

```
{
  ‘path’: ndarray(n_samples, n_character, n_vector_length)
  ‘p_words’ : ndarray(n_samples, n_words, n_character, n_vector_length)
  …
}
```

#### Preprocessing

Pre-processing is comprised of the following steps:.
1. | Available characters and token are defined and define char to int mappings
2. | Read raw data
3. | Read data into a dictionary mapping column names onto lists of strings
4. | For selected columns, split strings into ‘words’
5. | Vectorize strings by replacing characters with corresponding integers
6. | Read integers into bounding multidimensional array, padded with the padding token
7. | Convert integers to onehot encodings
8. | Split dataset into training and test sets
9. | Shuffle data subsets
After step 3 the data looks as follow:
{
  ‘path’: [
     ‘DIR/FILE.EXT’, 
     …
  ], 
  …
}

After step 4 some of the column’s strings are split into a list of ‘words’ at punctuation marks and symbols. The punctuations marks or symbols become words as well. This is only applied to some columns where it is reasonable.

```
‘DIR/FILE.EXT’ -> [ ‘DIR’, ‘/’, ‘FILE’, ‘.’, ‘EXT’ ]
string -> list(string)
```

After step 5 strings have been converted into lists of integers:

```
‘DIR’ -> [ 4, 9, 18 ]
String -> list(int)
```

After step 7 each integer is converted into a fixed length vector that uses the onehot encoding. The length of this vector is the number of characters and tokens available.

```
2 -> [ 0, 0, 1, …, 0, 0, 0 ]
int -> list(floats)
```

The final column’s data would be a multidimensional array of floats and have one of the following shapes: 

If not split into words:
```
(n_samples, n_character, n_vector_length)
```

If split into words:
```
(n_samples, n_words, n_character, n_vector_length)
```

The final dictionary will map column names onto multidimensional arrays:
```
{
  ‘path’: ndarray(n_samples, n_character, n_vector_length)
  ‘p_words’ : ndarray(n_samples, n_words, n_character, n_vector_length)
  …
}
```

#### Embedding

As part of the processing of file paths 2 levels of embedding take place. Firstly the onehot vectors representing a single character are encoded (embedded) into a dense vector space. Secondly sequences of dense character vectors are encoded into dense word vectors. Encoding the data into much dense vector spaces allows the NN to handle the data much better and due to the smaller data sizes training is much faster.
The encoders and decoders are creates using auto-encoders (NN). This is performed once per feature (column).
Activation functions:
The following activation functions have been tried: softplus, softsign, relu, sigmoid, hard sigmoid, exponential, linear, and none. The configurations shown in the specific models below have shown to perform best.
Loss function:
Categorical cross entropy is the chosen loss function for the training of the encoders and decoders as the outputs are categorical onehot encodings. Mean square error was also tested but performed significantly worse.
Optimizer:
The Adam and RMSProp optimizers are used.

##### Character Embedding

The NN model for character embedding is very straight forward, one dense layer to encode the data and one dense layer to decode the data:

| Purpose | Type | Output | Activation |
|---------|------|--------|------------|
| (Input) |      | Onehot character vector | |
| Encode  | characters | Dense | Dense vector | None |
| Decode  | characters | Dense | Onehot character vector | Sigmoid |

After training these layers are duplicated and used in the more complex models.
Hidden layer size (latent variable dimensionality):
For a character set of around 50 characters and tokens a hidden size of 10 is the smallest size that effectively and accurately encodes the information.

##### Word Embedding

The word embedding learns a latent vector space transformation for arrays of embedded characters. The character encoders and decoders are copied from the first character embedding model. But the character encoder and decoder layers are still trainable. Therefore this step trains a new character encoder/decoders as well as the word encoders/decoders. Training the character and word encoder together has been shown to perform significantly better than fixing the character embedding layers. Setting the character weights to the pretrained character character’s model speeds up the training of this setp.
Model summary:

| Purpose | Type | Output | Activation |
|---------|------|--------|------------|
| (Input) |  | Array of onehot character vectors |  |
| Encode characters | Dense | Array of dense character vectors | None |
| Concatenate characters | Reshape | Concatenated character vector |  |
| Encode words | Dense | Dense word vector | None |
| Decode words | Dense | Concatenated character vector | Sigmoid |
| Split into characters | Reshape | Array of dense character vectors |  |
| Decode characters | Dense | Array of onehot character vectors | Sigmoid |

Hidden layer size (latent variable dimensionality):

For a word with around 20 characters (encoded) a hidden size of 35 is the smallest size that effectively and accurately encodes the information.

##### Core/Overal Model

Overall the model performs as follow: the character encoder encodes each character individually, then characters of a word are concatenated and each word is encoded individually. The LSTM then encodes the sequence of word vectors into a vector representing the entire input. This is then decoded into the encoded words. Each word is decoded into encoded characters and each encoded character is decoded into the onehot character vector.

The weights for the character and word encoders and decoders are copied from the pre-trained word embeddings.

The core is comprised of a LSTM cell and a dense network. The LSTM cells are given the array of encoded word vectors and processes them sequentially. After each word vector it updates its internal state and once all words have been processed it outputs its hidden state. The dense layer then decodes the hidden state into an array of word vectors. The hidden size should be a few time larger than the embedded size of the output so that it can contain the information of the output as well as some information regarding the current state of processing.

The total model architecture including all levels of encoders and decoders is the following:

| Purpose | Type | Output | Activation |
|---------|------|--------|------------|
| (Input) |  | Array of onehot character vectors |  |
| Encode characters | Dense | Array of dense character vectors | None |
| Concatenate characters | Reshape | Concatenated character vector |  |
| Encode words | Dense | Dense word vector | None |
| Encode sequence | LSTM cell | Latent vector  | Sigmoid |
| Decode sequence | Dense | Dense word vector | Sigmoid |
| Decode words | Dense | Concatenated character vector | Sigmoid |
| Split into characters | Reshape | Array of dense character vectors |  |
| Decode characters | Dense | Array of onehot character vectors | Sigmoid |



## Experimental Evaluation


### Testing Methodology

The main performance measure for the complete network is a custom written function (currently names exact_match_accuracy) that computes either a 0 or 1 for each sample prediction, where 1 indicates the prediction to a sample in vectorised form (Not onehot encoding) exactly matches the ground truth. The final number returned is the percentage of predictions that are entirely correct.



## Research and useful resources
Collection of relevant papers and information.

| Information | Type | URL |
|-------------|------|-----|
| Linguistic modeling example | Blog Post | https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada |
| Debugging NN | Blog Post | https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607 |
| Variational Autoencoders | Video | https://www.youtube.com/watch?v=9zKuYvjFFS8 |
| Types of NN | Blog Post | https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464 |
| Dropuout | Blog Post | https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5 |
| RNN Improvements | Blog Post | https://danijar.com/tips-for-training-recurrent-neural-networks/ |
|  |  |  |
