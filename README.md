# Meta data extraction from filepath

## Aim

Aim is to utilise machine learning (ML) to generate, train and improve a model (Nerual Network) to process filepaths and extract labled metadata such as project date-time, dimensionality, projetc name and/or other meta data that may or may not be part of the path.

## Problem Statement

Input: varying length file path, ie "/projectname/samples/sample1/sumary 03-08.pdf"

Output: varying number of metadata tags, ie Project name, Date Completed, Type

Formal problem: sequence to sequence translation

## Approach

1. Unsupervised learning: learn to segment and cluster similar parts. ie find dates and years but (Very hard)
2. Create supervised learnign data (input and lables)
3. Supervised learning: associate previously learned clusters of information with real lables/metadata lables
4. Provide feedback on new data to improve model

### Inputs

input will be the filepaths, from it we want to extract labled metadata. the data may occur in different formats and locations with in the path.

Data type: String

### Outputs

#### Return relevant lable & data pairs

Example desired results for 
"\\nas02\gsq\ram\FinalSpectrumData\QDEX_Data\3D_Surveys\Processed_And_Support_Data\2010\95287_FARAWELL_3D_2012\SEGY\FARAWELL_3D_MIGRATED_DMO_STACK_SMOOTH_FXY_DECON_SDU10548TA_243006.SGY":

Subject: Survey
Year: 2012
Dimensionality: 3D
Releated to type: SEGY
Project name: Farewell

~~Not know how to retrieve data in this format.~~
~~Exact representation in NN to be determinded.~~

Use tokens to indicate start and possibly end of data of specific types.
Example output from above where [lable name] denotes the start of a labled data item:

- Start token only: "[Subject] Survey [Year] 2012 [Dimensionality] 3D [Releated to type] SEGY [Project name] Farewell" 

Or alternatively use end of data token where the token can either clsoe a specified lable or be general.

- General closing token: "[Subject] Survey [End] [Year] ..."
- Specific closing token: "[Subject] Survey [End Subject] [Year] ..."

#### Input with path a query to get result for desired data

For example input the path as well as a 'date finalised' token/query. 
Possible ways of returning the data are:

- plain text data
- confidence map indicating which characters are part of the data

This method will require a path to be tested against all lables individually that might be of interest.

## Research

### Tensorflow
Tensorflow (TF) provides all the machine learning (ML) functionality and is a widely used scalable ML tool kit. Thus TF will be used for this project.

The ML process will generate and improve in an itterative process a model that can be used to predict metadata from new file path inputs. Note that the once a prediction has been made, it can be confirmed or corrected by a human and this response can be fed back into the ML process to improve the models accuracy.

### Recurrent Neural Networks (RNN)

The RNN will be able to process input of varying lengths, thus rather than feeding the entire path at once its fed character by character. The RNN will utilise Long Short Term Memory (LSTM) to process the entire path character by character and continously output information discovered at that stage.

Unlike regualr Neural Neworks (NN) which process each input seperately, RNN feeds data from the previous steps (previous characters processed) into the current computation.

### Deep Neural Network (DNN)

Likly This Network will become a Deep Neural Newtork (DNN) which means it has more than one hidden layer. Note that the NN can be both recurrent (RNN) and deep (DNN).

DNN are known for better processing of more complex data and thus is likly to increase model accuracy. Similarily to RNN DNN are more complex than regular NN.

### In Depth Research, Papers and Webpages
Collection of relevant papers and information

#### Linguistic Modeling

| Information | Type | URL |
|-------------|------|-----|
| Linguistic modeling example | Blog Post | https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada |
| Debugging NN | Blog Post | https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607 |
|  |  |  |
|  |  |  |

#### LSTM NN

### Advantages different Neural Networks

| Type     | Advantage | Disatvantage |
|----------|-----------|--------------|
| RNN      | Tempral data                | More complex |
|          | Variable input, output size | Harder to train |
|          | Known for better accuracy   | good at predicting how the input might continue (not what we want) |
| | | |
| DNN      | Better accuracy for complex data | More complex |
|          | Better incoorporates past inputs | Slightly slower |
| | | |
| LSTM     | Variable input, output size  | Very complex |
|          | Known for very good accuracy | Slower |
| | | |
| plain NN | Very fast to train | Likly less accurate |
|          | Simpler model      | fixed input, output size |
| | | |
| convolutional NN |  | Convulution does not apply to text based data |

~~In conclusion we will initially attempt to implement and test a Deep Rrecurrent Neural Network (DRNN) and possibly compare it to RNN and DNN implementation depending on performance results.~~

### Network Structure

Below are the network structures tested ordered newest (top) to oldest (bottom). In depth analysis of the structures is found in Testing and Progress

## Testing Data

For training data a reseanoble data set has to be created. This may require a significant amount of human work. More data is likley to yield better.

At this stage no actual testing data exists and a small set will need to be created.

## Experimentation, Testing and Progress

## Abbrviations

| Abbrviation | Meaning |
|-------------|---------|
| ML | Machine learning |
| TF | Tensorflow |
| NN | Neural network |
| RNN | Recurrent Neural network |
| DNN | Deep Neural Network | 
| CNN | Convolutional Neural Network |
| DRNN | Deep Recurrent Neural Networks |
| LSTM | Long short term memory |
|  |  |
|  |  |
|  |  |
|  |  |

## Definitions

| Word / Phrase | Meaning |
|---------------|---------|
| Temporal inputs / data | data through out time (not single instance), often of varying size |
|  |  |
