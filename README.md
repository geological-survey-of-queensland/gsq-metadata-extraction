# Meta data extraction from filepath

## Aim
---

Aim is to utilise machine learning (ML) to generate, train and improve a model (Nerual Network) to process filepaths and extract labled metadata such as project date-time, dimensionality, projetc name and/or other meta data that may or may not be part of the path.

## Problem Statement
---

Input: varying length file path, ie "/projectname/samples/sample1/sumary 03-08.pdf"

Output: varying number of metadata tags, ie Project name, Date Completed, Type

Formal problem: sequence to sequence translation

## Approach
---

1. Unsupervised learning: learn to segment and cluster similar parts. ie find dates and years but (Very hard)
2. Create supervised learnign data (input and lables)
3. Supervised learning: associate previously learned clusters of information with real lables/metadata lables
4. Provide feedback on new data to improve model

### Inputs

input will be the filepaths, from it we want to extract labled metadata. the data may occur in different formats and locations with in the path.

Data type: String

### Outputs

Outputs may be lables with the position of the corresponding string in the path. We may require to train for it to returm the relevant data position rather than the data itself to allow for unsupervised learning to takeplace.

Example desired results for 
"\\nas02\gsq\ram\FinalSpectrumData\QDEX_Data\3D_Surveys\Processed_And_Support_Data\2010\95287_FARAWELL_3D_2012\SEGY\FARAWELL_3D_MIGRATED_DMO_STACK_SMOOTH_FXY_DECON_SDU10548TA_243006.SGY":

Subject: Survey
Year: 2012
Dimensionality: 3D
Releated to type: SEGY
Project name: Farewell

## Research
---

### Tensorflow
Tensorflow (TF) provides all the machine learning (ML) functionality and is a widely used scalable ML tool kit. Thus TF will be used for this project.

The ML process will generate and improve in an itterative process a model that can be used to predict metadata from new file path inputs. Note that the once a prediction has been made, it can be confirmed or corrected by a human and this response can be fed back into the ML process to improve the models accuracy.

### Recurrent Neural Networks (RNN)

The RNN will be able to process input of varying lengths, thus rather than feeding the entire path at once its fed character by character. The RNN will utilise Long Short Term Memory (LSTM) to process the entire path character by character and continously output information discovered at that stage.

Unlike regualr Neural Neworks (NN) which process each input seperately, RNN feeds data from the previous steps (previous characters processed) into the current computation.

### Deep Neural Network (DNN)

Likly This Network will become a Deep Neural Newtork (DNN) which means it has more than one hidden layer. Note that the NN can be both recurrent (RNN) and deep (DNN).

DNN are known for better processing of more complex data and thus is likly to increase model accuracy. Similarily to RNN DNN are more complex than regular NN.
