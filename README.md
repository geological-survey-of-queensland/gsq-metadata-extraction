# Metadata extraction and generation

## Aim

Aim is to utilise machine learning (ML) to extract and generate metadata from the existing files by creating, training and improving a model (Nerual Network) to process currently existing files and the filestructures to extract labled metadata such as project date-time, dimensionality, projetc name and/or other metadata.

## Subprojects

- Filepath metadata extraction: extract different metadata from the file path of each file. (Current)
- Document metadata extraction: extract different metadata from coduments such as pdf reports.
- Hierarchical analysis and metadata sharing: copy metadata to related files based on hierachicl structure.

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

## Definitions

| Word / Phrase | Meaning |
|---------------|---------|
| Temporal inputs / data | data through out time (not single instance), often of varying size |
|  |  |

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