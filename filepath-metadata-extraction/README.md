# Filepath metadata extraction

## Problem Statement

Extract metadata from filepaths as relevant information is often found in the file and folder names.

| | |
|-|-|
| Input | varying length file path, ie "/projectname/samples/sample1/sumary 03-08.pdf" |
| Output | varying number of metadata tags, ie Project name, Date Completed, Type |
| Formal problem | sequence to sequence translation |

## Approach

1. Unsupervised learning: learn to segment and cluster similar parts. ie find dates and years (Very hard)
2. Create supervised learnign data (input and lables)
3. Supervised learning: associate previously learned clusters of information with real lables/metadata lables
4. Provide feedback on new data to improve model

### Unsupervised learning

An autoencoder can be used to train an encoder to produce a compressed vector of the 

### Inputs

Input will be the filepaths, from it we want to extract labled metadata. the data may occur in different formats and locations with in the path. Internally teh path (string) needs to be converted into a NN compatible vector.

System input data type: String
NN input data type: Vector (int) (where each number represents a character)

### Outputs

~~The NN will return a variable length Vector (int) where each field represents a character or token with special meaning.~~

| Token name | Meaning |
|------------|---------|
| START     | start of output |
| NAME      | start of a tag name |
| TAG       | start of a tag content |
| EOT       | end of a tag content |
| EOS       | end of output |

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
Collection of relevant papers and information.

| Information | Type | URL |
|-------------|------|-----|
| Linguistic modeling example | Blog Post | https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada |
| Debugging NN | Blog Post | https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607 |
| Variational Autoencoders | Video | https://www.youtube.com/watch?v=9zKuYvjFFS8 |
|  |  |  |

### Network Structure

Below are the network structures tested ordered newest (top) to oldest (bottom). In depth analysis of the structures is found in Testing and Progress

## Testing Data

For training data a reseanoble data set has to be created. This may require a significant amount of human work. More data is likley to yield better.

At this stage no actual testing data exists and a small set will need to be created.

## Experimentation, Testing and Progress