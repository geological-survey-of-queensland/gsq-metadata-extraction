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
