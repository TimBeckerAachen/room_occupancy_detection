# room_occupancy_detection
The work presented in this repository is motivated by the paper [Accurate occupancy detection of an office room from light, 
temperature, humidity and CO2 measurements using statistical learning models](https://www.sciencedirect.com/science/article/abs/pii/S0378778815304357).

## Task
The idea is to predict the occupancy of an office room based on measurements of light, temperature, humidity and 
CO2. Subsequently, the predicted occupancy is used to control the HVAC system of the room and to minimize costs. 

## Dataset
The dataset can be found no [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/357/occupancy+detection).
It is a relatively small dataset with minute frequency data for a few days in February in a single office room. The 
dataset contains the following columns:
- Date (timestamp)
- Temperature (Celsius)
- Humidity (%)
- Light (Lux)
- CO2 (ppm)
- Humidity Ratio (kgwater-vapor/kg-air)
- Occupancy (0 for not occupied, 1 for occupied status)
It is split into 3 files which can be found in the [dataset](dataset) folder.

## Approach
In this repository, we will use a simple machine learning models to predict the occupancy of the room. We start with
a general data exploration and visualization of the data. Afterward, we will try several basic machine learning models,
a logistic regression, a decision tree, a random forest and a gradient boosting model. To tune the models, we will use
parameter search and cross validation. It is important to not use shuffling during cross validation due to the time 
series nature of the data. The models will be evaluated based on their accuracy, precision, recall and F1 score.

## Notebooks


## Presentation
A summary of the results can be found as presentation in pdf format: [presentation](presentation/room_occupancy_prediction.pdf).
It includes the most important results with reflections and thoughts and provides some ideas concerning the
deployment of the model in practice.

## Code structure

## Remarks
- The dataset is very small. The final scores can vary significantly on the data depending on what data is used for 
training and testing.

## Conclusion
While this tasks serves as a nice educational example, in practice a simple motion sensor might be a better solution. 
Moreover, it might not be technically feasible to control the HVAC system based on a highly dynamic occupancy prediction.

### TODO
- clean exploration notebook
- add & clean all model notebooks

- update requirements.txt
- explain how to run the code

- explain files and notebooks
- explain decisions briefly
- conclusion in the readme

- github page
- add images to markdown

### done
- explain the task and dataset
- link to the paper
- add data
- add presentation
- clean code + docstrings (no comments)