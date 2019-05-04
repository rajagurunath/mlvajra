# mlVajra 
A framework or best practices to develop end to end machine learning pipeline ,use it as a standlone library or use some strategies listed here to use in your usecases.

## Aim :
    To built robust depoyment pipeline strategies using Open source stack
    planning to add as many strategies in this repo pertaining to ML-deployment

## Installation :

`pip install mlvajra` - only installs mlvajra binaries

To install complete dependencies: (for time being)

git clone https://github.com/rajagurunath/mlvajra.git

```
create virtualenv

virtualenv -p python3 vajra_env

source vajra_env\bin\activate

cd mlvajra
Install all required dependencies from repo

pip install -r requirements.txt

```
## Goal of mlvajra:
- Easily set your infrastructure /deployment in client machines
- No need to write glue code everytime to deploy ML models in production (Lets have a unifying Deployment strategy)  glue codes so far -Archana ,coeus,century-link
- start your Experiment from the day one (over ambitious here)
    should have a common format for every model (framework agnostic) to deploy and devops cycle.
    your path to Research to production should as easiest & earliest as possible


## Components of mlvajra 

### data
   Read /write utilities for both batch and streaming data
   sqlalchemy

### Vajron:
Packaging your model (from different frameworks) to unified format and deploy that in
production as single entity with embeded Preprocessing functions 

### Deployment strategies:
- Docker deployment using clipper.ai (as microservices) -framework agnostic  
- Psedo-edge deployment using MLflow -framework agnostic   
- Deploying transparent\managable end to end Model pipeline (using TFX) -For Tensorflow models only --> google strategy
- Jupyter Notebook as main Deployment strategy -Analytics Projects \Excellent visualization -->Netflix

### Explanations:
    To have a Explanation dashboard (model agnostic frameworks)
    which should explain your model both in training and inference time


### Ground-Truth Generation:
    Using Annotation tool for timeseries and NLP data
    Using Active learning to get correct\refined examples to train the model
    Using Data programming concept to prepare ground truth

### Experiment Tracking:
- Experiment trackers --> track yours experiments 
- Easily share /communicate your code/results with fellow teammates
- Track the Experiment for the project in a unified manner (all data scientists can use this Tracker to submit their scores) From the Best Experiment -the model should be easily deployable in production (in Different Deployment startegies)

### Quick Experiments:
    using this tools we can quickly prepare our Experiments and test different models/configurations and
    productionize our models
- ### Tools:
    - Neural Network Intelligence
    - Ludwig (exclusively for Tensorflow )-making Tensorflow Tensorflow as easy as possible

`
Hidden Technical Debt of Machine learning systems  : (this paper discusses about the problem faced during ML deployment -only theory)
`

![hidden_technical_dept](IMG/typical_ML_workflow.gif)

## TODO list
### Deploy
- Mlflow
- Tensorflow serving

### model-Training /distribuited
- mlflow -generic classification metrics (done)
- nnictl-automl -tensorflow /pytorch 

### Feature Engineering
- pandas
- pyspark-Flint

### preprocessing
- cyclic features (done)
- lag features
- window features


















