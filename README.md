# mlVajra 
A framework or best practices to develop end to end machine learning pipeline (also has some tips for ML-management people )
Aim :
    To built robust depoyment pipeline strategies using Open source stack
    planning to add as many strategies in this repo pertaining to ML-deployment

Installation :
`pip install mlvajra` only installs mlvajra binaries

To install complete dependencies: (for time being)

git clone https://github.com/rajagurunath/mlvajra.git

```
create virtualenv

virtualenv -p python3 vajra_env

source vajra_env\bin\activate

Install all required dependencies from repo

pip install -r requirements.txt

```

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


















