### Why we need mlvajra?
- There was no standard Design patterns exists for reasearching fields like Machine learning

Disclaimer :
- For past few months I have been spending my time reading different Design patterns (The take way was simple - Layer to Layer comunication should happen without depending on the underlying technology stack.
  - for example : Data layer should abstract the underlying databases)
  
  
- I am not an expert in this space still learning (for more than 2.5 years) so take this guidelines with pinch of salt and I am open to change any principles discussed here if it has some drawback.

## components of mlvajra:
- Data 
- preprocess 
- model
    - model repository
- evaluation
- visualization
    - Dashboarding
    - Monitoring 
- explanations
- schedulers
- Active learning 
    - modAL
- serving
    - tf serving
- Experiment tracking
- Deployment strategies
- Microservices
    - clipper.ai
- Documentation
    - mkdocs
- Test cases
    - doc test
    - pytest

- Bonus (for Management people)
## Data :
     In this data layer ,plugins for connecting with different data sources are placed ranging from 
     local csv,json ,parquet to Hadoop filesystems (orc,parquet,avro)

     Also reading from Different databases should be added here,ranging from sqlite ,ealsticsearch,hive ,iginite etc
     
     Also Streaming plugins will be helpful.
     for training the data from Hadoop file system, a python package called *`petastorm`* may be useful

     Above things should be written in Deployment percespective 

     For Experimenting :
        see python-intake package to get the feel of how version/maintain/abstract data
    
## preprocess :
    I have Deployed few models in production (sklearn and keras models) and preprocessing was an essential step or pivotal step in Deployment .

    How we can handle the preprocessing in production ?

    
    - so far:
    -   we are have different system/place for preprocessing i.e we treat preprocessing as a separate system, and we will speak to it or trigger it based on Rest api or schedulers.
    -  we save the intermediate preprocessed output in separate 
    databases and take the preprocessed output from the database and run prediction on that data.
    - Is this a Bad practice ? no !!

      - (some people claim this has a efficient way of deployment since we can use the preprocessed data later.)

### Take away :
-   Try to write a pipeline which consists of preprocessing ,feature Engineering and model prediction and deploy it as a single standlone service in Production .
#### pros:
    - Low Latency
    - No need of extra database to store intermediate data
    - workflow complexity will be reduced
#### cons :
-   yet to be identified

Is this Pipeline possible for all frameworks, can we serialize ,deserialize the pipeline in all framework ??
    
    mlvajra to rescue ....!

## Model

- First mandatory rule was to retrain the model on the new data to preserver model
freshness
-  Models can be built with any open source package - only catch was, the package should support pipelining with preprocessing step natively or we have to implement the our own pipeline.

### One model vs Model repository

To illustrate this further ,lets consider the case, where we trained the model and 
productionized the model, The model was doing good, as days progress the incoming data
distribuition changes and our model performance was slowly decreasing (losing freshness)

To Avoid this we planned to train the model whenever we feel the model was not performing upto the 
level. 
- we have two options We use active learning (kind of online learning) startegy to implement 
the retraining (i.e the same model was retrained with new data and serialized to the disk)

- In other hand we will train the old model with new data and compare the 
model performance with pervious(old) model and deploy the best model according to our metrics (this is because sometimes our new model's performance was slightly lower than the old model) (we will maintain model repository)

### which one is best startegy??

Thing to consider here , the new input data was hand annotated by the SME in their respective dashboards, now considering the our two cases which is the preferred startegy ?! when we retrain the model with hand annotated data if the trained model's performance was low, we will rollback to the
previous model (if you notice here we will be losing valuable hand annotated data in this case)

In future we are also planning to integerate with Active learning libraries like modAL 

so One vs Model repository has to be used carefully (without losing any data)

## Evalution 

Whenever we start with Data science project, we start with the scope of the 
Data product which is some value add given to the customer.

This scope should be effectively converted into a script which is useful to 
evaluate our model in future (we will optimize our model for this metrics)

For example :
- keeping F1 score as standard for imbalanced class data
- keeping Detection Time as a metric for Time series Anomaly detection


