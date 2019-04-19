from ..evaluation import classification_metrics,regression_metrics
from mlflow import (create_experiment, set_experiment, run,ActiveRun ,
                    log_metric, log_param, set_tag, log_artifacts, log_artifact,
                    active_run, start_run ,end_run, get_artifact_uri )



class BaseExperiment(object):
    def __init__(self,config):
        pass
    
    def register_experiment(self,**kwargs):
        pass
    
    def warp_train_method(self,**kwargs):
        pass
    
    def evaluate(self,**kwargs):
        pass

class clasificationExperiement(BaseExperiment):
    def __init__(self):
        pass
    
class regressionExperiement(BaseExperiment):
    def __init__(self):
        pass


        



    