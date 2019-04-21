# sfrom ..evaluation import classification_metrics,regression_metrics
# from mlflow import (create_experiment, set_experiment, run,ActiveRun ,
#                     log_metric, log_param, set_tag, log_artifacts, log_artifact,
#                     active_run, start_run ,end_run, get_artifact_uri )

from mlvajra.evaluation import classification_metrics,regression_metrics
import os
class BaseExperiment(object):
    def __init__(self,config):
        pass
    
    def register_experiment(self,**kwargs):
        pass
    
    def wrap_train_method(self,**kwargs):
        pass
    
    def evaluate(self,**kwargs):
        pass

class clasificationExperiement(BaseExperiment):
    """
    >>> y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    >>> y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    >>> clfExp=clasificationExperiement()
    >>> cm=clfExp.warp_train_method(y_actu,y_pred)
    >>> print(cm)
    """
    def __init__(self):
        pass
    def wrap_train_method(self,y_true,y_predict,save_as_report=None):
        cm=classification_metrics.ConfusionMatrix(actual_vector=y_true,predict_vector=y_predict)    
        if save_as_report:
            cm.save_html(save_as_report+'cm.html')
            cm.save_csv(save_as_report+'cm.csv')
            cm.save_stat(save_as_report+'cm')
            cm.save_obj(save_as_report+'cm')
        return cm
    
class regressionExperiement(BaseExperiment):
    def __init__(self):
        pass
    def wrap_train_method(self,y_true,y_predict,save_as_report=None):
        experiment_dict={}
        for metric in['mean_absolute_error','mean_squared_error',
            'mean_squared_log_error','median_absolute_error','r2_score']:
            experiment_dict[metric]=regression_metrics.__dict__[metric](y_true,y_predict)
        if save_as_report:
            import json
            json.dump(experiment_dict,open(save_as_report+'reg_metrics.json','w'))
        return experiment_dict

    

if __name__=='__main__':
    y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    clfExp=clasificationExperiement()
    cm=clfExp.wrap_train_method(y_actu,y_pred)
    print(cm)

        



    