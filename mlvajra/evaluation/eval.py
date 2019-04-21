import matplotlib.pyplot as plt
from pycm import ConfusionMatrix,Compare
import sklearn.metrics as metrics
from collections import namedtuple

__all__=['classification_metrics','regression_metrics']

clf_metrics=['ConfusionMatrix','Compare']
reg_metrics=['mean_absolute_error','mean_squared_error',
            'mean_squared_log_error','median_absolute_error','r2_score']

classification_metrics=namedtuple('classification_metrics',clf_metrics)
classification_metrics.ConfusionMatrix=ConfusionMatrix
classification_metrics.Compare=Compare

regression_metrics=namedtuple('regression_metrics',reg_metrics)
regression_metrics.mean_absolute_error=metrics.mean_absolute_error
regression_metrics.mean_squared_error=metrics.mean_squared_error
regression_metrics.mean_squared_log_error=metrics.mean_squared_log_error
regression_metrics.median_absolute_error=metrics.median_absolute_error
regression_metrics.r2_score=metrics.r2_score


