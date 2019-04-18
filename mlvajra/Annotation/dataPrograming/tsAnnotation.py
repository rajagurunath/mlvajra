"""
TODO:
Authoring Different Labeling functions to annotate timeseries 
data
For example:

3-standarddeviation
===============================================================
Comparing different Labeling functions /merging different 
Labeling functions to achieve better classification accuracy
Example:
    helper functions to visualize Distribuitions of different 
    generated Label functions

===============================================================

Using Active learning to collect Data Labels

(can be used for both Regression /classifications)

"""
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import numpy as np
annotation_config={}

def stdAnnotation(df,features,alpha=3,label_column_name='std_label'):
    df[label_column_name]=np.zeros(df.shape[0],dtype='int')
    means=df[features].mean()
    for col,mean in zip(means.index,means.values):
        print(col,mean)
        df[label_column_name][(df[col]-mean-alpha*df[col].std())>0]=1
    print('Class Count:',df[label_column_name].value_counts())
    return df

def naAnnotation():
    pass

def kmeansAnnotation():
    pass

def isolationForestAnnotation():
    pass

def EllipticEnvelopeAnnotation():
    pass
def activeLearningAnnotation():
    pass













