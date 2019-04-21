import shap
import numpy as np
import pandas as pd
import enum
from typing import Callable
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler
try:
    import tensorflow as tf
    from sklearn.base import BaseEstimator
    import catboost
    import xgboost
except ImportError as e:
    print(e)
finally:
    print("Above Modules are not available -use other supported packages")


class Explain(object):
    def __init__(self,path:str,modelloader:Callable):
        self.modelloader=modelloader
    def explain(self,exptype):
        pass
    def applytransform(self,):
        pass
    def exportExplanation(self,):
        pass

             
class LimeExp(Explain):
    def __init__(self,config,df=None):
        self.config=config    
        self.sc=MinMaxScaler()
        if not isinstance(df,pd.DataFrame):
            df=pd.read_csv('../temp_data/val.csv.tar.gz')
        features=self.sc.fit_transform(df[config['columns']].dropna())    
        self.explainer=LimeTabularExplainer(features,
                feature_names=self.config['columns'], 
                class_names=['not anomaly','anomaly'], 
                discretize_continuous=True)
        

        #pickle.dump(self.explainer,open('{}/limeexp.pkl'.format(self.DIR)))
    def explain(self,array:np.array,model):
        #self.explainer=pickle.load(open('{}/limeexp.pkl'.format(self.DIR)))
        response,contrib=[],[]
        for instance in array:
            exp = self.explainer.explain_instance(instance,model.predict_proba, num_features=5,num_samples=1000)
            dict_=dict(exp.as_list())
            response.append(",".join(list(dict_.keys())))
            contrib.append(",".join(list(map(str,dict_.values()))))
        return response,contrib

class ShapExp(Explain):
    def __init__(self,config,df=None,model=None):
        self.config=config    
        if not isinstance(df,pd.DataFrame):
            df=pd.read_csv('../temp_data/val.csv.tar.gz')
        if not model:
            import tensorflow as tf
            model=tf.keras.models.load_model(config['model_path'])
        self.explainer = shap.KernelExplainer(model.predict, df.loc[:10000,config['columns']])
    def explain(self,array):
        shap_list=[]
        for instance in array:
            shap_values = self.explainer.shap_values(instance, nsamples=50)
            shap_list.append(np.mean(np.array(shap_values),axis=0))
        return [self.config['columns']]*len(shap_list),np.array(shap_list)



# if __name__=='__main__':
#     CONFIG={}
#     she=ShapExp(CONFIG)
#     #s,a=she.explain(X.iloc[:10].values)
