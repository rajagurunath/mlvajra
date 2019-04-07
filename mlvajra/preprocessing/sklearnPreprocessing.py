from collections import namedtuple
import pandas as pd
import glob
from sklearn.base import TransformerMixin,BaseEstimator

#Tabular transformers
from sklearn.preprocessing import (Binarizer,FunctionTransformer,KBinsDiscretizer,
                                    KernelCenterer,LabelBinarizer,LabelEncoder,
                                    MultiLabelBinarizer,MaxAbsScaler, MinMaxScaler,
                                    Normalizer,OneHotEncoder,OrdinalEncoder,
                                    PolynomialFeatures,PowerTransformer,QuantileTransformer,
                                    RobustScaler,StandardScaler,add_dummy_feature ,binarize,
                                    label_binarize,maxabs_scale,minmax_scale,normalize,
                                    quantile_transform,robust_scale,scale,power_transform)

# NLP transformers
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer,TfidfVectorizer,
                                             FeatureHasher,HashingVectorizer)

class ParagraphVectors(BaseEstimator,TransformerMixin):
    """
    sklearn style transformer for DOC2vec as paragraphVectors
    """
    
    
    def __init__ (self,max_epochs=100,
                  vec_size=100,
                  alpha=0.025,
                  dm=1,
                  filename='user_story',
                  **doc2vec_args):
        try :
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
            from nltk.tokenize import word_tokenize
        except ImportError:
            print("gensim or nltk was not installed")

        self.max_epochs=max_epochs
        self.vec_size=vec_size
        self.alpha=alpha
        self.dm=dm
        self.filename=filename
        self.DIRECTORY_PATH=r'tmp\\'

        
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_tagged_data(self,text_series):


        data=text_series.tolist()
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
        return tagged_data
       
    def fit(self,text_series,**doc2vec_args):
        
        """
        Distribuited memory vectors
        """
        
        if not glob.glob(self.DIRECTORY_PATH+self.filename+'*.model'):
            
        
            tagged_data=self.get_tagged_data(text_series)
            model = Doc2Vec(size=self.vec_size,
                            alpha=self.alpha, 
                            min_alpha=0.025,
                            min_count=1,
                            dm =self.dm,
                            **doc2vec_args)
              
            model.build_vocab(tagged_data)
        else:
            
            tagged_data=self.get_tagged_data(text_series)

            model= Doc2Vec.load("{}{}_d2v.model".format(self.DIRECTORY_PATH,self.filename))

            
        for epoch in range(self.max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        
        model.save("{}{}_d2v.model".format(self.DIRECTORY_PATH,self.filename))
        print("Model Saved to {}{}".format(self.DIRECTORY_PATH,self.filename))
        
   
    
    def transform(self,test_sent_list):
       
        if isinstance(test_sent_list,pd.core.series.Series):
            test_sent_list=test_sent_list.tolist()

            
        if not (isinstance(test_sent_list,list) or isinstance(test_sent_list,np.ndarray)):            
            test_sent_list=[test_sent_list]
        
        
        model= Doc2Vec.load("{}{}_d2v.model".format(self.DIRECTORY_PATH,self.filename))
        pred=[]
        for sent in test_sent_list:
            pred.append(model.infer_vector(sent))
        return np.vstack([pred])

# __all__=['Binarizer','FunctionTransformer','KBinsDiscretizer',
#             'KernelCenterer','LabelBinarizer','LabelEncoder',
#             'MultiLabelBinarizer','MaxAbsScaler', 'MinMaxScaler',
#             'Normalizer','OneHotEncoder','OrdinalEncoder',
#             'PolynomialFeatures','PowerTransformer','QuantileTransformer',
#             'RobustScaler','StandardScaler','add_dummy_feature' ,'binarize',
#             'label_binarize','maxabs_scale','minmax_scale','normalize',
#             'quantile_transform','robust_scale','scale','power_transform']


tabular_transforms=['Binarizer','FunctionTransformer','KBinsDiscretizer',
            'KernelCenterer','LabelBinarizer','LabelEncoder',
            'MultiLabelBinarizer','MaxAbsScaler', 'MinMaxScaler',
            'Normalizer','OneHotEncoder','OrdinalEncoder',
            'PolynomialFeatures','PowerTransformer','QuantileTransformer',
            'RobustScaler','StandardScaler','add_dummy_feature' ,'binarize',
            'label_binarize','maxabs_scale','minmax_scale','normalize',
            'quantile_transform','robust_scale','scale','power_transform']


nlp_transformers=['CountVectorizer','TfidfTransformer','TfidfVectorizer',
                    'FeatureHasher','HashingVectorizer','ParagraphVectors']

Tabular=namedtuple('tabular',tabular_transforms)
Tabular.Binarizer=Binarizer
Tabular.FunctionTransformer=FunctionTransformer
Tabular.KBinsDiscretizer=KBinsDiscretizer
Tabular.KernelCenterer=KernelCenterer
Tabular.LabelBinarizer=LabelBinarizer
Tabular.LabelEncoder=LabelEncoder
Tabular.MultiLabelBinarizer=MultiLabelBinarizer
Tabular.MaxAbsScaler=MaxAbsScaler
Tabular.MinMaxScaler=MinMaxScaler
Tabular.Normalizer=Normalizer
Tabular.OneHotEncoder=OneHotEncoder
Tabular.OrdinalEncoder=OrdinalEncoder
Tabular.PolynomialFeatures=PolynomialFeatures
Tabular.PowerTransformer=PowerTransformer
Tabular.QuantileTransformer=QuantileTransformer

nlp=namedtuple('nlp',nlp_transformers)
nlp.CountVectorizer=CountVectorizer
nlp.FeatureHasher=FeatureHasher
nlp.HashingVectorizer=HashingVectorizer
nlp.TfidfTransformer=TfidfTransformer
nlp.TfidfVectorizer=TfidfVectorizer
nlp.ParagraphVectors=ParagraphVectors


