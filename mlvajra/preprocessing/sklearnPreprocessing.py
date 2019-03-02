from collections import namedtuple
from sklearn.preprocessing import (Binarizer,FunctionTransformer,KBinsDiscretizer,
                                    KernelCenterer,LabelBinarizer,LabelEncoder,
                                    MultiLabelBinarizer,MaxAbsScaler, MinMaxScaler,
                                    Normalizer,OneHotEncoder,OrdinalEncoder,
                                    PolynomialFeatures,PowerTransformer,QuantileTransformer,
                                    RobustScaler,StandardScaler,add_dummy_feature ,binarize,
                                    label_binarize,maxabs_scale,minmax_scale,normalize,
                                    quantile_transform,robust_scale,scale,power_transform)


__all__=['Binarizer','FunctionTransformer','KBinsDiscretizer',
            'KernelCenterer','LabelBinarizer','LabelEncoder',
            'MultiLabelBinarizer','MaxAbsScaler', 'MinMaxScaler',
            'Normalizer','OneHotEncoder','OrdinalEncoder',
            'PolynomialFeatures','PowerTransformer','QuantileTransformer',
            'RobustScaler','StandardScaler','add_dummy_feature' ,'binarize',
            'label_binarize','maxabs_scale','minmax_scale','normalize',
            'quantile_transform','robust_scale','scale','power_transform']


tabular_transforms=['Binarizer','FunctionTransformer','KBinsDiscretizer',
            'KernelCenterer','LabelBinarizer','LabelEncoder',
            'MultiLabelBinarizer','MaxAbsScaler', 'MinMaxScaler',
            'Normalizer','OneHotEncoder','OrdinalEncoder',
            'PolynomialFeatures','PowerTransformer','QuantileTransformer',
            'RobustScaler','StandardScaler','add_dummy_feature' ,'binarize',
            'label_binarize','maxabs_scale','minmax_scale','normalize',
            'quantile_transform','robust_scale','scale','power_transform']

Tabular=namedtuple('Tabular',tabular_transforms)
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



