from ..evaluation import classification_metrics,regression_metrics
from mlflow import (create_experiment, set_experiment, run,ActiveRun ,
                    log_metric, log_param, set_tag, log_artifacts, log_artifact,
                    active_run, start_run ,end_run, get_artifact_uri )



class mlflowFillMissingUDF(Preprocessing):
    __name__="mlflowFillMissingUDF"
    def __init__(self,config,sklearnTransformer):
        self.config=config
        self.logger=gogo.Gogo(
                        'spark_preprocesing_transformers.mlflowUDF',
                        low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(self.config['LOGS']['path'],
                                                                              self.__name__)),
                        low_formatter=formatter,
                        high_level='error',
                        high_formatter=formatter).logger
        self.sklearnTransformer=sklearnTransformer
        self.DIR=self.config[self.__name__]['DIR']#/opt/apps/coeus/coeus/model_repository
        self.index=self.config[self.__name__]['index_columns'].split(',')
        self.columns=self.config[self.__name__]['reqcolumns'].split(',')
        self.groupbycolumns=self.config[self.__name__]['groupbycolumns'].split(',')
    
    def preprocessing_during_train(self,df,
                                        columns=['BerPreFecMax',  'PhaseCorrectionAve',  'PmdMin','Qmin',  'SoPmdAve'],
                                        index= ['nodeid','section','module'],
                                        transformer=StandardScaler,
                                        path='/opt/apps/coeus/coeus/model_repository/'):
            
        scalar_dict={}
        list_df=[]
        grouped=df.groupby(index)
        #columns=['BerPreFecMax',  'PhaseCorrectionAve',  'PmdMin','Qmin',  'SoPmdAve']
        #index=['nodeid','section','module']
        for name,v in grouped.groups.items():
            scalar=transformer()
            a=scalar.fit_transform(df.iloc[v][columns])
            tmp1=pd.DataFrame(a,columns=columns)
            tmp2=pd.DataFrame(df.iloc[v][index].values,columns=index)
            list_df.append(pd.concat([tmp2,tmp1]))
            scalar_dict[name[0]+'_'+name[2]]=scalar
        pickle.dump(scalar_dict,open(path+'{}_scalar_dict.pkl'.format(self.__name__),'wb'))
        pickle.dump(columns,open(path+'{}_columns.pkl'.format(self.__name__),'wb'))
        pickle.dump(index,open(path+'{}_index.pkl'.format(self.__name__),'wb'))

        return pd.concat(list_df)


    def apply_train(self,df):
        """
        call preprocessing_during_train
        """
        artifacts = {
        "scalar_path":"{}/{}_scalar_dict.pkl".format(self.DIRself.__name__) ,
        "columns_path":"{}/{}_columns.pkl".format(self.DIR,self.__name__),
        "index_path":"{}/{}_index.pkl".format(self.DIR,self.__name__)
        }
        pdf=df.toPandas()
        df=self.preprocessing_during_train(df,self.columns,self.groupbycolumns,self.sklearnTransformer,self.DIR)
        self.mlflow_pyfunc_model_path = "{}/{}_preprocessingUDF_{}".format(self.DIR,self.__name__,datetime.now().strftime("%Y_%M_%d"))
        
        mlflow.pyfunc.save_model(
                dst_path=self.mlflow_pyfunc_model_path, python_model=preprocessingUDF(), artifacts=artifacts)
        return df

    def apply_test(self,df,op):
        """
        expected order nodeid,section,module,ts,BerPreFecMax,PhaseCorrectionAve,PmdMin,Qmin,SoPmdAve
        """
        df.createTempView('temp')
        df=op.spark.sql("select nodeid,section,module,ts,BerPreFecMax,PhaseCorrectionAve,PmdMin,Qmin,SoPmdAve from temp")
        
        pyfunc_udf = mlflow.pyfunc.spark_udf(op.spark,self.mlflow_pyfunc_model_path, result_type=ArrayType(FloatType()))
        df = df.withColumn("preprocessed_output",pyfunc_udf('nodeid','section','module','ts','BerPreFecMax',  
                                                          'PhaseCorrectionAve',  'PmdMin','Qmin',  'SoPmdAve'))
        unnested_df=df.cols.unnest("preprocessed_output")
        #pre_actual_old_columns=['nodeid','section','module','ts']+[f"preprocessed_output_{i}" for i in range(0,5)]
        nested_df=unnested_df.cols.nest([f"preprocessed_output_{i}" for i in range(0,5)],output_col='preprocessed_output',separator=',')
        #pre_actual_df=unnested_df.cols.select(pre_actual_old_columns)
        #pre_actual_new_columns=['nodeid','section','module','ts']+['BerPreFecMax','PhaseCorrectionAve',  'PmdMin','Qmin',  'SoPmdAve']
        #pre_actual_df=pre_actual_df.toDF(pre_actual_new_columns)
        df=nested_df.select(['nodeid','section','module','ts','preprocessed_output'])
        #nested_df=unnested_df.cols.nest([f"preprocessed_output_{i}" for i in range(0,5)],output_col='preprocessed_output',separator=',')
        #expected order nodeid,section,module,ts,BerPreFecMax,PhaseCorrectionAve,PmdMin,Qmin,SoPmdAve
        # The prediction column will contain all the numeric columns returned by the model as floats
        #TODO: df.createTempView('temp')
        # TODO:april_test=op.spark.sql("select nodeid,section,module,ts,BerPreFecMax,PhaseCorrectionAve,PmdMin,Qmin,SoPmdAve from temp")
        return df, #pre_actual_df


