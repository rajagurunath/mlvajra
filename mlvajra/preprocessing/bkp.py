__doc__="""
preprocessing wrapper scripts

1. Read data from Hive :
    1.JDBC
    2. Hive Thrift server
    3. Parquet (from hdfs or local file system)
    4. CSV (from hdfs or local file system)

2 .Transforms the data
    utilities available are :
        1.All spark native preprocessing 
        2. Wrapper Class to create UDFs with custom functions
        3.Wrapper Class to create custom pipeline

3 .Write data to Hive:
    writes actual data to prestg
    writes processed data to prestg
    writes processed data to stg table

4. Alert Report:
    sending reports to mail 
    sending reports to teams group
"""
#from __future__ import print_function
__author__='Gurunath'

import os
import time
import pickle
import configparser
import sys
import matplotlib
matplotlib.use('agg')
from argparse import ArgumentParser
os.environ['SPARK_CONF_DIR']='/usr/hdp/current/spark2-client/conf'
os.environ['PYSPARK_PYTHON']='python3.6'
os.environ['SPARK_HOME']='/usr/hdp/current/spark2-client/'
import pandas as pd
from datetime import datetime
from custom_transformers import *
from read_write_utils import *
import logging
import pygogo as gogo
from datetime import datetime ,timedelta
print("="*100)
print(sys.executable)
print("="*100)
#from optimus import Optimus
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from sklearn.preprocessing import StandardScaler
sys.path.append('../evalution/')
import send_mail
try:
    import teams_alert
except ImportError:
    pass


def get_optimus(spark,app_name):
    sys.path.append('/opt/apps/coeus/ENVS/pre_env/lib/python3.6/site-packages')
    from optimus import Optimus
    op=Optimus(app_name=app_name)
    op.Spark=spark
    return op


config = configparser.ConfigParser()
config.read('preprocess_config.ini')
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
logger = gogo.Gogo(
                        'spark_preprocesing_transformers.fmt',
                        low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(config['LOGS']['path'],'preprocess_fmt_log')),
                        low_formatter=formatter,
                        high_level='error',
                        high_formatter=formatter).logger

def readData(op,filetype,config):
    """
    Reads data from Hive or CSV
    """
    try:
        if filetype=='csv':
            print('csv')
            csvU=optimusCsvUtils(config)
            hr=datetime.now().hour
            today=str(datetime.now().date())
            path=os.path.join(config['data']['DIR'],'performance_{}_hr_{}.csv'.format(today,hr))
            print(path)
            df=csvU.readData(op,path=path)
        elif filetype=='hive':
            print(HQL)
            hive=optimusHiveUtils(config)
            df=hive.readData(op,HQL)
        elif filetype=='jdbc':
            print(HQL)
            jdbc=optimusJDBC(config)
            df=jdbc.readData(op,HQL)
        elif filetype=='parquet':
            parqU=optimusParquetUtils(config)
            path=os.path.join(config['data']['DIR'],'performance_{}_hr_{}.csv'.format(today,hr))
            df=parqU.readData(op,path=path)
            df=parqU.readData(op)
        return df
    except Exception as e:
        logger.critical('Exception occured during Reading the data - {}'.format(e))
        raise Exception('Exception occured during Reading the data - {}'.format(e))

def prepare_pre_actual_for_prestg(unnested_df):
    """
    preparing preprocessed data to be pushed into prestg table 
    """
    pre_actual_old_columns=['nodeid','section','module','ts','nodename']+[f"preprocessed_output_{i}" for i in range(0,5)]
    pre_actual_df=unnested_df.cols.select(pre_actual_old_columns)
    pre_actual_new_columns=['nodeid','section','module','ts','nodename','BerPreFecMax','PhaseCorrectionAve',  'PmdMin','Qmin',  'SoPmdAve']
    pre_actual_df=pre_actual_df.toDF(*pre_actual_new_columns)
    cols=['BerPreFecMax','PhaseCorrectionAve',  'PmdMin','Qmin',  'SoPmdAve']
    index=['nodeid','section','module','ts','nodename']
    pre_actual_df=pre_actual_df.melt(index,cols,var_name='measure',value_name='val')
    return pre_actual_df



def prepare_pre_actual_for_stg(unnested_df):
    """
    preparing preprocessed data to be pushed into stg table 
    """
    nested_df=unnested_df.cols.nest([f"preprocessed_output_{i}" for i in range(0,5)],output_col='preprocessed_output',separator=',')
    df=nested_df.select(['nodeid','section','module','ts','preprocessed_output','nodename'])
    return df
    

def process_during_train(df,config):
    """
    gets stats  from the training data and serializes to the disk
    """
    try :
        
        scalar_udf=mlflowUDF(config,StandardScaler)
        df=scalar_udf.apply_train(df)
        sdf=op.spark.createDataFrame(df)
        
        return sdf
    except Exception as e:
        logger.critical('Exception occured during Training process - {}'.format(e))
        raise Exception('Exception occured during Training process - {}'.format(e))

def process_during_test(df,config):
    """
    Loads the serialized stats(models) from the disk and apply for the new data
    """
    try:
        
        scalar_udf=mlflowUDF(config,StandardScaler)
        scalar_udf.mlflow_pyfunc_model_path="/opt/apps/coeus/coeus/model_repository/preprocessingUDF" #TODO: removed next week after first training
        df=scalar_udf.apply_test(df,op,return_unnested=True)
        print(df.printSchema())
        prestg_df=prepare_pre_actual_for_prestg(df)
        stg_df=prepare_pre_actual_for_stg(df) 
        
        return prestg_df,stg_df
    except Exception as e:
        logger.critical('Exception occured during Testing process - {}'.format(e))
        raise Exception('Exception occured during Testing process - {}'.format(e))


def schedule(df,config,may_be_train='train'):
    """
    schedule train or test flow
    """
    if may_be_train=='train':
        df=process_during_train(df,config)
    elif may_be_train=='test':
        prestg_df,stg_df=process_during_test(df,config)
        return prestg_df,stg_df
        #df=process_during_test(df,config)
    return df

def writeData(df,config,op,filetype='parquet'):
    """
    Wrapper for writing utilities
    """
    try :
        if filetype=='csv':
            hr=datetime.now().hour
            today=str(datetime.now().date())
            path=os.path.join(config['data']['DIR'],'processed_performance_{}_hr_{}.csv'.format(today,hr))
            csvU=optimusCsvUtils(config)
            csvU.writeData(df,path=path)
        elif filetype=='hive':
            hiveU=optimusHiveUtils(config)
            hiveU.writeData(df,op)
        elif filetype=='parquet':
            hr=datetime.now().hour
            today=str(datetime.now().date())
            path=os.path.join(config['data']['DIR'],'processed_performance_{}_hr_{}.csv'.format(today,hr))
            parqU=optimusParquetUtils(config,)
            parqU.writeData(df,path=path)
        elif filetype=='jdbc':
            jdbc=optimusJDBC(config)
            jdbc.writeData(df)

    except Exception as e:
        logger.critical('Exception occured during Writing the data - {}'.format(e))
        raise Exception('Exception occured during Writing the data - {}'.format(e))

def main():
    appName='DNTX_PREPROCESSING'
    #op=Optimus(app_name=appName)
    spark = SparkSession \
             .builder \
             .master("local[4]") \
             .appName("Coeus spark hourly preprocessing") \
             .config("spark.sql.warehouse.dir","/apps/spark/warehouse") \
             .config("spark.executor.memory","4g") \
             .config("spark.driver.memory",'4g') \
             .enableHiveSupport() \
             .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    global op
    op=get_optimus(spark,appName)
    start=time.time()
    

    #Reading the data 
    df=readData(op,'jdbc',config)
    print(df.show())    
    if df.count()<1:
        raise Exception("empty data from hive")
    td=transformData(config) 

    ########################################### write actual data to prestg ###############################################
    jdbc=optimusJDBC(config)
    columns="""vendor,nodeid,section,module,ts,measure,val,type,algo_name,predicted_continous,predicted_categorical,score_continous,score_categorical,feedback_flg,feedback_value_categorical,feedback_value_continous,forecast_time,model_id,location_id,nodename""".split(',')
    df=df.select(['nodeid','section','module','ts','measure','val','nodename']).withColumn('type',lit('actual'))
    df=df.withColumn('vendor',lit('DTNx'))
 
    jdbc.writeData(df,"coeusapp.coeus_ml_performance_metrics_prestg",columns)#['vendor','nodeid','section','module','ts','measure','val','type'])
    
    df=td.transform(df)
    

    ##############################################Sending Missing columns report ###############################
    pdf=df.toPandas()
    hour_dict={"start_time":START_DATE,"end_time":END_DATE}
    grouped=pdf.groupby(['nodeid','module'])
    res_pdf=grouped[['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']].apply(lambda x:pd.isnull(x).sum())#.reset_index()
    res_pdf=res_pdf[res_pdf[['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']]>0.0].dropna(axis=0,how='all').reset_index().fillna('-')
    unique_nodeid=res_pdf['nodeid'].unique()
    for n in unique_nodeid:
        res_pdf1=res_pdf[res_pdf['nodeid']==n]
        res_dict=res_pdf1.to_dict(orient='records')
        teams_alert.missing_value_report(res_dict,hour_dict)

    
    if train_type=='TRAIN':
        stg_df=schedule(df,config)
        stg_df=stg_df.withColumn('vendor',lit('DTNx'))
    else:
        prestg_df,stg_df=schedule(df,config,may_be_train='test')
        stg_df=stg_df.withColumn('vendor',lit('DTNx'))
    ###################################################### sending missing value report in MSTeams #####################################
    writeData(stg_df,config,op,filetype='jdbc')
    jdbc=optimusJDBC(config)
    prestg_df=prestg_df.select(['nodeid','section','module','ts','measure','val','nodename']).withColumn('type',lit('pr_actual'))
    prestg_df=prestg_df.withColumn('vendor',lit('DTNx'))    
    jdbc.writeData(prestg_df,"coeusapp.coeus_ml_performance_metrics_prestg",columns)#['vendor','nodeid','section','module','ts','measure','val','type'])
    

    logger.info('Time taken to complete the process {} :'.format(time.time()-start))
    logger.info('All preprocessing stages are completed successfully ')


if __name__=='__main__':
    Parser = ArgumentParser(description="spark preprocessing")
    Parser.add_argument("-s", "--START_DATE", type=str, help="start date to preprocess")
    Parser.add_argument("-e", "--END_DATE", type=str, help="end date to preprocess")
    Parser.add_argument("-hql","--HQL",type=str,help="sql to fetch the data for preprocessing")
    Parser.add_argument("-type","--TRAINING_TYPE",type=str,help="TRAIN or TEST flag")
    args=Parser.parse_args()
    train_type=args.TRAINING_TYPE
    START_DATE=args.START_DATE
    END_DATE=args.END_DATE
    HQL=args.HQL
    if not HQL:
        HQL="select nodeid,section,module,ts,valid,measure,val,nodename from dnapm.performance_metrics where \
                ts>'{}' and ts<'{}' and measure in ( 'BerPreFecMax',  'PhaseCorrectionAve',  'PmdMin',  'Qmin',  'SoPmdAve' ) and \
                module in ( '13-L1-9',  '13-L1-5',  '13-L1-6',  '13-L1-10',  '13-L1-3',  '13-L1-4',  '13-L1-8',\
               '13-L1-2',  '13-L1-7',  '13-L1-1',  '10-L1-9',  '10-L1-3',  '10-L1-1',  '10-L1-5',  '10-L1-7',  '10-L1-4', \
                '10-L1-10',  '10-L1-6',  '10-L1-8',  '10-L1-2',  '11-L1-7',  '11-L1-3',  '11-L1-8',  '11-L1-9',  '11-L1-1', \
               '11-L1-6',  '11-L1-5',  '11-L1-2',  '11-L1-4',  '11-L1-10' )".format(START_DATE,END_DATE)

        #HQL="select nodeid,section,module,ts,measure,val from coeusapp.ml_performance_metrics_prestg where measure<>'null' and (to_timestamp(ts)>=to_timestamp('{}')) and (to_timestamp(ts) <to_timestamp('{}'))".format(START_DATE,END_DATE)
    if START_DATE and END_DATE:
        START_DATE=args.START_DATE
        END_DATE=args.END_DATE
        main()
        
    else:
        START_DATE=(datetime.utcnow()-timedelta(hours=2)).replace(minute=0,second=0).strftime('%Y-%m-%d %H:%M:%S')
        END_DATE=datetime.utcnow().replace(minute=0,second=0).strftime('%Y-%m-%d %H:%M:%S')
        print('DATE interval',START_DATE,END_DATE)
        main()






#from optimus import Optimus
from pyspark.sql import DataFrame
import pandas as pd
import configparser
import os
import logging
import pygogo as gogo
import sys
import types
import time
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

config = configparser.ConfigParser()
config.read('/opt/apps/coeus/preprocessing_service/preprocess_config.ini')
#op =Optimus()


__all__=['Utils','optimusCsvUtils','optimusParquetUtils','optimusHiveUtils','optimusJDBC']
class Utils(object):
        def __init__(self,config,logger):
                
                self.config=config
                self.logger =logger
        def readData(self,**kwargs):
                raise NotImplementedError
               
        def writeData(self,**kargs):
                raise NotImplementedError

class airflowJDBC(Utils):
    def __init__(self,config,logger):
        self.config =config
        self.logger=logger
    def readData(self,optimus):
        pass
    def writeData(self,**kargs):
        pass

class optimusJDBC(Utils):
    __name__='optimusJDBC'
    def __init__(self,config):
        try :
            import jaydebeapi
        except ImportError:
            raise Exception("jaydebeapi not installed")
        self.config=config
        self.logger = gogo.Gogo(
                    'readWriteUtils.fmt',
                    low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(self.config['LOGS']['path'],
                                                                          self.__name__)),
                    low_formatter=formatter,
                    high_level='error',
                    high_formatter=formatter).logger
        self.conn=jaydebeapi.connect(self.config[self.__name__]['classname'],
                    self.config[self.__name__]['jdbc_url'],
                    [self.config[self.__name__]['username'],self.config[self.__name__]['password']],
                    self.config[self.__name__]['jar_path'])
    def readData(self,optimus,sql):
        pdf=pd.read_sql(sql,self.conn)
        sdf=optimus.spark.createDataFrame(pdf)
        return sdf
    def push_to_hive(self,df,tablename="coeusapp.coeus_ml_performance_metrics_prestg",
                 columns=None, partition=None):
           
        nodename_grouped=df.groupby('nodename')
        for nodename,df in nodename_grouped:
            q = "INSERT INTO {}".format(tablename)+ " PARTITION (nodename='{}') VALUES ".format(nodename)

            q1 = ''
            #print(df.columns,df.columns[:-1])
            df=df[df.columns[:-1]]
            #print(df.columns)
            for ind, row in df.iterrows():

                tmp = str(tuple(row.values.tolist()))+','
                q1+=tmp

            q1 = q1[:-1].replace(" , ",",")
            q1 = q1.replace(", ",",")
            q1 = q1.replace("""'null'""","""null""")

            query = q+q1

            print (query[:1000])

            start = time.time()    
        #    write = HiveInterface()    
         #   write.write_to_hive_table(query)
            curr=self.conn.cursor()
            curr.execute(query)

            print ('Done in ', (time.time()-start), df.shape)

    def push_to_hive_old(self,df,tablename,columns):
        
        q1 = "INSERT INTO {} ".format(tablename) + str(tuple(columns)).replace("""'""", "") + " VALUES "

        for ind, row in df.iterrows():
            #print(q1)
            tmp = str(tuple(row.values.tolist()))+','
            q1+=tmp

        q1 = q1[:-1]
        #print(q1)
        curr=self.conn.cursor()
        curr.execute(q1)
    def writeData(self,sdf,tablenames=None,columns=None):
        pdf=sdf.toPandas()    
        if not columns:
            columns=self.config[self.__name__]['stg_columns'].split(',')
        if not tablenames:
            tablenames=self.config[self.__name__]['writetablename']
        print(tablenames,columns)
        if tablenames=='coeusapp.coeus_ml_performance_metrics_prestg': 
            df=pd.DataFrame(columns=columns)
            print(pdf.columns)
            print(columns)
            for col in pdf.columns:
                df[col]=pdf[col]
            #df[columns]=pdf
            #df['nodename']=pdf['nodename']
            #columns=columns+['nodename']
            df=df.fillna('null')
            df=df[columns]
        else:
            print('else')
            df=pdf[columns]
        for i in range(0,len(df),10000):
            print(i)
            tmp=df[i:i+10000]
            self.push_to_hive(tmp,tablenames,columns)

class optimusCsvUtils(Utils):
        __name__='optimusCsvUtils'
        def __init__(self, config):
            self.config=config
            self.logger = gogo.Gogo(
                    'readWriteUtils.fmt',
                    low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(self.config['LOGS']['path'],
                                                                          self.__name__)),
                    low_formatter=formatter,
                    high_level='error',
                    high_formatter=formatter).logger
        def readData(self,optimus,path=None):
            try:
                if not path:
                    path=self.config[self.__name__]['read_path']
                    
                df=optimus.load.csv(path=path,header=self.config[self.__name__]['read_header'],
                                                         infer_schema=self.config[self.__name__]['inferschema'])
                self.logger.info('{}-Data read from CSV -{}'.format(self.__name__,path))
                return df
            except Exception as e:
                self.logger.error('Error occured {} Reading data- {}'.format(self.__name__,e))
        def writeData(self,df,path=None):
            try:
                if not path:
                    path=self.config[self.__name__]['write_path']
                df.save.csv(path=path,
                            header=self.config[self.__name__]['write_header'],
                            mode=self.config[self.__name__]['mode'])
                self.logger.info('{}-Data written to the CSV  {} -'.format(self.__name__,path))
            except Exception as e:
                self.logger.error('Error occured in {} Writing data- {}'.format(self.__name__,e))


class optimusParquetUtils(Utils):
    __name__='optimusParquetUtils'
    def __init__(self, config):
        self.config=config
        self.logger = gogo.Gogo(
                'readWriteUtils.fmt',
                low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(self.config['LOGS']['path'],
                                                                      self.__name__)),
                low_formatter=formatter,
                high_level='error',
                high_formatter=formatter).logger
    def readData(self,optimus,path=None,**kwargs):
        try:
            if not path:
                path=self.config[self.__name__]['read_path']
            df=optimus.load.parquet(path=path,**kwargs)
            self.logger.info('{}-Data read from CSV -'.format(self.__name__))
            return df
        except Exception as e:
            self.logger.error('Error occured {} Reading data- {}'.format(self.__name__,e))
    def writeData(self,df,path=None,**kwargs):
        try:
            if not path:
                path=self.config[self.__name__]['write_path']
            df.save.parquet(path=path,mode=self.config[self.__name__]['mode'])
            self.logger.info('{}-Data written to the CSV  {} -'.format(self.__name__,self.config[self.__name__]['write_path']))
        except Exception as e:
            self.logger.error('Error occured in {} Writing data- {}'.format(self.__name__,e))


class optimusHiveUtils(Utils):
        __name__='optimusHiveUtils'
        def __init__(self, config):
            self.config=config
            self.logger = gogo.Gogo(
                    'readWriteUtils.fmt',
                    low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(self.config['LOGS']['path'],
                                                                          self.__name__)),
                    low_formatter=formatter,
                    high_level='error',
                    high_formatter=formatter).logger
                #super().__init__(config, logger)
        def readData(self,optimus,sql=None):
            try:
                if not sql:
                    sql=self.config[self.__name__]['sql']
                print(sql)
                df=optimus.Spark.sql(sql)
                self.logger.info('{}-Data read from Hive-{}'.format(self.__name__,sql))
                self.logger.info("{}".format(df.printSchema()))
                return df
            except Exception as e:
                self.logger.error('Error occured {} Reading data- {}'.format(self.__name__,e))
        def writeData(self,df,op):
            try:
                print(df.show())
                df.registerTempTable('temptable')
                #nodeid ,section,ts,module,preprocessed_output
                sql="insert into table {} select * from temptable".format(self.config[self.__name__]['writetablename'])
                print(sql)
                op.Spark.sql(sql)
                # df.write.format(self.config[self.__name__]['format']) \
                # .mode(self.config[self.__name__]['mode']) \
                # .saveAsTable(self.config[self.__name__]['writetablename'])
                self.logger.info('{}-Data written to the hive table {} '.format(self.__name__,self.config[self.__name__]['writetablename']))
            except Exception as e:
                self.logger.error('Error occured in {} Writing data- {}'.format(self.__name__,e))

