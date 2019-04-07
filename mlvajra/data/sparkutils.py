try:
    from optimus import Optimus
    from pyspark.sql import DataFrame,SparkSession
    import configparser
    import os
    import logging
    import pygogo as gogo
    import sys
    import types
    from typing import Any
    from functools import reduce
except ImportError as e:
    print("Some packages are not installed ",e)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

config = configparser.ConfigParser()
config.read('preprocess_config.ini')
op =Optimus()


__all__=['Utils','optimusCsvUtils','optimusParquetUtils','HiveUtils']
class Utils(object):
        def __init__(self,config:configparser.ConfigParser)->None:
                self.config=config
        def readData(self,optimus:Any,path:str=None,**kargs)->DataFrame:
                raise NotImplementedError
               
        def writeData(self,df:DataFrame,path:str=None,**kargs)->None:
                raise NotImplementedError
    
class optimusCsvUtils(Utils):
        __name__='optimusCsvUtils'
        def __init__(self, config:configparser.ConfigParser)->None:
            self.config=config
            self.logger = gogo.Gogo(
                    'readWriteUtils.fmt',
                    low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(self.config['LOGS']['path'],
                                                                          self.__name__)),
                    low_formatter=formatter,
                    high_level='error',
                    high_formatter=formatter).logger
                #super().__init__(config, logger)
        def readData(self,optimus:Optimus,path:str=None,**kargs)->DataFrame:
            try:
                if not path:
                    path=self.config[self.__name__]['read_path']
                    
                df=optimus.load.csv(path=path,header=self.config[self.__name__]['read_header'],
                                                         infer_schema=self.config[self.__name__]['inferschema'])
                self.logger.info('{}-Data read from CSV -{}'.format(self.__name__,path))
                return df
            except Exception as e:
                self.logger.error('Error occured {} Reading data- {}'.format(self.__name__,e))
        def writeData(self,df:DataFrame,path:str=None,**kargs)->None:
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
    def __init__(self, config:configparser.ConfigParser)->None:
        self.config=config
        self.logger = gogo.Gogo(
                'readWriteUtils.fmt',
                low_hdlr=gogo.handlers.file_hdlr('{}/{}.log'.format(self.config['LOGS']['path'],
                                                                      self.__name__)),
                low_formatter=formatter,
                high_level='error',
                high_formatter=formatter).logger
    def readData(self,optimus:Optimus,path:str=None,**kwargs)->DataFrame:
        try:
            if not path:
                path=self.config[self.__name__]['read_path']
            df=optimus.load.parquet(path=path,**kwargs)
            self.logger.info('{}-Data read from CSV -'.format(self.__name__))
            return df
        except Exception as e:
            self.logger.error('Error occured {} Reading data- {}'.format(self.__name__,e))
    def writeData(self,df:DataFrame,path:str=None,**kwargs)->None:
        try:
            if not path:
                path=self.config[self.__name__]['write_path']
            df.save.parquet(path=path,mode=self.config[self.__name__]['mode'])
            self.logger.info('{}-Data written to the CSV  {} -'.format(self.__name__,self.config[self.__name__]['write_path']))
        except Exception as e:
            self.logger.error('Error occured in {} Writing data- {}'.format(self.__name__,e))


class HiveUtils(Utils):
        __name__='HiveUtils'
        def __init__(self, config:configparser.ConfigParser)->None:
                self.config=config
                self.logger = gogo.Gogo(
                        'readWriteUtils.fmt',
                        low_hdlr=gogo.handlers.file_hdlr('/opt/apps/DTNX/logs/{}.log'.format(self.__name__)),
                        low_formatter=formatter,
                        high_level='error',
                        high_formatter=formatter).logger
        def readData(self,sparkSession:SparkSession,**kwargs)->DataFrame:
                try:
                        df=sparkSession.sql(self.config[self.__name__]['read_sql'],**kwargs)
                        self.logger.info('{}-Data read from Hive -'.format(self.__name__))
                        return df
                        
                except Exception as e:
                        self.logger.error('Error occured {} Reading data- {}'.format(self.__name__,e))
        def writeData(self,df:DataFrame,**kargs)->None:
                try:
                        columns=[df[col] for col in self.config[self.__name__]['columns_to_write']]
                        df=df.select(*columns,df[self.config[self.__name__]['vector_column']].cast("string"))
                        df.write.saveAsTable(self.config[self.__name__]['table_name'])
                        self.logger.info('{}-Data written to the Hive table {} -'.format(self.__name__,self.config[self.__name__]['table_name']))
                except Exception as e:
                        self.logger.error('Error occured in {} Writing data- {}'.format(self.__name__,e))

def unionAll(*dfs):
        return reduce(DataFrame.unionAll, dfs)



