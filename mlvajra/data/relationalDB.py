
from sqlalchemy.engine import create_engine
from sqlalchemy import inspect
from sqlalchemy import *
from sqlalchemy.schema import *
import pandas as pd


class sqlDB(object):
    def __init__(self,uri='sqlite:///default.db',**kargs):
        self.engine=create_engine(uri,**kargs)
    def readData(self,sql):
        df=pd.read_sql(sql,self.engine)
        return df
    def writeData(self,df,tablename=None):
        df.to_sql(tablename,self.engine)






















