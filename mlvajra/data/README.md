## Reading and writing data utilities for different data stores (DBs) and data sources (streaming jobs)
Adding helper functions to read and write from different data sources

# adding new Data stores
Class Definition consists of simple api pattern and can be extended if needed (suggestions welcomed)

```
class Utils(object):
        def __init__(self,config:configparser.ConfigParser)->None:
                self.config=config
        def readData(self,optimus:Any,path:str=None,**kargs)->DataFrame:
                raise NotImplementedError
               
        def writeData(self,df:DataFrame,path:str=None,**kargs)->None:
                raise NotImplementedError


```
inhereting this utility class and writing your DB specific writing /reading data functions 

TODO: Adding support to read /write from data streams (spark/kafka)


