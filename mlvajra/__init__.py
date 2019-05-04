"""A Last weapon to save Data scientist"""
__version__='0.1.4.3'
__name__='mlvajra'
import mlvajra 
from mlvajra import preprocessing
from mlvajra import data
from mlvajra import evaluation
from mlvajra import explanations
from mlvajra import model
from mlvajra import visualization
from mlvajra import Annotation
from mlvajra import activelearning
from mlvajra import deploy
from mlvajra import experiment
__all__=['mlvajra','data','preprocessing','evaluation','explanations','model',
            'visualization','Annotation','experiment','activelearning']
