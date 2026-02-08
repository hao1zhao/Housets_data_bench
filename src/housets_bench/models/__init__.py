from .base import BaseForecaster
from .registry import get, available, register

# register built-ins
from .naive.ar_univariate import ARUnivariateForecaster  

# statistical + ML baselines
from .ml.rf import RandomForestPCAForecaster  
from .ml.xgb import XGBPCAForecaster 

# deep learning baselines
from .dl.rnn import RNNForecaster  
from .dl.lstm import LSTMForecaster  
from .dl.dlinear import DLinearForecaster  
from .dl.timemixer import TimeMixerForecaster 
from .dl.patchtst import PatchTSTForecaster  
from .dl.informer import InformerForecaster 
from .dl.autoformer import AutoformerForecaster  
from .dl.fedformer import FEDformerForecaster 

# optional foundation-model wrappers
from .foundation.timesfm import TimesFMZeroForecaster, TimesFMCalibratedForecaster, TimesFMFullFineTuneForecaster  
from .foundation.chronos import ChronosZeroForecaster, ChronosCalibratedForecaster, ChronosFullFineTuneForecaster 
