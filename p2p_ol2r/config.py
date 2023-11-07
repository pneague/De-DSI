from torch import nn, optim
from configparser import ConfigParser

class Config:
    def __init__(self, parser: ConfigParser):
        self.single_output = parser.getboolean('SingleOutput')
        self.loss_fn = getattr(nn, parser.get('LossFunction'))
        self.optimizer = getattr(optim, parser.get('Optimizer'))
        self.epoch_scale = parser.getint('EpochScale')
        self.lr = parser.getfloat('LearningRate')
        self.hidden_layers = parser.getint('HiddenLayers')
        self.hidden_units = parser.getint('HiddenUnits')
        self.dropout = parser.getfloat('Dropout')
        self.quantize = parser.getboolean('Quantize')
        self.number_of_results = parser.getint('NumberOfResults')

    def __str__(self) -> str:
        return (f"Config(SingleOutput={self.single_output}, "
                f"LossFunction={self.loss_fn.__name__ if self.loss_fn else 'None'}, "
                f"Optimizer={self.optimizer.__name__ if self.optimizer else 'None'}, "
                f"EpochScale={self.epoch_scale}, "
                f"LearningRate={self.lr}, "
                f"HiddenLayers={self.hidden_layers}, "
                f"HiddenUnits={self.hidden_units}, "
                f"Dropout={self.dropout}, "
                f"Quantize={self.quantize}, "
                f"NumberOfResults={self.number_of_results})")