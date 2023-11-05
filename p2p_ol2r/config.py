from torch import nn, optim
from configparser import ConfigParser

class Config:
    def __init__(self, parser: ConfigParser):
        self.single_output = parser.getboolean('SingleOutput')
        self.loss_fn = getattr(nn, parser.get('LossFunction'))
        self.optimizer = getattr(optim, parser.get('Optimizer'))
        self.epochs = parser.getint('Epochs')
        self.lr = parser.getfloat('LearningRate')
        self.hidden_layers = parser.getint('HiddenLayers')
        self.hidden_units = parser.getint('HiddenUnits')
        self.quantize = parser.getboolean('Quantize')
        self.number_of_results = parser.getint('NumberOfResults')
        self.dropout = parser.getfloat('Dropout')
