import torch
import random
import numpy as np
from configparser import ConfigParser
from p2p_ol2r.config import Config

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

cfgParser = ConfigParser()
cfgParser.read('config.ini')
cfg = Config(cfgParser['DEFAULT'])
