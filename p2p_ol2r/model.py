import io
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, fuse_modules, QuantStub, DeQuantStub
from random import shuffle
from .utils import *
from .config import Config

class ModelInput(torch.Tensor):
    """
    A tensor representing the input to the model (the query-doc-doc triplet).
    """
    @staticmethod
    def __new__(cls, query_vector: np.ndarray, sup_doc_vector: np.ndarray, inf_doc_vector: np.ndarray):
        input_array = np.concatenate([query_vector, sup_doc_vector, inf_doc_vector])
        tensor = torch.as_tensor(input_array, dtype=torch.float32)
        obj = torch.Tensor._make_subclass(cls, tensor)
        obj.query = query_vector
        obj.sup_doc = sup_doc_vector
        obj.inf_doc = inf_doc_vector
        return obj
    
    def inverse(self):
        """
        Returns the inverse of the model input (i.e. the same input with the superior and inferior document swapped).
        """
        return ModelInput(self.query, self.inf_doc, self.sup_doc)

# Alias for a labeled model input to be used for training
LabeledModelInput = tuple[ModelInput, bool]

class LTRModel:
    
    def __init__(self, config: Config) -> None:
        self.cfg = config

        layers = [
            ('fc1', nn.Linear(3*768, self.cfg.hidden_units)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(p=self.cfg.dropout)),
        ]

        # adding layers from config
        for i in range(self.cfg.hidden_layers):
            if i < self.cfg.hidden_layers - 1:
                layers.extend([
                    (f'fc{i+2}', nn.Linear(self.cfg.hidden_units, self.cfg.hidden_units)),
                    (f'relu{i+2}', nn.ReLU()),
                    (f'drop{i+2}', nn.Dropout(p=self.cfg.dropout)),
                ])
            else:
                # last layer before output
                if self.cfg.single_output:
                    layers.extend([
                        (f'fc{i+2}', nn.Linear(self.cfg.hidden_units, 1)),
                        (f'sigmoid', nn.Sigmoid()),
                    ])
                else:
                    layers.append(
                        (f'fc{i+2}', nn.Linear(self.cfg.hidden_units, 2)),
                    )

        if self.cfg.quantize:
            torch.backends.quantized.engine = 'qnnpack'
            self.model = nn.Sequential(OrderedDict([
                ('quant', QuantStub())] + layers + [('dequant', DeQuantStub())
            ]))
            self.model.eval()
            self.model.qconfig = get_default_qat_qconfig('qnnpack')

            modules = [[f'fc{i}', f'relu{i}'] for i in range(1, self.cfg.hidden_layers + 1)]
            self.model = fuse_modules(self.model, modules)
            self.model = prepare_qat(self.model.train())
        else:
            self.model = nn.Sequential(OrderedDict(layers))

        self._criterion = self.cfg.loss_fn()
        self._optimizer = self.cfg.optimizer(self.model.parameters(), lr=self.cfg.lr)

    def serialize_model(self) -> io.BytesIO:
        buffer = io.BytesIO()
        if self.cfg.quantize:
            self.model.eval()
            model = torch.quantization.convert(self.model, inplace=False)
        else:
            model = self.model
        torch.save(model, buffer)
        return buffer

    def _train_step(self, model_input: ModelInput, label: bool) -> float:
        """
        Performs a single training step on the given input data and label.

        Args:
            model_input: The input data to train on.
            label: The label associated with the input data.

        Returns:
            float: The loss value obtained during the training step.
        """
        self.model.eval()
        output = self.model(model_input)
        self.model.train()
        if self.cfg.single_output:
            label_tensor = torch.tensor([float(label)])
        else:
            label_tensor = torch.tensor([1.0, 0.0] if label else [0.0, 1.0])
        loss = self._criterion(output, label_tensor)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def train(self, true_train_data: list[ModelInput], epochs: int = 1):
        """
        Trains the model on the given training data and its inverse (if i>j is true then j>i must be false).

        Args:
            true_train_data: The training data to train on to be classified as true.
            epochs: The number of epochs to train each training data item on.
        """
        shuffle(true_train_data)
        epochs *= self.cfg.epoch_scale
        print(fmt(f'Epoch [0/{epochs}], Loss: n/a', 'gray'), end='')
        for epoch in range(epochs):
            losses = []
            for mi in true_train_data:
                losses.append(self._train_step(mi, True))
                losses.append(self._train_step(mi.inverse(), False))
            loss = f'{(sum(losses) / len(losses)):.4f}' if len(losses) > 0 else 'n/a'
            print(fmt(f'\rEpoch [{epoch + 1}/{epochs}], Loss: {loss}', 'gray'), end='')
        print()

    def infer(self, model_input: ModelInput) -> (bool, float|tuple):
        """
        Infer the relative relevance of two documents given a query from the model.

        Args:
            query: The query vector.
            sup_doc: The supposedly superior document vector.
            inf_doc: The supposedly inferior document vector.

        Returns:
            bool: True if the superior document is more relevant than the inferior document, False otherwise.
            float|tuple: The output of the model (either a single probability or a tuple for the probabilities to be True and False).
        """
        self.model.eval()
        with torch.no_grad():
            res = self.model(model_input)
            if self.cfg.single_output:
                return res.item() > 0.5, res.item()
            else:
                a, b = F.softmax(res, dim=0)
                return a.item() > b.item(), (a.item(), b.item())

    def apply_updates(self, update_model):
        if self.cfg.quantize:
            updates = {
                name: module for name, module in update_model.named_modules() if isinstance(module, nn.quantized.Linear)
            }
            for name, param in self.model.named_parameters():
                layer, attr = name.split('.')
                if attr == 'weight':
                    data = updates[layer].weight()
                elif attr == 'bias':
                    data = updates[layer].bias()
                param.data = (param.data + torch.dequantize(data).data) / 2.0
        else:
            update_model_state = update_model.state_dict()
            for name, param in self.model.named_parameters():
                if name in update_model_state:
                    param.data = (param.data + update_model_state[name]) / 2.0
        print(fmt('Model updated', 'gray'))