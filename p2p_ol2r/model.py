import io
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, fuse_modules, QuantStub, DeQuantStub
from .utils import *
from .config import Config
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
                        (f'lin{i+2}', nn.Linear(self.cfg.hidden_units, 2)),
                    )

        if self.cfg.quantize:
            # TODO: Fix fused modules
            raise Exception('Quantization currently not supported')
            self.model = nn.Sequential(OrderedDict([
                ('quant', QuantStub())] + layers + [('dequant', DeQuantStub())
            ]))
            self.model.eval()
            self.model.qconfig = get_default_qat_qconfig('onednn')
            self.model = fuse_modules(self.model,
                [['lin1', 'relu1'], ['lin2', 'relu2']])
            self.model = prepare_qat(self.model.train())
        else:
            self.model = nn.Sequential(OrderedDict(layers))

        self._criterion = self.cfg.loss_fn()
        self._optimizer = self.cfg.optimizer(self.model.parameters(), lr=self.cfg.lr)

    def serialize_model(self) -> io.BytesIO:
        buffer = io.BytesIO()
        if self._quantize:
            self.model.eval()
            model = torch.quantization.convert(self.model, inplace=False)
        else:
            model = self.model
        torch.save(model, buffer)
        return buffer

    def make_input(
            self, 
            query_vector: np.ndarray, 
            sup_doc_vector: np.ndarray, 
            inf_doc_vector: np.ndarray
        ) -> np.ndarray:
        """
        Make (query, document-pair) input for model.
        """
        return np.array([query_vector, sup_doc_vector, inf_doc_vector], dtype=np.float32).flatten()

    def _train_step(self, train_data: np.ndarray, label: bool) -> float:
        """
        Performs a single training step on the given input data and label.

        Args:
            train_data (np.ndarray): The input data to train on.
            label (bool): The label associated with the input data.

        Returns:
            float: The loss value obtained during the training step.
        """
        output = self.model(torch.from_numpy(train_data))
        if self.cfg.single_output:
            label_tensor = torch.tensor([float(label)])
        else:
            label_tensor = torch.tensor([1.0, 0.0] if label else [0.0, 1.0])
        loss = self._criterion(output, label_tensor)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def train(self, pos_train_data: np.ndarray, neg_train_data: np.ndarray):
        """
        Trains the model on the given training data.

        Args:
            pos_train_data (np.ndarray): The dataset to be trained to be True.
            neg_train_data (np.ndarray): The dataset to be trained to be False.
        """
        self.model.train()
    
        print(fmt(f'Epoch [0/{self.cfg.epochs}], Loss: n/a', 'gray'), end='')
        for epoch in range(self.cfg.epochs):
            losses = [
                self._train_step(data, True) for data in pos_train_data
                ] + [
                self._train_step(data, False) for data in neg_train_data
                ]
            loss = f'{(sum(losses) / len(losses)):.4f}' if len(losses) > 0 else 'n/a'
            print(fmt(f'\rEpoch [{epoch + 1}/{self.cfg.epochs}], Loss: {loss}', 'gray'), end='')
        print()

    def infer(self, query: np.ndarray, sup_doc: np.ndarray, inf_doc: np.ndarray) -> (bool, float|tuple):
        """
        Infer the relative relevance of two documents given a query from the model.

        Args:
            query (np.ndarray): The query vector.
            sup_doc (np.ndarray): The supposedly superior document vector.
            inf_doc (np.ndarray): The supposedly inferior document vector.

        Returns:
            bool: True if the superior document is more relevant than the inferior document, False otherwise.
            float|tuple: The output of the model (either a single probability or a tuple for the probabilities to be True and False).
        """
        self.model.eval()
        _input = torch.from_numpy(self.make_input(query, sup_doc, inf_doc))
        with torch.no_grad():
            res = self.model(_input)
            if self.cfg.single_output:
                return res.item() > 0.5, res.item()
            else:
                a, b = F.softmax(res, dim=0)
                return a.item() > b.item(), (a, b)

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