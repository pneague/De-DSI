import io
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import OrderedDict

class LTRModel:
    
    def __init__(self) -> None:
        self.model = nn.Sequential(
            nn.Linear(3*768, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.last_state = deepcopy(self.model.state_dict())
        self._criterion = nn.BCELoss()
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def serialize_model(self) -> io.BytesIO:
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer

    def make_input(self, query_vector, sup_doc_vector, inf_doc_vector):
        """
        Make (query, document-pair) input for model.
        """
        return np.array([query_vector, sup_doc_vector, inf_doc_vector], dtype=np.float32).flatten()

    def train(self, pos_train_data, neg_train_data, num_epochs):
        self.last_state = deepcopy(self.model.state_dict())
        for epoch in range(num_epochs):
            losses = []
            # train positive pairs
            for data in pos_train_data:
                output = self.model(data)
                loss = self._criterion(output, torch.tensor([1.0]))
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                losses.append(loss.item())
            # train negative pairs too
            for data in neg_train_data:
                output = self.model(data)
                loss = self._criterion(output, torch.tensor([0.0]))
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                losses.append(loss.item())

            if (epoch + 1) == num_epochs:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {(sum(losses) / len(losses)):.4f}')

    # def last_update(self):
    #     updates = {}
    #     for name, param in self.model.named_parameters():
    #         updates[name] = param.data - last_state[name]
    #     return updates

    def apply_updates(self, update_model: OrderedDict):
        print('applying updates...')
        update_state = update_model
        for name, param in self.model.named_parameters():
            if name in update_state:
                param.data = (param.data + update_state[name]) / 2.0
