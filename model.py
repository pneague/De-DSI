import io
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.ao.quantization import get_default_qat_qconfig, default_observer, default_weight_observer, prepare_qat, fuse_modules, QuantStub, DeQuantStub

class LTRModel:
    
    def __init__(self) -> None:
        model_fp32 = nn.Sequential(OrderedDict([
            ('quant', QuantStub()),
            ('lin1', nn.Linear(3*768, 256)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(256, 256)),
            ('relu2', nn.ReLU()),
            ('lin3', nn.Linear(256, 256)),
            ('relu3', nn.ReLU()),
            ('lin4', nn.Linear(256, 1)),
            ('sigmoid', nn.Sigmoid()),
            ('dequant', DeQuantStub())
        ]))

        # model must be set to eval for fusion to work
        model_fp32.eval()

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'x86' for server inference and 'qnnpack'
        # for mobile inference. Other quantization configurations such as selecting
        # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
        # can be specified here.
        # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
        # for server inference.
        # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        qat_qconfig = get_default_qat_qconfig('x86')
        model_fp32.qconfig = qat_qconfig

        # fuse the activations to preceding layers, where applicable
        # this needs to be done manually depending on the model architecture
        model_fp32_fused = fuse_modules(model_fp32,
            [['lin1', 'relu1'], ['lin2', 'relu2'], ['lin3', 'relu3']])

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model needs to be set to train for QAT logic to work
        # the model that will observe weight and activation tensors during calibration.
        self.model = prepare_qat(model_fp32_fused.train())

        #######
        #self.model = model_fp32

        self._criterion = nn.BCELoss()
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def serialize_model(self) -> io.BytesIO:
        buffer = io.BytesIO()
        
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        torch.save(quantized_model, buffer)

        return buffer

    def make_input(self, query_vector, sup_doc_vector, inf_doc_vector):
        """
        Make (query, document-pair) input for model.
        """
        return np.array([query_vector, sup_doc_vector, inf_doc_vector], dtype=np.float32).flatten()

    def train(self, pos_train_data, neg_train_data, num_epochs):
        self.model.train()
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
        self.model.eval()

    def apply_updates(self, update_model):
        print('applying updates...')
        update_state = update_model.state_dict()
        for name, param in self.model.named_parameters():
            if name in update_state:
                param.data = (param.data + update_state[name]) / 2.0
