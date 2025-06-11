import numpy as np
import torch
from framework.bitflip import bitflip_float32

class Injector():
    def __init__(self, trained_model, X, y, criterion, device = None):
        """
        Initlaise the injector

        WARNING: this is built assuming floating point 32 values are used

        - [ ] Check that the model is compatitble
        - [x] Check that the X can do a forward pass
        - [x] Check that the result of the forward pass gives a result with y and criterion
        - [x] move model and data into the correct device
        - [ ] allows other floating point architectures
        """

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.criterion = criterion
        self.model = trained_model.to(device)
        self.model.eval()
        self.y = y

        if isinstance(X, torch.Tensor):
            self.X = X.clone().detach().to(device=device, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32, device=device)

        print(f"Testing a forward pass on {self.device}...")

        self.baseline_score = criterion(self.model, self.X, self.y, self.device)
        print("Basline Criterion Score:", self.baseline_score)
        


    def run_seu(self, bit_i: int, layer_name__ = None):
        """Perform a bitflip at index i across every variable in the nn"""

        assert bit_i in range(0,33)

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": []
        }

        with torch.no_grad(): # disable tracking gradients 
            # iterate though each layer of the nn
            for layer_name, tensor in self.model.named_parameters():

                if layer_name__: # check if it is specified for a layer
                    if layer_name__ != layer_name: # skip layer if not the layer name
                        continue

                original_tensor = tensor.data.clone() # copy original tensor values
                tensor_cpu = original_tensor.cpu().numpy() # move to cpu for iteration of indexes

                for idx in np.ndindex(tensor_cpu.shape):
                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i) # perform bitfliip

                    tensor.data[idx] = torch.tensor(seu_val, device = self.device, dtype=tensor.dtype)
                    criterion_score = self.criterion(self.model, self.X, self.y, self.device)
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results
    
    def run_stochastic_seu(self, bit_i: int, p: float, layer_name__ = None):
        """Perform a bitflip at index i across every variable in the nn"""

        assert (p >= 0) and (p <= 1)
        assert bit_i in range(0,33)

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": []
        }

        with torch.no_grad(): # disable tracking gradients 
            # iterate though each layer of the nn
            for layer_name, tensor in self.model.named_parameters():

                if layer_name__: # check if it is specified for a layer
                    if layer_name__ != layer_name: # skip layer if not the layer name
                        continue

                original_tensor = tensor.data.clone() # copy original tensor values
                tensor_cpu = original_tensor.cpu().numpy() # move to cpu for iteration of indexes

                for idx in np.ndindex(tensor_cpu.shape):
                    if np.random.uniform(0,1) > p:
                        continue

                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i) # perform bitfliip

                    tensor.data[idx] = torch.tensor(seu_val, device = self.device, dtype=tensor.dtype)
                    criterion_score = self.criterion(self.model, self.X, self.y, self.device)
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results