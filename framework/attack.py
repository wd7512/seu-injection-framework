import numpy as np
import torch
from framework.bitflip import bitflip_float32, cauchy
from tqdm import tqdm
from random import sample
torch.set_float32_matmul_precision("high")

class Injector:
    def __init__(
        self, trained_model, criterion, device=None, X=None, y=None, data_loader=None
    ):
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

        self.use_data_loader = False

        print(f"Testing a forward pass on {self.device}...")

        if data_loader:
            if X or y:
                raise ValueError("Cannot pass both a dataloader and X and y values")

            self.use_data_loader = True
            self.data_loader = data_loader


            try:
                compiled_model = torch.compile(self.model) 
                self.baseline_score = criterion(compiled_model, self.data_loader, self.device)
                print("Compiled model works")
                self.model = compiled_model
            except Exception as e:
                print(f"Compilation failed: {e}")
                self.baseline_score = criterion(self.model, self.data_loader, self.device)

        else:
            self.y = y

            if isinstance(X, torch.Tensor):
                self.X = X.clone().detach().to(device=device, dtype=torch.float32)
            else:
                self.X = torch.tensor(X, dtype=torch.float32, device=device)

            self.baseline_score = criterion(self.model, self.X, self.y, self.device)

        print("Basline Criterion Score:", self.baseline_score)

    def get_criterion_score(self):
        if self.use_data_loader:
            return self.criterion(self.model, self.data_loader, self.device)
        else:
            return self.criterion(self.model, self.X, self.y, self.device)

    def run_seu(self, bit_i: int, layer_name__=None):
        """Perform a bitflip at index i across every variable in the nn"""

        assert bit_i in range(0, 33)

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

        with torch.no_grad():  # disable tracking gradients
            # iterate though each layer of the nn
            for layer_name, tensor in self.model.named_parameters():

                if layer_name__:  # check if it is specified for a layer
                    if layer_name__ != layer_name:  # skip layer if not the layer name
                        continue

                print("Testing Layer: ", layer_name)

                original_tensor = tensor.data.clone()  # copy original tensor values
                tensor_cpu = (
                    original_tensor.cpu().numpy()
                )  # move to cpu for iteration of indexes

                for idx in tqdm(np.ndindex(tensor_cpu.shape)):
                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i)  # perform bitfliip

                    tensor.data[idx] = torch.tensor(
                        seu_val, device=self.device, dtype=tensor.dtype
                    )
                    criterion_score = self.get_criterion_score()
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results

    def run_stochastic_seu(self, bit_i: int, p: float, layer_name__=None):
        """Perform a bitflip at index i across every variable in the nn"""

        assert (p >= 0) and (p <= 1)
        assert bit_i in range(0, 33)

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

        with torch.no_grad():  # disable tracking gradients
            # iterate though each layer of the nn
            for layer_name, tensor in self.model.named_parameters():

                if layer_name__:  # check if it is specified for a layer
                    if layer_name__ != layer_name:  # skip layer if not the layer name
                        continue

                print("Testing Layer: ", layer_name)

                original_tensor = tensor.data.clone()  # copy original tensor values
                tensor_cpu = (
                    original_tensor.cpu().numpy()
                )  # move to cpu for iteration of indexes

                for idx in tqdm(np.ndindex(tensor_cpu.shape)):
                    if np.random.uniform(0, 1) > p:
                        continue

                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i)  # perform bitfliip

                    tensor.data[idx] = torch.tensor(
                        seu_val, device=self.device, dtype=tensor.dtype
                    )
                    criterion_score = self.get_criterion_score()
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results

    def run_n_seu(self, bit_i: int, n = 1,layer_name__=None):
        """Perform a bitflip at index i across every variable in the nn"""

        assert bit_i in range(0, 33)

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

        with torch.no_grad():  # disable tracking gradients
            # iterate though each layer of the nn
            for layer_name, tensor in self.model.named_parameters():

                if layer_name__:  # check if it is specified for a layer
                    if layer_name__ != layer_name:  # skip layer if not the layer name
                        continue

                print("Testing Layer: ", layer_name)

                original_tensor = tensor.data.clone()  # copy original tensor values
                tensor_cpu = (
                    original_tensor.cpu().numpy()
                )  # move to cpu for iteration of indexes

                indices = list(np.ndindex(tensor_cpu.shape))
                indices = sample(indices, min(n, len(indices)))

                for idx in tqdm(indices, desc=f"Flipping bits in {layer_name}"):
                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i)  # perform bitfliip

                    tensor.data[idx] = torch.tensor(
                        seu_val, device=self.device, dtype=tensor.dtype
                    )
                    criterion_score = self.get_criterion_score()
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results

    def run_stochastic_seu_layer(self, bit_i: int, k: int):
        """Perform a bitflip at index i across k randomly selected variables in each layer of the nn"""

        assert bit_i in range(0, 33)

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": []
        }

        with torch.no_grad():  # disable tracking gradients
            # iterate though each layer of the nn
            num_layers = 0
            total_k = 0
            for layer_name, tensor in tqdm(self.model.named_parameters(), desc="Layers"):
                # model.named_parameters() iterates layers in the model in order
                # We are adding the layer_name to a list, the list preserves the order
                num_layers += 1

                original_tensor = tensor.data.clone()  # copy original tensor values
                tensor_cpu = original_tensor.cpu().numpy()  # move to cpu for iteration of indexes

                # Generate all possible indices
                #   e.g. (4,2) =[(0,0), (0,1), (1,0),(1,1), (2,0),(2,1), (3,0),(3,1)]
                shape = tensor_cpu.shape
                all_indices = list(np.ndindex(shape))  # list of tuples
                if k > tensor_cpu.size:
                    # in case k > num params
                    print(
                        f"Given k {k}, not enough samples {tensor_cpu.size} for layer {layer_name}, sampling all {tensor_cpu.size} instead")
                    k = tensor_cpu.size
                selected_indices = list()
                #print(f"Selecting {k} from Layer: {layer_name}")
                total_k += k
                for i in np.random.choice(len(all_indices), k):
                    selected_indices.append(all_indices[i])

                for idx in tqdm(selected_indices, desc="k bitflips", leave=False):
                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i)  # perform bitfliip

                    tensor.data[idx] = torch.tensor(seu_val, device=self.device, dtype=tensor.dtype)
                    criterion_score = self.get_criterion_score()
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)
        #print(f"Selected {total_k} from {num_layers} layers")
        return results
    
    def run_singular_delta_seu(self, delta, layer_name__):
        """Add value deltas to the first value in a layer"""


        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

        with torch.no_grad():  # disable tracking gradients
            # iterate though each layer of the nn
            for layer_name, tensor in self.model.named_parameters():

                if layer_name__:  # check if it is specified for a layer
                    if layer_name__ != layer_name:  # skip layer if not the layer name
                        continue

                print("Testing Layer: ", layer_name)

                original_tensor = tensor.data.clone()  # copy original tensor values
                tensor_cpu = (
                    original_tensor.cpu().numpy()
                )  # move to cpu for iteration of indexes

                for idx in tqdm(np.ndindex(tensor_cpu.shape)):
                    original_val = tensor_cpu[idx]
                    seu_val = original_val + delta

                    tensor.data[idx] = torch.tensor(
                        seu_val, device=self.device, dtype=tensor.dtype
                    )
                    criterion_score = self.get_criterion_score()
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

                    break # only do it once 

        return results

    def run_singular_seu_cauchy(self, layer_name__):
        """Performs a cauchy delta to a random bit in each layer"""


        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

        with torch.no_grad():  # disable tracking gradients
            # iterate though each layer of the nn
            for layer_name, tensor in self.model.named_parameters():

                if layer_name__:  # check if it is specified for a layer
                    if layer_name__ != layer_name:  # skip layer if not the layer name
                        continue

                print("Testing Layer: ", layer_name)

                original_tensor = tensor.data.clone()  # copy original tensor values
                tensor_cpu = (
                    original_tensor.cpu().numpy()
                )  # move to cpu for iteration of indexes

                idx = tuple(np.random.randint(s) for s in tensor_cpu.shape)


                original_val = tensor_cpu[idx]
                seu_val = cauchy(original_val, bound=10)

                tensor.data[idx] = torch.tensor(
                    seu_val, device=self.device, dtype=tensor.dtype
                )
                criterion_score = self.get_criterion_score()
                tensor.data[idx] = original_tensor[idx]

                results["tensor_location"].append(idx)
                results["criterion_score"].append(criterion_score)
                results["layer_name"].append(layer_name)
                results["value_before"].append(original_val)
                results["value_after"].append(seu_val)

        return results