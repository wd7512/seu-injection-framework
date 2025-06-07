import numpy as np
import torch
from framework.legacy.tools import bitflip_float32

class injector():
    def __init__(self, trained_model, X, y, criterion, device = None):
        """
        Initlaise the injector

        - [ ] Check that the model is compatitble
        - [x] Check that the X can do a forward pass
        - [x] Check that the result of the forward pass gives a result with y and criterion
        - [ ] move model and data into the correct device
        - [ ] allows other floating point architectures
        """

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # assert that the model and inputs are in float 32

        self.model = trained_model
        self.model.eval()
        self.model.to(device)
        if isinstance(X, torch.Tensor):
            self.X = X.clone().detach().to(device=device, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32, device=device)

        print(f"Testing a forward pass on {self.device}...")
        y_pred = self.model(self.X)

        self.baseline_score = criterion(y, y_pred)
        print("Basline Criterion Score:", self.baseline_score)
        
        self.criterion = criterion
        self.y = y

    def run_seu(self, bit_i):
        """Perform a bitflip at index i across every variable in the nn"""
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
                original_tensor = tensor.data.clone() # copy original tensor values
                tensor_cpu = original_tensor.cpu().numpy() # move to cpu for iteration of indexes

                for idx in np.ndindex(tensor_cpu.shape):
                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i) # perform bitfliip

                    tensor.data[idx] = torch.tensor(seu_val, device = self.device, dtype=tensor.dtype)
                    criterion_score = self.criterion(self.y, self.model(self.X))
                    tensor.data[idx] = original_tensor[idx]

                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results


def attack(trained_model, X, y, bit_i):


    trained_model
    trained_model.eval()

    # Prepare inputs as torch tensors on the correct device
    X_tensor = torch.tensor(X, dtype=torch.float32, device = "cpu")

    idxs = []
    accs = []
    nams = []

    val_before = []
    val_after = []

    # Iterate through each named parameter (weights and biases) in the model
    for name, param in trained_model.named_parameters():
        # We'll work with a CPU numpy copy for bit‐flipping, but write back to param.data (on device).
        orig_tensor = param.data.clone()  # save original so we can restore
        arr_cpu = orig_tensor.cpu().numpy()

        # Iterate over every index in this parameter tensor
        for idx in np.ndindex(arr_cpu.shape):
            orig_val = float(arr_cpu[idx])
            # Flip the specified bit in the float32 representation
            flipped_val = bitflip_float32(orig_val, bit_i)

            # Write the flipped value back into the model's parameter (in-place)
            with torch.no_grad():
                # Create a 0-dim tensor with the flipped value, on the correct device
                param.data[idx] = torch.tensor(flipped_val, device=device, dtype=param.dtype)

            # Evaluate accuracy on the entire dataset
            with torch.no_grad():
                preds = (trained_model(X_tensor) > 0.5).float()
                accuracy = (preds.eq(y).sum() / len(y)).item()

            idxs.append(idx)
            accs.append(accuracy)
            nams.append(name)
            val_before.append(orig_val)
            val_after.append(flipped_val)

            # Restore the original value before the next bit flip
            with torch.no_grad():
                param.data[idx] = orig_tensor[idx]

        # No need to explicitly reassign param.data to orig_tensor here,
        # since we restored it index by index.

    result_df = pd.DataFrame({
        "IDX": idxs,
        "ACC": accs,
        "NAME": nams,
        "ORIG_VAL": val_before,
        "FLIP_VAL": val_after
    })

    return result_df
