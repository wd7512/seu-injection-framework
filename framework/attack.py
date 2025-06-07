import numpy as np
import pandas as pd
import torch
from framework.legacy.tools import bitflip_float32

def attack(trained_model, X, y, bit_i, device):
    trained_model.to(device)
    trained_model.eval()

    # Prepare inputs as torch tensors on the correct device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

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
