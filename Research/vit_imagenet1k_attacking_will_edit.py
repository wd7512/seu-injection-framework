
import os
import sys

# Save current working directory
cwd = os.getcwd()
# Change to parent directory
parent_dir = os.path.abspath(os.path.join(cwd, '..'))
os.chdir(parent_dir)
# Temporarily add parent directory to sys.path
sys.path.insert(0, parent_dir)
import framework
sys.path.pop(0)
# Return to original directory
os.chdir(cwd)



from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from framework.criterion import classification_accuracy_loader

from framework.attack import Injector

import pandas as pd
import torch

import json
from datetime import datetime
import numpy as np

from stopwatch import Stopwatch


class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_best_device():
    """
    Determines the best device (in {cpu, mps, cuda} to use.
    :return: device (as a str)
    """
    if torch.cuda.is_available():
        #return torch.device("cuda")
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        #return torch.device("mps")
        return "mps"
    else:
        #return torch.device("cpu")
        return "cpu"


def main():
    if len(sys.argv) != 3:
        print("Usage <data root dir> <k errors per layer> <index>")
        print("Example python vit_imagenet1k_attacking_will_edit.py ./data/ILSVRC2012_5K 1")
        return
    data_root_dir = sys.argv[1]
    delta     = float(sys.argv[2]) # which index is affected

    # =================================================== #
    # Define the backend
    # =================================================== #
    device = get_best_device()
    print(f"Setting device to {device}")

    # =================================================== #
    # Load pretrained ViT model
    # =================================================== #
    """
    ViT_B_16_Weights.IMAGENET1K_V1 trained from scratch on ImageNetk
    There are 152 layers in the ViT model
    Be aware that the ICLR paper ViT-B/16 is trained on ImageNet21k and transferred to ImageNet1k
    so there will be difference between the two.
    The paper indicated 77.91 (ImageNet21k -> ImageNet1k)
    We obtained 80.07 (vit_b_16, from scratch on ImageNet1k)
    We obtained 75.91 (vit_b_32, from scratch on ImageNet1k, much faster)
    """
    # NB: vit_b_16 is about twice as slow compared to vit_b_32 (e.g. 5.5 min vs 2 min)
    # model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()

    # ImageNet standard preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # =================================================== #
    # Load the ImageNet validation set
    # =================================================== #
    batch_size = 256
    num_workers = 0 # Getting too many file handles open, 0 makes loader run in main
    val_dataset = datasets.ImageNet(root=data_root_dir, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # =================================================== #
    # Evaluate the model with no errors (baseline)
    # =================================================== #
    # accuracy_score = classification_accuracy_loader(model, val_loader, device)
    # print(f"accuracy_score={accuracy_score}") # will be printed again in Injector constructor

    # =================================================== #
    # Adds errors Evaluate the model with no errors (baseline)
    # =================================================== #
    # This does one forward pass as a baseline
    inj = Injector(model, classification_accuracy_loader, device=device, data_loader=val_loader)

    """
    With 152 layers, we want to test 7 delta values = [1e-5,1e-3,1e-1,1,10,100,1000]
    I imagine these can be run in parallel? 
    """

    # =================================================== #
    # Evaluate the model with errors
    # NB there are 152 layers in the ViT model
    # =================================================== #

    params = model.named_parameters()
    for layer_name, tensor in params:
        stopwatch = Stopwatch()

        results = inj.run_singular_delta_seu(layer_name__=layer_name, delta=delta)
        df = pd.DataFrame(results)
        print(f"Dataframe for delta {delta}")
        print(df)
        bitflips = len(df)

        current_datetime = datetime.now()
        #date_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        date_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"results_delta{delta}_{date_string}.json"
        with open(filename, 'w') as fp:
            json.dump(results, fp, cls=NumpyTypeEncoder)

        print(f"Inference for {bitflips} bitflips took {stopwatch.elapsedTime()/60:.1f} minutes")
        print(f"Average {stopwatch.elapsedTime()/bitflips} seconds/bitflip")


if __name__ == "__main__":
    main()