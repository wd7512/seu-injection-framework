
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
import time

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
    if len(sys.argv) != 2:
        print("Usage <data root dir>")
        print("Example python vit_imagenet1k_attacking.py ./data/ILSVRC2012_5K")
        return
    data_root_dir = sys.argv[1]

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
    num_workers = 4
    val_dataset = datasets.ImageNet(root=data_root_dir, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # =================================================== #
    # Evaluate the model with no errors (baseline)
    # =================================================== #
    accuracy_score = classification_accuracy_loader(model, val_loader, device)
    #print(f"accuracy_score={accuracy_score}") # will be printed again in Injector constructor

    # =================================================== #
    # Adds errors Evaluate the model with no errors (baseline)
    # =================================================== #
    # This does one forward pass as a baseline
    inj = Injector(model, classification_accuracy_loader, device=device, data_loader=val_loader)
    total_res = pd.DataFrame()

    params = model.named_parameters()
    for bit_i in [0,1,3,6,10,15,21]:
        for layer_name, tensor in params:
            size = tensor.numel()

            n_attacks = min(10,int(size * 0.1))
            p = n_attacks / size

            # =================================================== #
            # Evaluate the model with errors
            # =================================================== #
            print("Attacking:", [bit_i, p, n_attacks,layer_name])
            s = time.perf_counter()
            results = inj.run_stochastic_seu(bit_i=bit_i,p= p, layer_name__=layer_name)
            e = time.perf_counter()
            # =================================================== #
            # Print results (pandas maybe prettier)
            # =================================================== #
            df = pd.DataFrame(results)
            df["bit_i"] = bit_i
            df["time_per_seu_(s)"] = (s - e) / len(df)
            print(df)

            total_res = pd.concat([total_res, df], axis = 0)
            total_res.to_csv("Current_Results.csv")

if __name__ == "__main__":
    main()