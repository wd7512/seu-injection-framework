
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from stopwatch import Stopwatch
import sys


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
    #model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()


    #model = torch.compile(model) # only works with cuda and cpu
    #model = torch.jit.script(model)  # mainly works with mps

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
    num_workers= 0
    val_dataset = datasets.ImageNet(root=data_root_dir, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # =================================================== #
    # Evaluate the model in the validation set
    # =================================================== #
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    stopwatch = Stopwatch()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = outputs.topk(5, 1, True, True)

            correct_top1 += (pred[:, 0] == labels).sum().item()
            correct_top5 += sum([labels[i] in pred[i] for i in range(len(labels))])
            total += labels.size(0)

    print(f"Inference for  {len(val_dataset)} images took {stopwatch.elapsedTime():.3f} seconds")

    # Print final results
    print(f"Top-1 Accuracy: {correct_top1 / total * 100:.2f}%")
    print(f"Top-5 Accuracy: {correct_top5 / total * 100:.2f}%")
    print(f"batch_size={batch_size} num_workers={num_workers}")


if __name__ == "__main__":
    main()