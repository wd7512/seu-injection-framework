

import os
import shutil
from scipy.io import loadmat

import random

def reorganise_val():
    """
    Reorganises the val ImageNety val folder after unzipping to be
    compatible with PyTorch datasets.
    :return:
    """
    # Paths, validation has 50k images, test set has 100k images
    # Sadly, the test set does not have ground truth since it is a competition :(
    # val_dir = "./data/ILSVRC2012/test"
    val_dir = "./data/ILSVRC2012/val"
    devkit_dir = "./data/ILSVRC2012/ILSVRC2012_devkit_t12"

    # Load ground truth
    with open(os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt")) as f:
        val_labels = [int(x.strip()) for x in f.readlines()]

    # Load synset mapping (index → WordNet ID)
    meta = loadmat(os.path.join(devkit_dir, "data", "meta.mat"))
    synsets = meta['synsets']
    # Select only 1,000 classes
    imagenet_synsets = [str(synsets[i][0][1][0]) for i in range(len(synsets)) if synsets[i][0][0][0] <= 1000]
    assert len(imagenet_synsets) == 1000

    # Sort filenames
    img_files = sorted([f for f in os.listdir(val_dir) if f.endswith(".JPEG")])

    # Make subfolders and move files
    for idx, filename in enumerate(img_files):
        label = val_labels[idx] - 1  # convert to 0-based index
        class_name = imagenet_synsets[label]
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        shutil.move(os.path.join(val_dir, filename), os.path.join(class_dir, filename))

    print("Validation set reorganized successfully.")


def sample_val():
    # Paths
    source_dir = "./data/ILSVRC2012/val"        # your original val/ directory with 1000 subfolders
    target_dir = "./data/ILSVRC2012/val_small"  # where 5 samples per class will be saved

    # Parameters
    samples_per_class = 5
    random.seed(42)  # for reproducibility

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Loop over class folders
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        if len(images) < samples_per_class:
            print(f"Not enough images in class '{class_name}' (found {len(images)}) — skipping.")
            continue

        # Randomly select samples
        selected_images = random.sample(images, samples_per_class)

        # Create corresponding folder in target
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        # Copy selected images
        for img in selected_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_class_dir, img)
            shutil.copy2(src, dst)

    print("Subset creation complete.")




def main():
    #reorganise_val()
    sample_val()

if __name__ == "__main__":
    main()