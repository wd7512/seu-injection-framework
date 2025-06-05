import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import struct
import pandas as pd
from tqdm import tqdm
import os
import json
import torch.optim as optim
from torch import nn
import time

def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluate the model on the given dataloader.

    Parameters:
    - model: The PyTorch model to evaluate.
    - dataloader: DataLoader for the test dataset.
    - device: Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
    - accuracy: The accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def get_dataloaders(batch_size=16):
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    # Convert data types
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = (y_train == "ship").astype(np.int64)
    y_test = (y_test == "ship").astype(np.int64)

    train_dataloader = DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size)
    test_dataloader = DataLoader(CustomDataset(X_test, y_test), batch_size=batch_size)

    print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
    print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)
    print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)
    print("y_test shape:", y_test.shape, "dtype:", y_test.dtype)

    return train_dataloader, test_dataloader

def train_cnn(cnn_model, train_dataloader, learning_rate=0.001, num_epochs=1000, loss_threshold=0.01):
    """
    Trains the CNN model using the provided dataloader and hyperparameters.

    Parameters:
        cnn_model (torch.nn.Module): The CNN model to train.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train.
        loss_threshold (float): Threshold for early stopping based on the average loss.

    Returns:
        None
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to GPU
    cnn_model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=cnn_model.weight_decay)
    
    # Training loop
    cnn_model.train()
    time_start = time.perf_counter()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_dataloader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate the loss
            running_loss += loss.item()
        
        # Average loss for this epoch
        average_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")
        
        # Check if loss threshold is reached for early stopping
        if average_loss < loss_threshold:
            print(f"Loss threshold reached. Stopping early at epoch {epoch + 1}.")
            break

    time_taken = time.perf_counter() - time_start
    print(f"Finished Training in {time_taken:.2f} seconds")

    training_data = {
        "Epochs": epoch + 1,
        "Time_Taken": time_taken,
        "Activation_Function": cnn_model.act.__str__(),
        "Final_Loss": average_loss
    }

    return training_data

def show_params(model: nn.Module):
    total_params = 0

    # Print table header
    print(f"{'Name':<20} {'Shape':<30} {'Size':<10}")
    print("="*65)

    # Print each parameter's details
    for name, param in model.named_parameters():
        param_size = param.numel()
        total_params += param_size
        print(f"{name:<20} {str(param.shape):<30} {param_size:<10}")

    # Print total number of parameters
    print("="*60)
    print(f"{'Total number of parameters:':<50} {total_params:<10}")

def bitflip_float32(x, bit_i = np.random.randint(0,32)):

    if hasattr(x, '__iter__'):
        x_ = np.zeros_like(x, dtype = np.float32)
        for i, item in enumerate(x):
            string = list(float32_to_binary(item))
            string[bit_i] = "0" if string[bit_i] == "1" else "1"
            x_[i] = binary_to_float32("".join(string))
    else:
        string = list(float32_to_binary(x))
        string[bit_i] = "0" if string[bit_i] == "1" else "1"
        x_ = binary_to_float32("".join(string))

    return x_

def float32_to_binary(f):
    # Pack float into 4 bytes, then unpack as a 32-bit integer
    [bits] = struct.unpack('!I', struct.pack('!f', f))
    # Format the integer as a 32-bit binary string
    return f'{bits:032b}'

def binary_to_float32(binary_str):
    # Convert binary string to a 32-bit integer
    bits = int(binary_str, 2)
    # Pack the integer into bytes, then unpack as a float
    return struct.unpack('!f', struct.pack('!I', bits))[0]

def tensor_difference(tensor1, tensor2):
    """
    Compute the difference between two tensors, broadcasting if necessary.

    Args:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.

    Returns:
        float: The Frobenius norm of the element-wise difference between the tensors.
    """
    # Broadcasting the tensors to the same shape and then computing the difference
    #print("t1t2", tensor1, tensor2)
    difference = abs(tensor1 - tensor2)
    #print(difference, "Diff")
    mean = torch.mean(difference)
    return mean.item()

def eval_vector(idx, param, bit_i, device, model, test_loader, name):

    original_vector = param[*idx].detach().clone()
    new_vector = [bitflip_float32(a.item(), bit_i) for a in original_vector]
    new_tensor = torch.tensor(new_vector, dtype = original_vector.dtype).to(device)
    param.data[*idx] = new_tensor

    test_accuracy = evaluate_model(model, test_loader, device)


    row = {"Name": name,
            "Index": idx,
            "Test_Accuracy": test_accuracy,
            "Norm": tensor_difference(new_tensor, original_vector.data),
            "Bit_Index": bit_i,
            "N_Attacked": len(new_vector),
            "Proportion": len(new_vector) / param.numel()}
    
    param[*idx].data = original_vector
    
    return pd.DataFrame([row])

def analyse_SEU(model, test_loader, bit_i, device):
    model.to(device)

    model_params = model.named_parameters()
    test_accuracies = pd.DataFrame()

    for name, param in model_params:
        shape = param.size()
        print(f"Testing {name}, {np.prod(shape)} values")
        print()
        for idx in tqdm(np.ndindex(shape)):
            
            original_value = param[idx].item()

            new_value = bitflip_float32(original_value, bit_i)

            param.data[idx] = new_value

            test_accuracy = evaluate_model(model, test_loader, device)

            row = {"Name": name,
                   "Index": idx,
                   "Test_Accuracy": test_accuracy,
                   "Norm": tensor_difference(torch.tensor(original_value, dtype = param.dtype), torch.tensor(new_value, dtype = param.dtype)),
                   "Bit_Index": bit_i}
            test_accuracies = pd.concat([test_accuracies, pd.DataFrame([row])])

            param.data[idx] = original_value
    
    return test_accuracies

def compare_state_dicts(sd1, sd2):
    if sd1.keys() != sd2.keys():
        return False
    
    for key in sd1:
        if not torch.equal(sd1[key], sd2[key]):
            return False
    return True


def analyse_LINE(model, test_loader, bit_i, device):

    model.to(device)
    #sd = model.state_dict()

    model_params = model.named_parameters()
    test_accuracies = pd.DataFrame()

    for name, param in model_params:
        shape = param.size()
        print(f"Testing {name}, {shape}")
        print()

        # flip individual values
        if len(shape) == 1:
            
            idx = [slice(None)]
            row = eval_vector(idx, param, bit_i, device, model, test_loader, name)
            test_accuracies = pd.concat([test_accuracies, row])

        # flip rows and cols of matrix
        elif len(shape) == 2:
            n_rows = shape[0]
            n_cols = shape[1]

            # iterate over rows
            for i in range(n_rows):
                idx = [i]
                row = eval_vector(idx, param, bit_i, device, model, test_loader, name)
                test_accuracies = pd.concat([test_accuracies, row])

            # iterate over columns
            for i in range(n_cols):
                idx = [slice(None), i]
                row = eval_vector(idx, param, bit_i, device, model, test_loader, name)
                test_accuracies = pd.concat([test_accuracies, row])

        # flip rows and cols of matricies within params
        elif len(shape) > 2:
            n_rows, n_cols = shape[-2:]
            for idx in np.ndindex(shape[:-2]):
                idx = list(idx)
                for i in range(n_rows):
                    idx_ = idx + [i]
                    row = eval_vector(idx_, param, bit_i, device, model, test_loader, name)
                    test_accuracies = pd.concat([test_accuracies, row])

                for i in range(n_cols):
                    idx_ = idx + [slice(None), i]
                    row = eval_vector(idx_, param, bit_i, device, model, test_loader, name)
                    test_accuracies = pd.concat([test_accuracies, row])
    
    return test_accuracies

def attack(bit_i, device, model_ids, attack_type, model):
    model_name = model.__name__
    save_folder = "Results"

    if attack_type == "SEU":
        analysis = analyse_SEU
        save_path = os.path.join(save_folder, "SEU")
    elif attack_type == "LINE":
        analysis = analyse_LINE
        save_path = os.path.join(save_folder, "LINE")
    elif attack_type == "MBU":
        analysis = analyse_MBU
        save_path = os.path.join(save_folder, "MBU")
    else:
        return "Invalid Attack Type"
    
    print(f"Performing {attack_type} on {device}")
    _, test_loader = get_dataloaders()

    for model_id in model_ids:

        metadata_path = f"Models/Metadata/{model_id}.json"
        # Load metadata from JSON file
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        

        state_dict_path = f"Models/State_Dicts/{model_id}.pt"
        test_acc_path = os.path.join(save_path, f"{model_name}_{model_id}_{bit_i}.csv")
        print(test_acc_path)

        if os.path.exists(test_acc_path):
            print("Passing", test_acc_path, "| (File already exists)")
            continue

        

        # Load state dictionary from .pt file
        state_dict = torch.load(state_dict_path, map_location=device)
        
        # Create activation function
        activation_function_str = metadata["Activation_Function"]
        if activation_function_str == "ReLU()":
            act = nn.ReLU()
        elif activation_function_str == "Sigmoid()":
            act = nn.Sigmoid()
        elif activation_function_str == "Tanh()":
            act = nn.Tanh()
        else:
            try:
                print(activation_function_str)
                # Use eval to dynamically create the activation function
                act = eval(f"lambda x: {activation_function_str}", {"torch": torch})
            except Exception as e:
                raise ValueError(f"Error creating activation function: {e}")
        
        cnn = model(act, metadata["Dropout"], metadata["Weight_Decay"])

        cnn.load_state_dict(state_dict)
        cnn.to(device)  # Move model to device

        # test adjusted model
        test_accuracies = analysis(cnn, test_loader, bit_i, device)
        test_accuracies["Model"] = model_name
        test_accuracies["Act_Func"] = activation_function_str
        test_accuracies["Test_Delta"] = metadata["Test_Accuracy"] - test_accuracies["Test_Accuracy"]

        # save results
        test_accuracies.to_csv(test_acc_path)

def get_model_ids():
    model_names = os.listdir("Models/State_Dicts")
    metad_names = os.listdir("Models/Metadata")

    model_ids = [m.split(".")[0] for m in model_names]
    metad_ids = [m.split(".")[0] for m in metad_names]

    # check metadata and state dicts align
    for i, m in enumerate(model_ids):
        if m in (metad_ids[i]):
            print(m)
        else:
            print("WARNING, metadata not found for", m)

    return model_ids

def choose_indices(original_index, array_length, std_dev):
    mean = original_index
    x = np.linspace(0,array_length - 1, array_length)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    
    rand = np.random.uniform(0,max(y), size = len(y))
    indices = np.where(rand < y)[0]
    
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, array_length - 1)
    indices = np.unique(indices)
    return indices

def analyse_MBU(model, test_loader, bit_i, device):
    model.to(device)
    sd = model.state_dict()

    model_params = model.named_parameters()
    test_accuracies = pd.DataFrame()
    stds = [1,2,5,10,20]

    for name, param in model_params:
        n = min(100,np.prod(param.size()))
        shape = param.size()
        print(f"Testing {name}, {shape} |{n} times")
        print()

        for std in stds:
            for i in tqdm(range(n)):
                original_vector = param.data.clone()
                new_vector = param.data.flatten().detach().cpu().numpy()
                num_elements = new_vector.size

                original_index = np.random.randint(0, num_elements)
                attack_indices = choose_indices(original_index, num_elements, std)

                for idx in attack_indices:
                    new_vector[idx] = bitflip_float32(new_vector[idx], bit_i)

                param.data = torch.tensor(new_vector, dtype=original_vector.dtype).reshape(shape).to(device)
                test_accuracy = evaluate_model(model, test_loader, device)

                row = {"Name": name, 
                       "Index": "STD: "+str(std), 
                       "Test_Accuracy": test_accuracy, 
                       "Norm": tensor_difference(param.data, original_vector.data),
                       "Bit_Index": bit_i,
                       "N_Attacked": len(attack_indices),
                       "Proportion": len(attack_indices) / param.numel()}
                test_accuracies = pd.concat([test_accuracies, pd.DataFrame([row])])

                param.data = original_vector
        
    return test_accuracies