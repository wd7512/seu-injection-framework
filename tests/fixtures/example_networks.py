"""
This code is entirely ai generated for quick testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_moons_data(test_size=0.3, random_state=0):
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return X_train, X_test, y_train, y_test


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_model(model, x_train, y_train, epochs=300, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()


def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = (model(x_test) > 0.5).float()
        accuracy = preds.eq(y_test).sum().item() / len(y_test)
    return accuracy


def get_example_network(
    net_name="simple", test_size=0.3, random_state=0, train=False, epochs=300, lr=0.01
):
    """
    Args:
        net_name (str): 'simple', 'cnn', 'rnn', or 'gnn'
        test_size (float)
        random_state (int)
        train (bool): If True, trains the model before returning
        epochs (int): Training epochs
        lr (float): Learning rate

    Returns:
        model: trained or untrained model
        X_train, X_test, y_train, y_test: data tensors (except GNN)
        train_fn: training function
        eval_fn: evaluation function

    For GNN:
        returns model, data, labels, train_fn, eval_fn (no training done internally)
    """
    nets = {
        "nn": SimpleNN,
        "cnn": SimpleCNN,
        "rnn": SimpleRNN,
    }

    if net_name not in nets or nets[net_name] is None:
        raise ValueError(
            f"Network '{net_name}' not implemented or missing dependencies."
        )

    else:
        X_train, X_test, y_train, y_test = generate_moons_data(test_size, random_state)
        model = nets[net_name]()

        if train:
            train_model(model, X_train, y_train, epochs=epochs, lr=lr)

        return model, X_train, X_test, y_train, y_test, train_model, evaluate_model
