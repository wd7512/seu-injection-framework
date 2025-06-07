"""
This code is entirely ai generated for quick testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.data import Data
    import torch_geometric.nn as pyg_nn
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False


def generate_moons_data(test_size=0.3, random_state=0):
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return X_train, X_test, y_train, y_test


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class SimpleGNN(torch.nn.Module):
    def __init__(self):
        if not GNN_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for SimpleGNN")
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(2, 16)
        self.conv2 = pyg_nn.GCNConv(16, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return self.softmax(x)


def train_model(model, X_train, y_train, epochs=300, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = (model(X_test) > 0.5).float()
        accuracy = preds.eq(y_test).sum().item() / len(y_test)
    return accuracy


def get_example_network(net_name='simple', test_size=0.3, random_state=0, train=False, epochs=300, lr=0.01):
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
        'simple': SimpleNN,
        'cnn': SimpleCNN,
        'rnn': SimpleRNN,
        'gnn': SimpleGNN if GNN_AVAILABLE else None,
    }

    if net_name not in nets or nets[net_name] is None:
        raise ValueError(f"Network '{net_name}' not implemented or missing dependencies.")

    if net_name == 'gnn':
        edge_index = torch.tensor([
            [0, 1, 2, 3, 0, 2],
            [1, 0, 3, 2, 2, 0]
        ], dtype=torch.long)
        x = torch.randn((4, 2), dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        model = nets[net_name]()
        # No internal training for GNN (complex)
        return model, data, y, train_model, evaluate_model

    else:
        X_train, X_test, y_train, y_test = generate_moons_data(test_size, random_state)
        model = nets[net_name]()

        if train:
            train_model(model, X_train, y_train, epochs=epochs, lr=lr)

        return model, X_train, X_test, y_train, y_test, train_model, evaluate_model
