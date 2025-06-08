from torch import nn

class CNN(nn.Module):
    def __init__(self, act, dropout_rate=0, weight_decay=0):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(5*5 * 32, 2)  # Output layer with 2 neurons for binary classification
        
        self.act = act
        self.dropout = nn.Dropout(p=dropout_rate)
        self.weight_decay = weight_decay
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        x = self.pool(x)
        
        x = self.dropout(x)
        x = x.view(-1, 5*5 * 32)
        x = self.fc1(x)

        return x