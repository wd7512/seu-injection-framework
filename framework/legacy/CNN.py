from torch import nn
from torch.nn import functional as F
import torch


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
        #x = self.softmax(x)
        return x
    
class SmartPoolCNN(CNN):
    def __init__(self, act, dropout_rate=0, weight_decay=0):
        super(SmartPoolCNN, self).__init__(act, dropout_rate, weight_decay)
        self.smart_pool = self.smart_pool_layer
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.threshold = 10

    def smart_pool_layer(self, x):
        # Clone the tensor to avoid in-place modification issues
        x = x.clone()
        # Replace extreme values with -inf to be ignored by max pooling
        x[x.abs() > self.threshold] = -10
        return self.pool(x)

    def forward(self, x):
        x = self.smart_pool(self.act(self.conv1(x)))
        x = self.smart_pool(self.act(self.conv2(x)))
        x = self.smart_pool(self.act(self.conv3(x)))
        x = self.smart_pool(x)
        
        x = self.dropout(x)
        x = x.view(-1, 5*5 * 32)
        x = self.fc1(x)
        #x = self.softmax(x)
        return x
    
    
class SmartPoolSlowCNN(CNN):
    def __init__(self, act, dropout_rate=0, weight_decay=0):
        super(SmartPoolSlowCNN, self).__init__(act, dropout_rate, weight_decay)
        self.pool = SmartPool2D(kernel_size=2)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartPool2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, threshold=10):
        super(SmartPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.threshold = threshold

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        
        n, c, h, w = x.shape
        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        
        # Unfold the input tensor to get sliding windows
        x_unfold = F.unfold(x, kernel_size=(kh, kw), stride=(sh, sw))
        x_unfold = x_unfold.view(n, c, kh*kw, -1)
        
        # Apply the threshold condition and set values exceeding the threshold to the minimum value in each region
        min_val, _ = torch.min(x_unfold, dim=2, keepdim=True)
        mask = x_unfold.abs() > self.threshold
        x_unfold[mask] = min_val.expand_as(x_unfold)[mask]
        
        # Compute the maximum value in each region
        max_val, _ = torch.max(x_unfold, dim=2)
        
        # Reshape to the output size
        output = max_val.view(n, c, out_h, out_w)
        
        return output
    

# class FaultInjectionCNN(CNN):
#     def __init__(self, act, dropout_rate=0, weight_decay=0):
#         super(FaultInjectionCNN, self).__init__(act, dropout_rate, weight_decay)

#     def fault_injection(self, x):
#         x = x.clone()  # Avoid in-place modification issues
#         n = x.numel()
#         mask = torch.rand(n, device=x.device) < 0.1  # 10% chance to flip a bit
#         indices = torch.nonzero(mask).flatten()
#         for idx in indices:
#             x.view(-1)[idx] = torch.tensor(, dtype=x.dtype, device=x.device)
#         return x

#     def forward(self, x):
#         x = self.pool(self.act(self.conv1(x)))
#         x = self.pool(self.act(self.conv2(x)))
#         x = self.pool(self.act(self.conv3(x)))
#         x = self.pool(x)
        
#         x = self.dropout(x)
#         x = x.view(-1, 5*5 * 32)
#         x = self.fc1(x)
#         #x = self.softmax(x)
#         return x