import torch 
from torch import nn
import torch.optim as optim

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class LogisticRegressionAddtive(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionAddtive, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
        self.linear2 = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x, z):
        outputs = torch.sigmoid(self.linear1(x) + self.linear2(z))
        return outputs

class LogisticRegressionBilinear(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(LogisticRegressionBilinear, self).__init__()
        self.bilinear = torch.nn.Bilinear(input_dim1, input_dim2, output_dim)
        
    def forward(self, x, z):
        outputs = torch.sigmoid(self.bilinear(x,z))
        return outputs