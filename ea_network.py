import math
import torch
import torch.nn as nn

class EANetwork(nn.Module):
    def __init__(self):
        super(EANetwork, self).__init__()


        # One 3x3 Convolutional layer with 32 filters follows by 2 InceptionResidualBlocks and finally a 3x3 Convolutional layer with 3 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.inception1 = InceptionResidualBlock(in_channels=32)
        self.inception2 = InceptionResidualBlock(in_channels=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.inception1(x)
        x = self.relu(x)
        x = self.inception2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.tanh(x)

        # Reshape to match EB output

        BS, _, H, W = x.shape
        
        c1 = math.floor((3 * H * W) / (64 * 64))
        # Reshape the tensor
        x = x.view(BS, c1, 64, 64)
        return x



def test():
    x = torch.randn((1, 3, 256, 256))
    model = EANetwork()
    preds = model(x)
    print(preds.shape)


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResidualBlock, self).__init__()
        
        # Branch 1: 1x1 Convolution
        self.branch1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        
        # Branch 2: 1x1 Convolution followed by 3x3 Convolution
        self.branch2_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Branch 3: 1x1 Convolution followed by a 5x5 Convolution 
        self.branch3_1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch3_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        # 1x1 Convolution applied after concat to reduce from 96 to 32 channels
        self.conv_linear = nn.Conv2d(96, in_channels, kernel_size=1)

        # Activation function
        self.relu = nn.ReLU()

        # Last layer Activation (tanh)
        self.tanh = nn.Tanh()

       


    def forward(self, x):
        # Branch 1
        branch1 = self.branch1(x)
        branch1 = self.relu(branch1)
        
        # Branch 2
        branch2 = self.branch2_1(x)
        branch2 = self.relu(branch2)
        branch2 = self.branch2_2(branch2)
        branch2 = self.relu(branch2)
        
        # Branch 3
        branch3 = self.branch3_1(x)
        branch3 = self.relu(branch3)
        branch3 = self.branch3_2(branch3)
        branch3 = self.relu(branch3)
        
        # Concatenate branches
        out = torch.cat([branch1, branch2, branch3], 1)

        # Linear convolution
        out = self.conv_linear(out)
        
        # Residual connection
        out += x
        
        # Activation function
        out = self.tanh(out)
        
        return out
    


if __name__ == "__main__":
    test()