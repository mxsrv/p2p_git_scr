import torch
import torch.nn as nn

class ECNetwork(nn.Module):
    def __init__(self, c1, c2, c):
        super(ECNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=(c1+c2), out_channels=48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=c, kernel_size=3, padding=1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.Tanh()(x)
        return x
    
def test():

    x = torch.randn((1, 64, 256, 256))
    y = torch.randn((1, 128, 256, 256))
    model = ECNetwork(64, 128, 256)
    preds = model(x, y)
    print(preds.shape)


if __name__ == "__main__":
    test()
        