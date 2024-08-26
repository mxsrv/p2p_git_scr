import torch 
import torch.nn as nn
import config
import ea_network, eb_network, ec_network
import numpy as np

class WatermarkExtractionModel(nn.Module):
    def __init__(self, key_channels=1, input_dims=[256, 256], target_dims=[64, 64]):
        super(WatermarkExtractionModel, self).__init__()

        c = 3
        c1 = int(np.floor((3*input_dims[0]*input_dims[1])/(target_dims[0]*target_dims[1])))
        c2 = 48 # emprically set, see paper

        self.ea_network = ea_network.EANetwork().to(config.DEVICE)
        self.eb_network = eb_network.EBNetwork(key_channels, c2).to(config.DEVICE, dtype=torch.float32)
        self.ec_network = ec_network.ECNetwork(c1, c2, c).to(config.DEVICE)

    def forward(self, image, key):
        # compute ea and eb then concat and input into ec
        ea = self.ea_network.forward(image)
        eb = self.eb_network.forward(key)
        eb = eb.expand_as(ea)
        ec = self.ec_network.forward(ea, eb)
        return ec
    

def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 64, 256, 256))
    model = WatermarkExtractionModel(64, [256, 256], [256, 256])
    preds = model(x, y)
    print(preds.shape)



if __name__ == "__main__":
    test()






