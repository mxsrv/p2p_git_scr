import torch.nn as nn
import config
from generate_key import generate_secret_key, save_key, load_key

class EBNetwork(nn.Module):
    def __init__(self, t, c2):
        super(EBNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=t, out_channels=12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=c2, kernel_size=3, padding=1)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.tanh(x)
        return x

def test():
    h, w, t = 64, 64, 1  # Example dimensions
    c2 = 48  # Example output channels

    # Generate and save the secret key
    key_tensor = generate_secret_key(h, w, t, block_size=(4, 4), seed=187)
    save_key(key_tensor, "secret_key.pt")

    # Load the key and test the EBNetwork
    loaded_key = load_key("secret_key.pt")
    model = EBNetwork(t, c2)
    preds = model(loaded_key)
    print("Input shape:", loaded_key.shape)
    print("Output shape:", preds.shape)

if __name__ == "__main__":
    test()