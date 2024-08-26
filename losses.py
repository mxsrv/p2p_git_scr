import torch
import torch.nn as nn

class CustomWatermarkLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, p=2):
        super(CustomWatermarkLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.p = p

    def forward(self, G, E, S1, S2, k, w, wz, incorrect_keys):
        # Calculate L_w1 using batch processing
        generated_images = G(S1)
        extracted_watermarks = E(generated_images, k)
        L_w1 = torch.norm(extracted_watermarks - w.unsqueeze(0), p=self.p, dim=(1, 2, 3)).mean()
        
        # Calculate L_w2 using batch processing
        extracted_watermarks = E(S2, k)
        L_w2 = torch.norm(extracted_watermarks - wz.unsqueeze(0), p=self.p, dim=(1, 2, 3)).mean()
        
        # Calculate L_w3 using batch processing
        L_w3 = 0
        for kx in incorrect_keys:
            extracted_watermarks = E(generated_images, kx)
            L_w3 += torch.norm(extracted_watermarks - wz.unsqueeze(0), p=self.p, dim=(1, 2, 3)).mean()
        L_w3 /= len(incorrect_keys)
        
        # Combine the losses with the respective weights
        total_loss = self.alpha * L_w1 + self.beta * L_w2 + self.gamma * L_w3
        print("Total Loss:", total_loss.item())
        print("L_w1:", L_w1.item())
        print("L_w2:", L_w2.item())
        print("L_w3:", L_w3.item())
        return total_loss