import torch
from dataset import MapDataset
from generate_key import generate_incorrect_keys_batch, load_key
from generator_model import Generator
from watermark_extraction import WatermarkExtractionModel
from utils import load_checkpoint, load_watermark_image, generate_zero_watermark
import config
from torch.utils.data import DataLoader
from losses import CustomWatermarkLoss
import torch.nn as nn
import torch.optim as optim

def evaluate_task_performance(gen, test_loader, l1_loss):
    gen.eval()
    total_l1_loss = 0
    psnr_total = 0
    num_samples = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            y_fake = gen(x)
            
            # Task performance metric: L1 loss
            loss = l1_loss(y_fake, y)
            total_l1_loss += loss.item() * x.size(0)
            
            # Calculate PSNR (Peak Signal-to-Noise Ratio) as another performance metric
            mse = torch.mean((y_fake - y) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            psnr_total += psnr.item() * x.size(0)
            
            num_samples += x.size(0)

    avg_l1_loss = total_l1_loss / num_samples
    avg_psnr = psnr_total / num_samples
    return avg_l1_loss, avg_psnr


def test_model_accuracy():
    # Initialize models
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(list(gen.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint(f"{config.CHECKPOINT_GEN}", gen, optimizer=opt_gen, lr=config.LEARNING_RATE)

    # Load test dataset
    test_dataset = MapDataset(root_dir="dataset/val")
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    l1_loss = nn.L1Loss()

    # Evaluate task accuracy (L1 loss + PSNR)
    task_l1_loss, task_psnr = evaluate_task_performance(gen, test_loader, l1_loss)
    print(f"Task Accuracy - Avg L1 Loss: {task_l1_loss}, Avg PSNR: {task_psnr}")




if __name__ == "__main__":
    test_model_accuracy()