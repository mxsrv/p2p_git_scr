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

def evaluate_watermark_accuracy(gen, extraction_model, test_loader, custom_loss):
    extraction_model.eval()
    total_custom_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            y_fake = gen(x)
            
            # Load the required watermark components
            watermark_image = load_watermark_image(height=64, width=64, filepath="dataset/watermark/watermark.jpg")
            zero_watermark = generate_zero_watermark(height=64, width=64)
            key = load_key()
            incorrect_keys = generate_incorrect_keys_batch("secret_key.pt", 8, x.size(0))
            
            # Calculate the watermark loss
            w_loss = custom_loss(
                G=gen,
                E=extraction_model,
                S1=x,
                S2=y,
                k=key,
                w=watermark_image,
                wz=zero_watermark,
                incorrect_keys=incorrect_keys
            )
            total_custom_loss += w_loss.item() * x.size(0)
            num_samples += x.size(0)
    
    avg_custom_loss = total_custom_loss / num_samples
    return avg_custom_loss

def test_model_accuracy():
    # Initialize models
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    extraction_model = WatermarkExtractionModel().to(config.DEVICE)
    opt_gen = optim.Adam(list(gen.parameters()) + list(extraction_model.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))


    # Load trained models
    load_checkpoint(f"checkpoints/_epoch999_{config.CHECKPOINT_GEN}", gen, optimizer=opt_gen, lr=config.LEARNING_RATE)
    load_checkpoint(f"checkpoints/_epoch999_extraction_model.pth.tar", extraction_model, optimizer=opt_gen, lr=config.LEARNING_RATE)
    
    # Load test dataset
    test_dataset = MapDataset(root_dir="dataset/val")
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    # Define losses
    l1_loss = nn.L1Loss()
    custom_loss = CustomWatermarkLoss(alpha=1.0, beta=0.5, gamma=0.3, p=2)
    
    # Evaluate task accuracy (L1 loss + PSNR)
    task_l1_loss, task_psnr = evaluate_task_performance(gen, test_loader, l1_loss)
    print(f"Task Accuracy - Avg L1 Loss: {task_l1_loss}, Avg PSNR: {task_psnr}")
    
    # Evaluate watermark accuracy (Custom watermark loss)
    watermark_loss = evaluate_watermark_accuracy(gen, extraction_model, test_loader, custom_loss)
    print(f"Watermark Accuracy - Avg Custom Loss: {watermark_loss}")

if __name__ == "__main__":
    test_model_accuracy()
