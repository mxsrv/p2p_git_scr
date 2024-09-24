import torch
from segmentation_dataset import SegmentationDataset
from utils import save_checkpoint, load_checkpoint, save_some_examples, load_watermark_image, generate_zero_watermark, save_watermark_extraction
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from flowers_dataset import FlowersDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from watermark_extraction import WatermarkExtractionModel  # Import the extraction model
from losses import CustomWatermarkLoss  # Assuming you implemented the loss in a separate file
from generate_key import load_key, generate_incorrect_keys_batch

torch.backends.cudnn.benchmark = True

def train_fn(
        disc, gen, loader, opt_disc, opt_gen, l1_loss, bce_loss, g_scaler, d_scaler, model, extraction_model = None, extraction_loss = None, naive_model = None, naive_loss = None
):
    train_loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(train_loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
                # Train Discriminator
        print("Training Discriminator")
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        print("Training Generator")
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

            if model == "watermark":
                # Calculate the custom watermark loss
                # Note: Adjust the inputs to the CustomWatermarkLoss as per your requirement.
                custom_w_loss = extraction_loss(
                    G=gen,
                    E=extraction_model,
                    S1=x,  
                    S2=y,  
                    k=load_key(),
                    w=load_watermark_image(height=64, width=64, filepath="dataset/watermark/watermark.jpg"),
                )
                G_loss += custom_w_loss
            
            elif model == "naive":
                print("Training Naive Model")
                naive_loss = 0
            
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            metrics = {
                'D_real': torch.sigmoid(D_real).mean().item(),
                'D_fake': torch.sigmoid(D_fake).mean().item(),
            }
            
            if model == "watermark":
                metrics['watermarking_loss'] = torch.sigmoid(custom_w_loss).mean().item()
            elif model == "naive" :
                metrics['GAN_loss'] = torch.sigmoid(naive_loss).mean().item()
            
            train_loop.set_postfix(**metrics)

def validation_fn(
        disc, gen, loader, opt_disc, opt_gen, l1_loss, bce_loss, g_scaler, d_scaler, model, extraction_model = None, extraction_loss = None, naive_model = None, naive_loss = None
):
    gen.eval()
    disc.eval()
    if model == "watermark":
        extraction_model.eval()
    elif model == "naive":
        naive_model.eval()

    val_loop = tqdm(loader, leave=True)
    total_gen_loss = 0
    total_disc_loss = 0

    num_batches = len(loader)

    with torch.no_grad():  # No gradient updates during validation
        for idx, (x, y) in enumerate(val_loop):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Generate fake images
            y_fake = gen(x)

            # Discriminator on real and fake images
            D_real = disc(x, y)
            D_fake = disc(x, y_fake)

            # Compute losses for the discriminator
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            total_disc_loss += D_loss.item()

            # Compute losses for the generator
            G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
            total_gen_loss += G_loss.item()

            if idx % 10 == 0:
                val_loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                    G_loss=G_loss.item(),
                )

    # Return average losses over the validation set
    avg_gen_loss = total_gen_loss / len(loader)
    avg_disc_loss = total_disc_loss / len(loader)

    gen.train()  # Set generator back to training mode
    disc.train()  # Set discriminator back to training mode

    return avg_gen_loss, avg_disc_loss



def main():
    print(f"Training model {config.MODEL} on dataset {config.DATASET}")
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, out_channels=3).to(config.DEVICE)
    extraction_model = None
    naive_model = None
    watermarking_loss = None
    if config.MODEL == "watermark":
        extraction_model = WatermarkExtractionModel().to(config.DEVICE)
        watermarking_loss = CustomWatermarkLoss()
    elif config.MODEL == "naive":
        naive_model = None # TODO: Initialize the naive model
        watermarking_loss = None
    
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()

    if(config.DATASET == "segmentation"):
        dataset = SegmentationDataset(root_dir=config.TRAIN_DIR)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = MapDataset(root_dir=config.TRAIN_DIR) if config.DATASET == "maps" else FlowersDataset(root_dir=config.TRAIN_DIR)
        val_dataset = MapDataset(root_dir=config.VAL_DIR) if config.DATASET == "maps" else FlowersDataset(root_dir=config.VAL_DIR)


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1, BCE, g_scaler, d_scaler, config.MODEL, extraction_model, watermarking_loss, naive_model, naive_loss
        )

        # Perform validation at the end of each epoch
        val_gen_loss, val_disc_loss = validation_fn(
            disc, gen, val_loader, opt_disc, opt_gen, L1, BCE, g_scaler, d_scaler, config.MODEL, extraction_model, watermarking_loss, naive_model, naive_loss
        )

        print(f"Epoch: {epoch} | Generator Loss: {val_gen_loss} | Discriminator Loss: {val_disc_loss}")

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=f"gen_{epoch}.pth.tar")
            save_checkpoint(disc, opt_disc, filename=f"disc_{epoch}.pth.tar")

        save_some_examples(gen, val_loader, epoch, folder="evaluation", model=config.MODEL)
        save_watermark_extraction(gen, extraction_model, val_loader, epoch, folder="evaluation", model=config.MODEL)