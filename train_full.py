import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, load_watermark_image, generate_zero_watermark, save_watermark_extraction
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from watermark_extraction import WatermarkExtractionModel  # Import the extraction model
from losses import CustomWatermarkLoss  # Assuming you implemented the loss in a separate file
from generate_key import load_key, generate_incorrect_keys_batch

torch.backends.cudnn.benchmark = True


def train_fn_with_watermark_extraction(
    disc, gen, extraction_model, loader, opt_disc, opt_gen, l1_loss, bce, custom_loss, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        print("Training Discriminator")
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        print("Training Generator")
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

            # Calculate the custom watermark loss
            # Note: Adjust the inputs to the CustomWatermarkLoss as per your requirement.
            custom_w_loss = custom_loss(
                G=gen,
                E=extraction_model,
                S1=x,  
                S2=y,  
                k=load_key(),
                w=load_watermark_image(height=64, width=64, filepath="dataset/watermark/watermark.jpg"),
                wz=generate_zero_watermark(height=64, width=64),
                incorrect_keys=generate_incorrect_keys_batch("secret_key.pt", 8, x.size(0)),  
            )
            G_loss += custom_w_loss
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                custom_loss=torch.sigmoid(custom_w_loss).mean().item(),  # Log the custom watermark loss
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    extraction_model = WatermarkExtractionModel().to(config.DEVICE)  # Initialize watermark extraction model
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(list(gen.parameters()) + list(extraction_model.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    custom_loss = CustomWatermarkLoss(alpha=1.0, beta=0.5, gamma=0.3, p=2)  # Initialize the custom watermark loss

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn_with_watermark_extraction(
            disc, gen, extraction_model, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, custom_loss, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and epoch % 100 == 0 or epoch == config.NUM_EPOCHS - 1:
            save_checkpoint(gen, opt_gen, filename=f"checkpoints/_epoch{epoch}_{config.CHECKPOINT_GEN}")
            save_checkpoint(disc, opt_disc, filename=f"checkpoints/_epoch{epoch}_{config.CHECKPOINT_DISC}")
            save_checkpoint(extraction_model, opt_gen, filename=f"checkpoints/_epoch{epoch}_extraction_model.pth.tar")

        save_some_examples(gen, val_loader, epoch, folder="evaluation")
        save_watermark_extraction(extraction_model, epoch, val_loader, load_key(), folder="evaluation")


if __name__ == "__main__":
    main()