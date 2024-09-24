import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
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

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def validate_fn(disc, gen, loader, l1_loss, bce):
    gen.eval()  # Set generator to evaluation mode
    disc.eval()  # Set discriminator to evaluation mode

    total_gen_loss = 0
    total_disc_loss = 0
    loop = tqdm(loader, leave=True)

    with torch.no_grad():  # No gradient updates during validation
        for idx, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Generate fake images
            y_fake = gen(x)

            # Discriminator on real and fake images
            D_real = disc(x, y)
            D_fake = disc(x, y_fake)

            # Compute losses for the discriminator
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            total_disc_loss += D_loss.item()

            # Compute losses for the generator
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
            total_gen_loss += G_loss.item()

            if idx % 10 == 0:
                loop.set_postfix(
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
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

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
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 4, shuffle=True)
    scheduler_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=100, gamma=0.5)
    scheduler_gen = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=100, gamma=0.5)

    # In your training loop after each epoch
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        # Run validation
        avg_gen_loss, avg_disc_loss = validate_fn(disc, gen, val_loader, L1_LOSS, BCE)

        print(f"Epoch: {epoch}, Gen Loss: {avg_gen_loss}, Disc Loss: {avg_disc_loss}")


        scheduler_disc.step()
        scheduler_gen.step()
        
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation_map_gan")


if __name__ == "__main__":
    main()
