import torch
import config
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x *0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(y, folder + f"/label_{epoch}.png")
    gen.train()

def save_watermark_extraction(gen, watermark_extractor, epoch, data_loader, folder, secret):
    x, y = next(iter(data_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    y_fake = gen(x)
    watermark_extractor.eval()
    with torch.no_grad():
        extracted_watermark = watermark_extractor(y_fake, secret)
        extracted_watermark = extracted_watermark * 0.5 + 0.5  # remove normalization
        save_image(extracted_watermark, folder + f"/extracted_watermark_{epoch}.png")
    watermark_extractor.train()
    gen.train()



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def load_watermark_image(height, width, filepath):
    # Load the watermark image
    watermark = Image.open(filepath).convert("RGB")
    watermark = watermark.resize((height, width))
    watermark = transforms.ToTensor()(watermark)
    watermark = watermark.unsqueeze(0).to(config.DEVICE)
    return watermark


## function that generates an all zero tensor of the same size as the watermark image
def generate_zero_watermark(height, width):
    return torch.zeros((1, 3, height, width)).to(config.DEVICE)


## test the last two functions if both have same dimensions


def test():
    height, width = 256, 256
    filepath = "dataset/watermark/watermark.jpg"
    watermark = load_watermark_image(height, width, filepath)
    zero_watermark = generate_zero_watermark(height, width)
    assert watermark.shape == zero_watermark.shape
    print("Success")



if __name__ == "__main__":
    test()