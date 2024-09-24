import torch
from utils import load_checkpoint, load_watermark_image, generate_zero_watermark, save_image
from generate_key import load_key
import config
from generator_model import Generator
from watermark_extraction import WatermarkExtractionModel
from dataset import MapDataset
from flowers_dataset import FlowersDataset
from torch.utils.data import DataLoader
import torch.optim as optim

def transform_back(image):
    return (image + 1) / 2.

def test_model():
    # Initialize models
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    extraction_model = WatermarkExtractionModel().to(config.DEVICE)
    opt_gen = optim.Adam(list(gen.parameters()) + list(extraction_model.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Load checkpoints
    load_checkpoint(f"checkpoints/_epoch500_{config.CHECKPOINT_GEN}", gen, optimizer=opt_gen, lr=config.LEARNING_RATE)
    load_checkpoint(f"checkpoints/_epoch500_extraction_model.pth.tar", extraction_model, optimizer=opt_gen, lr=config.LEARNING_RATE)
    
    # Load the validation dataset
    val_dataset = FlowersDataset(root_dir=config.TEST_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Set the number of test images to generate
    num_test_images = getattr(config, 'NO_TEST_IMAGES', 5)  # Use config.NO_TEST_IMAGES if available, otherwise default to 5

    gen.eval()
    extraction_model.eval()

    for idx, (sample_image, _) in enumerate(val_loader):
        if idx >= num_test_images:
            break
        
        sample_image = sample_image.to(config.DEVICE)

        # Generate the fake image using the generator
        with torch.no_grad():
            generated_image = gen(sample_image)

        # Extract the watermark from the generated image
        with torch.no_grad():
            extracted_watermark = extraction_model(generated_image, load_key())

        # Extract the watermark from the sample image (original, not generated)
        with torch.no_grad():
            extracted_watermark_sample = extraction_model(sample_image, load_key())

        # Print the min/max values for debugging
        print(f"Image {idx + 1}:")
        print(f"Generated Image Max: {generated_image.max()}, Min: {generated_image.min()}")
        print(f"Extracted Watermark Max: {extracted_watermark.max()}, Min: {extracted_watermark.min()}")
        print(f"Sample Image Max: {sample_image.max()}, Min: {sample_image.min()}")
        print(f"Extracted Watermark Sample Max: {extracted_watermark_sample.max()}, Min: {extracted_watermark_sample.min()}")

        # Save the output images for inspection
        save_image(transform_back(generated_image), f"output/generated_image_{idx + 1}.png")
        save_image(extracted_watermark, f"output/extracted_watermark_{idx + 1}.png")
        save_image(transform_back(sample_image), f"output/sample_image_{idx + 1}.png")
        save_image(extracted_watermark_sample, f"output/extracted_watermark_sample_{idx + 1}.png")

    print(f"Generated images and extracted watermarks for {num_test_images} samples have been saved to the output folder.")

if __name__ == "__main__":
    test_model()
