import os
import shutil
import random
from math import floor

# Set the seed for reproducibility
random.seed(42)

# Path to the dataset
dataset_dir = 'flowers_dataset/jpg'  # Change this to your dataset path
output_dir = 'flowers_split'  # Change this to your desired output directory

# List all image files in the dataset
all_images = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

# Shuffle the dataset to ensure randomness
random.shuffle(all_images)

# Total number of images
total_images = len(all_images)

# Define split proportions
train_split = 0.8
val_split = 0.1
test_split = 0.1

# Calculate the number of images for each split
train_size = floor(train_split * total_images)
val_size = floor(val_split * total_images)
test_size = total_images - train_size - val_size  # Remaining for the test set

# Split the images
train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]

# Create output directories for each split
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to copy images to the respective folder
def copy_images(image_list, src_dir, dst_dir):
    for image in image_list:
        shutil.copy(os.path.join(src_dir, image), os.path.join(dst_dir, image))

# Copy images to the corresponding directories
copy_images(train_images, dataset_dir, train_dir)
copy_images(val_images, dataset_dir, val_dir)
copy_images(test_images, dataset_dir, test_dir)

print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")