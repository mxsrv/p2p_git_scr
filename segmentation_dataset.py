# This script either loads the dataset for train/val or test
# The folders are structured as follows:
# - images (jpg files)
# - masks (bmp files)

# Create a custom dataset class that either loads the train/val or test dataset

import os
import numpy as np
import config
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, "images")
        self.masks_dir = os.path.join(self.root_dir, "masks")
        self.list_images = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, index):
        img_file = self.list_images[index]
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, img_file.replace(".jpg", ".bmp"))

        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        print(image.shape, mask.shape)
        #assert image.shape == mask.shape, f"Image and mask should be the same size, but are not. Image: {image.shape}, Mask: {mask.shape}. Filename: {img_path}"
  

        input_image = config.transform_only_input(image=image)["image"]
        target_image = config.transform_only_mask(image=mask)["image"]
        print(input_image.shape)
        print(target_image.shape) 

        return input_image, target_image