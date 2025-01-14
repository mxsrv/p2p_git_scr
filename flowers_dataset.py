import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image

# Dataset for Image Colorization

class FlowersDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        target_image = np.array(Image.open(img_path))

        input_image = np.mean(target_image, axis=2).astype(np.uint8)
        input_image = np.stack([input_image] * 3, axis=-1)

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image