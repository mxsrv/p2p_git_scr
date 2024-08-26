import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        target_image = image[:, :600, :]
        
        # Create an input image that is the same as the target image but without colors, only black and white

        input_image = np.mean(target_image, axis=2).astype(np.uint8)
        input_image = np.stack([input_image] * 3, axis=-1)
         
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        # use imshow to visualize the images
        
        # Visualize the images using imshow
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(target_image)
        plt.title('Target Image')
        plt.axis('off')

        plt.show()

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("dataset/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
