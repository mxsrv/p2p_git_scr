import torch
import numpy as np

import config


def generate_secret_key(h, w, t, block_size=(4, 4), seed=None):
    # Set the seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Total size of the key
    key_size = h * w * t

    # Generate a binary key of the specified size
    key = np.random.randint(0, 2, size=(h // block_size[0], w // block_size[1], t))

    # Resize the key to match the block size
    key = np.repeat(np.repeat(key, block_size[0], axis=0), block_size[1], axis=1)

    # Convert the key to a torch tensor and resize
    key_tensor = torch.tensor(key, dtype=torch.float16)

    # Reshape to h x w x t and add batch dimension
    key_tensor = key_tensor.unsqueeze(0)  # Add batch dimension (1, h, w, t)
    key_tensor = key_tensor.permute(0, 3, 1, 2)  # Change to (batch, t, h, w)

    return key_tensor

def save_key(key_tensor, filepath="secret_key.pt"):
    # Save the generated key to a file
    torch.save(key_tensor, filepath)

def load_key(filepath="secret_key.pt"):
    # Load the key from a file
    # make key half precision
    key_tensor = torch.load(filepath, map_location=config.DEVICE)
    return key_tensor



def generate_incorrect_keys_batch(key_path, Td, batch_size):
    original_key = torch.load(key_path)
    
    flat_key = original_key.flatten()
    
    num_bits = flat_key.numel()
    
    incorrect_keys_batch = torch.empty((batch_size,) + original_key.shape, dtype=original_key.dtype)
    
    # Generate incorrect keys for the entire batch
    for i in range(batch_size):
        # Clone the original key for this instance
        incorrect_key = flat_key.clone()
        
        # Determine how many bits to flip (randomly chosen up to Td)
        num_flips = torch.randint(1, Td + 1, (1,)).item()
        
        # Randomly select indices to flip
        flip_indices = torch.randperm(num_bits)[:num_flips]
        
        # Flip the selected bits
        incorrect_key[flip_indices] = 1 - incorrect_key[flip_indices]
        
        # Reshape and store the incorrect key in the batch
        incorrect_keys_batch[i] = incorrect_key.view_as(original_key)
    
    return incorrect_keys_batch.to(config.DEVICE)


