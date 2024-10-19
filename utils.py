import tensorflow as tf
import numpy as np
import random


def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return np.fliplr(image_array)

def setup_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_flip_grid(w, h, device):
    # Create grid for flipping attention maps
    x_ = tf.range(w, dtype=tf.float32)
    y_ = tf.range(h, dtype=tf.float32)
    
    x_ = tf.expand_dims(x_, 0)
    y_ = tf.expand_dims(y_, 1)
    
    x_ = tf.tile(x_, [h, 1])
    y_ = tf.tile(y_, [1, w])
    
    grid = tf.stack([x_, y_], axis=0)
    grid = tf.expand_dims(grid, 0)  # Unsqueeze for batch dimension
    grid = tf.tile(grid, [1, 1, 1, 1])  # Expand for batch size
    
    grid = tf.cast(grid, tf.float32)
    grid = tf.convert_to_tensor(grid)

    # Normalize the grid
    grid_x = 2 * grid[0, 0, :, :] / (w - 1) - 1
    grid_y = 2 * grid[0, 1, :, :] / (h - 1) - 1
    
    grid_x = -grid_x  # Flip horizontally
    
    grid = tf.stack([grid_x, grid_y], axis=0)
    grid = tf.expand_dims(grid, 0)
    
    return grid
