import tensorflow as tf

def ACLoss(att_map1, att_map2, grid_l, output):
    # Expand grid_l to match the batch size of the output
    flip_grid_large = tf.tile(grid_l, [tf.shape(output)[0], 1, 1, 1])
    
    # No need for Variable, use tf.stop_gradient for not requiring gradient updates
    flip_grid_large = tf.stop_gradient(flip_grid_large)
    
    # Permute axes from (B, H, W, 2) to (B, 2, H, W) equivalent
    flip_grid_large = tf.transpose(flip_grid_large, [0, 2, 3, 1])
    
    # Perform grid sampling (resampling in TensorFlow)
    att_map2_flip = tf.keras.layers.Resampling2D(att_map2, flip_grid_large, interpolation='bilinear', crop_to_aspect_ratio=True)
    
    # Compute the mean squared error loss between att_map1 and the flipped att_map2
    flip_loss_l = tf.reduce_mean(tf.keras.losses.mean_squared_error(att_map1, att_map2_flip))
    
    return flip_loss_l
