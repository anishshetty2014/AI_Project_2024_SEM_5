# -*- coding: utf-8 -*-

import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf

from utils import *
from resnet import *  # Import from the fixed ResNet implementation

open_num = 0

class RafDataset(tf.keras.utils.Sequence):
    def __init__(self, args, phase, batch_size=512, basic_aug=True):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.batch_size = batch_size
        df = pd.read_csv(args.label_path, sep=' ', header=None)

        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            dataset = df[df[name_c].str.startswith('test')]

        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values

        #### create open set: open_num is the class number indicating open class     
        openidx = [i for i in range(len(self.label)) if self.label[i] != open_num]
        self.label = np.array(self.label)[np.array(openidx)]
        
        new_label = [self.label[j] - 1 if self.label[j] >= open_num else self.label[j] for j in range(len(openidx))]
        self.label = np.array(new_label)
        images_names = np.array(images_names)[np.array(openidx)]

        #### end creating open set
        
        self.aug_func = [flip_image, add_g]
        self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f"{f.split('.')[0]}_aligned.jpg") for f in images_names]

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        images, labels = [], []

        for i in batch_indices:
            label = self.label[i]
            image = cv2.imread(self.file_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.phase == 'train':
                if self.basic_aug and random.uniform(0, 1) > 0.5:
                    index = random.randint(0, 1)
                    image = self.aug_func[index](image)
            images.append(image)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        return images, labels

class Flatten(tf.keras.layers.Layer):
    def call(self, input):
        return tf.keras.layers.Flatten()(input)

class ResNet18Feature(tf.keras.Model):
    def __init__(self, pretrained=True, num_classes=6, drop_rate=0.4, out_dim=64):
        super(ResNet18Feature, self).__init__()
        
        # Use the new ResNet18 implementation directly
        self.features = ResNet18(num_classes=1000, include_top=False)  
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x, target, phase='train'):
        x = self.features(x)  # Pass through the feature extractor (ResNet18)
        x = tf.keras.layers.Flatten()(x)
        output = self.fc(x)
        return output

def train():
    setup_seed(0)
    res18 = ResNet18Feature()
    
    data_transforms = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomRotation(0.1)
    ])
    
    train_dataset = RafDataset(args, phase='train', batch_size=args.batch_size)
    test_dataset = RafDataset(args, phase='test', batch_size=args.batch_size)

    train_loader = tf.data.Dataset.from_generator(
        lambda: train_dataset,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.experimental.AUTOTUNE)

    test_loader = tf.data.Dataset.from_generator(
        lambda: test_dataset,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.experimental.AUTOTUNE)

    res18.build(input_shape=(None, 224, 224, 3))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    best_acc = 0

    for epoch in range(args.epochs):
        # Training Loop
        for imgs, labels in train_loader:
            with tf.GradientTape() as tape:
                outputs = res18(imgs, labels, phase='train')
                loss = loss_fn(labels, outputs)
            gradients = tape.gradient(loss, res18.trainable_variables)
            optimizer.apply_gradients(zip(gradients, res18.trainable_variables))

        # Validation Loop
        total_correct = 0
        total_loss = 0.0
        for imgs, labels in test_loader:
            outputs = res18(imgs, labels, phase='test')
            loss = loss_fn(labels, outputs)
            total_loss += loss
            predictions = tf.argmax(outputs, axis=1)
            total_correct += tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32))

        acc = total_correct / len(test_loader)

        if acc > best_acc:
            best_acc = acc
            res18.save_weights("model_weights.h5")

        print(f'Epoch: {epoch + 1}, Train Loss: {loss.numpy():.4f}, Test Acc: {acc:.4f}')

    print('Best Accuracy:', best_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./DATASET', help='raf_dataset_path')
    parser.add_argument('--pretrained_backbone_path', type=str, default='./checkpoint/resnet18_msceleb.h5', help='pretrained_backbone_path')
    parser.add_argument('--label_path', type=str, default='./DATASET/train_labels.csv', help='label_path')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--out_dimension', type=int, default=512, help='feature dimension')
    args = parser.parse_args()

    train()
