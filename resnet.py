import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class BasicBlock(Model):
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample

    def call(self, x, training=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(x, training=training)

        out += residual
        out = self.relu(out)

        return out

class ResNet(Model):
    def __init__(self, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top

        # Initial convolutional layer
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        # Manually define the convolutional layers with specific filters
        self.conv2_x = self._make_basic_block(64, 2)
        self.conv3_x = self._make_basic_block(128, 2, stride=2)
        self.conv4_x = self._make_basic_block(256, 2, stride=2)
        self.conv5_x = self._make_basic_block(512, 2, stride=2)

        # Global average pooling layer
        self.avgpool = layers.GlobalAveragePooling2D()

        if self.include_top:
            self.fc = layers.Dense(num_classes)

    def _make_basic_block(self, filters, blocks, stride=1):
        layers_list = []

        # First block might need downsampling if stride != 1
        downsample = None
        if stride != 1:
            downsample = keras.Sequential([
                layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

        # First block with downsample
        layers_list.append(BasicBlock(filters, stride, downsample))
        
        # Remaining blocks with normal stride
        for _ in range(1, blocks):
            layers_list.append(BasicBlock(filters))
        
        return keras.Sequential(layers_list)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        # Apply the convolutional layers
        x = self.conv2_x(x, training=training)  # 64 filters
        x = self.conv3_x(x, training=training)  # 128 filters
        x = self.conv4_x(x, training=training)  # 256 filters
        x = self.conv5_x(x, training=training)  # 512 filters

        x = self.avgpool(x)

        if self.include_top:
            x = self.fc(x)

        return x

# ResNet18 model creation with explicit layer definitions
def ResNet18(num_classes=1000, include_top=True):
    return ResNet(num_classes=num_classes, include_top=include_top)

if __name__ == '__main__':
    model = ResNet18(num_classes=1000)
    input = tf.random.normal([1, 224, 224, 3])
    output = model(input)
    print("Output shape:", output.shape)  # Should output (1, 1000) if include_top is True
