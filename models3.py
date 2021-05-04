import tensorflow as tf
from resnet.models import resnet_layer_simple
from tensorflow.keras import layers, Sequential, Model, Input, utils
from tensorflow.keras.regularizers import l2

l2_reg = 1e-4


def reset18_multi_image_encoder(height, width, num_images, name):
    input_tensor = Input(shape=(height, width, 3 * num_images))
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn')(x)
    x = layers.ReLU(name='layer1_relu')(x)
    outputs = [x]  # (batch_size, h/2, w/2, 64)
    x = layers.MaxPool2D(name='layer2_pool')(x)
    x = resnet_layer_simple(x, 2, False, 2)
    outputs.append(x)  # (batch_size, h/4, w/4, 64)
    x = resnet_layer_simple(x, 2, True, 3)
    outputs.append(x)  # (batch_size, h/8, w/8, 128)
    x = resnet_layer_simple(x, 2, True, 4)
    outputs.append(x)  # (batch_size, h/16, w/16, 256)
    x = resnet_layer_simple(x, 2, True, 5)
    outputs.append(x)  # (batch_size, h/32, w/32, 512)
    return Model(inputs=input_tensor, outputs=outputs, name=name)


def build_pose_net(height, width):
    encoder = reset18_multi_image_encoder(height, width, 2, 'ResNet18MultiImage')
    x = encoder.outputs[-1]  # (batch_size, h/32, w/32, 512)
    x = layers.Conv2D(256, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(6, 1)(x)  # (batch_size, h/32, w/32, 6)
    x = tf.reduce_mean(x, axis=[1, 2])  # (batch_size, 6)
    x = x * 0.01
    return Model(inputs=encoder.input, outputs=x, name="pose_net")


def upsampling_block(x, num_outputs, previous_features):
    x = layers.Conv2D(num_outputs, 3, padding='same', activation='elu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    if previous_features is not None:
        x = tf.concat([x, previous_features], axis=-1)
    x = layers.Conv2D(num_outputs, 3, padding='same', activation='elu')(x)
    return x


def build_depth_net(height, width, pretrained_weights_path):
    encoder = reset18_multi_image_encoder(height, width, 1, 'ResNet18')

    read_result = encoder.load_weights(pretrained_weights_path)
    read_result.assert_existing_objects_matched()

    x = encoder.outputs[-1]  # (batch_size, h/32, w/32, 512)

    x = upsampling_block(x, 256, encoder.output[-2])  # (batch_size, h/16, w/16, 512)
    disparities = [layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)]  # (bs, h/16, w/16, 1)

    x = upsampling_block(x, 128, encoder.output[-3])  # (batch_size, h/8, w/8, 128)
    disparities.append(layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x))  # (bs, h/8, w/8, 1)

    x = upsampling_block(x, 64, encoder.output[-4])  # (batch_size, h/4, w/4, 64)
    disparities.append(layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x))  # (bs, h/4, w/4, 1)

    x = upsampling_block(x, 32, encoder.output[-5])  # (batch_size, h/2, w/2, 32)
    disparities.append(layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x))  # (bs, h/2, w/2, 1)

    x = upsampling_block(x, 16, None)  # (batch_size, h, w, 16)
    disparities.append(layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x))  # (bs, h, w, 1)

    return Model(inputs=encoder.input, outputs=disparities, name="depth_net")
