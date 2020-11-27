import tensorflow as tf
from tensorflow.keras.layers import Conv2D


def upsampling_block(x, idx, nchannels, skip_connection, compute_disp):
    # TODO: Using 1 output channel in the first convolution could save a lot of parameters
    x = Conv2D(nchannels, 3, padding='same', activation='elu', name='dec_b' + str(idx) + '_conv1')(x)
    x = tf.keras.layers.UpSampling2D(name='dec_b' + str(idx) + '_up')(x)
    if skip_connection is not None:
        x = tf.keras.layers.Concatenate(name='dec_b' + str(idx) + '_concat')([x, skip_connection])
    x = Conv2D(nchannels, 3, padding='same', activation='elu', name='dec_b' + str(idx) + '_conv2')(x)
    if compute_disp:
        disp = Conv2D(1, 3, padding='same', activation='sigmoid', name='dec_b' + str(idx) + '_disp')(x)
        return x, disp
    else:
        return x


def build_depth_net():
    # The encoder is a ResNet:
    resnet50 = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))
    conv1_out = resnet50.get_layer('conv1_relu').output  # (112, 112, 64)
    conv2_out = resnet50.get_layer('conv2_block3_out').output  # (56, 56, 256)
    conv3_out = resnet50.get_layer('conv3_block4_out').output  # (28, 28, 512)
    conv4_out = resnet50.get_layer('conv4_block6_out').output  # (14, 14, 1024)
    conv5_out = resnet50.get_layer('conv5_block3_out').output  # (7, 7, 2048)
    encoder = tf.keras.Model(inputs=resnet50.input,
                             outputs=[conv1_out, conv2_out, conv3_out, conv4_out, conv5_out],
                             name='depth_encoder')

    # Decoder:
    x = upsampling_block(conv5_out, 1, 256, conv4_out, False)  # (14, 14, 256)
    x, disp3 = upsampling_block(x, 2, 128, conv3_out, True)  # (28, 28, 128)
    x, disp2 = upsampling_block(x, 3, 64, conv2_out, True)  # (56, 56, 64)
    x, disp1 = upsampling_block(x, 4, 32, conv1_out, True)  # (112, 112, 32)
    x, disp0 = upsampling_block(x, 5, 16, None, True)  # (224, 224, 16)

    # # Upsample the disparities:
    # disp1 = tf.keras.layers.UpSampling2D(size=(2, 2), name="disp1_up")(disp1)
    # disp2 = tf.keras.layers.UpSampling2D(size=(4, 4), name="disp2_up")(disp2)
    # disp3 = tf.keras.layers.UpSampling2D(size=(8, 8), name="disp3_up")(disp3)

    disparities = [disp0, disp1, disp2, disp3]

    return tf.keras.Model(inputs=encoder.input, outputs=disparities, name='depth_net')


def build_pose_net(num_input_frames=3):
    inputs = tf.keras.Input(shape=(224, 224, 3 * num_input_frames), name='pose_input')
    x = Conv2D(16, 7, strides=2, padding='same', activation='relu', name='pose_conv1')(inputs)  # (112, 112, 16)
    x = Conv2D(32, 5, strides=2, padding='same', activation='relu', name='pose_conv2')(x)  # (56, 56, 32)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='pose_conv3')(x)  # (28, 28, 64)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu', name='pose_conv4')(x)  # (14, 14, 128)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='pose_conv5')(x)  # (7, 7, 256)
    # TODO: From here on it would make more sense to me to remove strides and padding
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='pose_conv6')(x)  # (4, 4, 256)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='pose_conv7')(x)  # (2, 2, 256)
    x = Conv2D(6 * (num_input_frames - 1), 1, name='pose_conv8')(x)  # (2, 2, 6 * (num_input_frames - 1))
    x = tf.keras.layers.GlobalAveragePooling2D(name='pose_avg')(x)  # (6 * (num_input_frames - 1))
    return tf.keras.Model(inputs=inputs, outputs=[x])


print('')
print('depth_net')
depth_net = build_depth_net()
depth_net.summary()

print('')
print('pose_net')
pose_net = build_pose_net()
pose_net.summary()
