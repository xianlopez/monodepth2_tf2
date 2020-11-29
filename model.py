import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda

height = 192
width = 640


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


def build_depth_net(inputs):
    # inputs: (height, width, 3 * 3)
    # Take only the image in the middle, which is the current one:
    current_img = Lambda(lambda x: x[..., 3:6], output_shape=(height, width, 3))(inputs)
    # The encoder is a ResNet:
    resnet50 = tf.keras.applications.ResNet50(input_tensor=current_img, include_top=False)
    conv1_out = resnet50.get_layer('conv1_relu').output  # (96, 320, 64)
    conv2_out = resnet50.get_layer('conv2_block3_out').output  # (48, 160, 256)
    conv3_out = resnet50.get_layer('conv3_block4_out').output  # (24, 80, 512)
    conv4_out = resnet50.get_layer('conv4_block6_out').output  # (12, 40, 1024)
    conv5_out = resnet50.get_layer('conv5_block3_out').output  # (6, 20, 2048)

    # Decoder:
    x = upsampling_block(conv5_out, 1, 256, conv4_out, False)  # (12, 40, 256)
    x, disp3 = upsampling_block(x, 2, 128, conv3_out, True)  # (24, 80, 128)
    x, disp2 = upsampling_block(x, 3, 64, conv2_out, True)  # (48, 160, 64)
    x, disp1 = upsampling_block(x, 4, 32, conv1_out, True)  # (96, 320, 32)
    x, disp0 = upsampling_block(x, 5, 16, None, True)  # (192, 640, 16)

    # # Upsample the disparities:
    # disp1 = tf.keras.layers.UpSampling2D(size=(2, 2), name="disp1_up")(disp1)
    # disp2 = tf.keras.layers.UpSampling2D(size=(4, 4), name="disp2_up")(disp2)
    # disp3 = tf.keras.layers.UpSampling2D(size=(8, 8), name="disp3_up")(disp3)

    disparities = [disp0, disp1, disp2, disp3]

    return tf.keras.Model(inputs=inputs, outputs=disparities, name='depth_net')


def build_pose_net(inputs):
    # inputs: (height, width, 3 * 3)
    num_input_frames = 3
    x = Conv2D(16, 7, strides=2, padding='same', activation='relu', name='pose_conv1')(inputs)  # (96, 320, 16)
    x = Conv2D(32, 5, strides=2, padding='same', activation='relu', name='pose_conv2')(x)  # (48, 160, 32)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='pose_conv3')(x)  # (24, 80, 64)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu', name='pose_conv4')(x)  # (12, 40, 128)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='pose_conv5')(x)  # (6, 20, 256)
    # TODO: From here on it would make more sense to me to remove strides and padding
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='pose_conv6')(x)  # (3, 10, 256)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='pose_conv7')(x)  # (2, 5, 256)
    x = Conv2D(6 * (num_input_frames - 1), 1, name='pose_conv8')(x)  # (2, 5, 6 * (num_input_frames - 1))
    x = tf.keras.layers.GlobalAveragePooling2D(name='pose_avg')(x)  # (6 * (num_input_frames - 1))
    return tf.keras.Model(inputs=inputs, outputs=[x])


def training_model():
    inputs = tf.keras.Input(shape=(height, width, 3 * 3), name='input')
    pose_net = build_pose_net(inputs)  # (6 * (3 - 1))
    depth_net = build_depth_net(inputs)  # [disp0, disp1, disp2, disp3]
    outputs = pose_net.outputs
    outputs.extend(depth_net.outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
