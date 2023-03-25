from keras.layers import Conv1D, MaxPooling1D, SeparableConv1D, Activation, BatchNormalization, GlobalAveragePooling1D,\
    Reshape, concatenate, add, RepeatVector, DepthwiseConv1D, GlobalAveragePooling1D, Reshape, Dense, Multiply, Add

from keras import backend as K

# Multiscale block with different filter sizes:
def multiscale_block_different_filters(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(input_tensor)

    # convolutional layer with length 5 filters
    conv2 = Conv1D(32, 5, activation='relu', padding='same')(input_tensor)

    # convolutional layer with length 7 filters
    conv3 = Conv1D(32, 7, activation='relu', padding='same')(input_tensor)

    # concatenate the feature maps from the 3 layers
    concat = concatenate([conv1, conv2, conv3])

    return concat

# Multiscale block with different pooling sizes
def multiscale_block_different_pooling(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(input_tensor)

    # max pooling layer with 2x2 pool size
    pool1 = MaxPooling1D(2)(conv1)

    # convolutional layer with length 3 filters
    conv2 = Conv1D(32, 3, activation='relu', padding='same')(pool1)

    # max pooling layer with length 3 pool size
    pool2 = MaxPooling1D(3)(conv2)

    # concatenate the feature maps from the 2 pooling layers
    concat = concatenate([pool1, pool2])

    return concat


# Multiscale block with dilated convolution
def multiscale_block_dilated_convolution(input_tensor):
    # convolutional layer with length 3 filters and dilated convolution
    conv1 = Conv1D(32, 3, activation='relu', padding='same', dilation_rate=2)(input_tensor)

    # convolutional layer with length 5 filters
    conv2 = Conv1D(32, 5, activation='relu', padding='same', dilation_rate=2)(input_tensor)

    conv3 = Conv1D(32, 7, activation='relu', padding='same', dilation_rate=2)(input_tensor)

    # concatenate the feature maps from the 2 layers
    concat = concatenate([conv1, conv2, conv3])

    return concat

## Multiscale block with depth-wise separable convolution
def multiscale_block_depthwise_convolution(input_tensor):
    # depthwise separable convolutional layer with length 3 filters
    conv1 = SeparableConv1D(32, 3, activation='relu', padding='same')(input_tensor)

    # depthwise separable convolutional layer with length 5 filters
    conv2 = SeparableConv1D(32, 5, activation='relu', padding='same')(input_tensor)

    # depthwise separable convolutional layer with length 5 filters
    conv3 = SeparableConv1D(32, 7, activation='relu', padding='same')(input_tensor)

    # concatenate the feature maps from the 2 layers
    concat = concatenate([conv1, conv2, conv3])

    return concat


# Multiscale block with inception module
def multiscale_block_inception_convolution(input_tensor):

    # inception module with 1x1, length 3, and length 5 filters
    conv1 = Conv1D(16, 1, activation='relu', padding='same')(input_tensor)
    conv3 = Conv1D(16, 3, activation='relu', padding='same')(input_tensor)
    conv5 = Conv1D(16, 5, activation='relu', padding='same')(input_tensor)

    # max pooling layer with length 3 pool size
    pool = MaxPooling1D(3, strides=1, padding='same')(input_tensor)

    # concatenate the feature maps from the 4 layers
    concat = concatenate([conv1, conv3, conv5, pool])

    return concat


# Multiscale block with deeper network
def multiscale_block_deeper_network(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(input_tensor)
    conv2 = Conv1D(32, 3, activation='relu', padding='same')(conv1)

    # convolutional layer with length 5 filters
    conv3 = Conv1D(32, 5, activation='relu', padding='same')(input_tensor)
    conv4 = Conv1D(32, 5, activation='relu', padding='same')(conv3)

    # max pooling layer with 2 pool size
    pool = MaxPooling1D(2)(input_tensor)

    # concatenate the feature maps from the 4 layers
    concat = concatenate([conv1, conv2, conv3, conv4, pool])

    return concat


def base_model(input_tensor):

    # 1-D convolution and global max-pooling
    convolved = Conv1D(32, 3, padding="same", activation="tanh")(input_tensor)

    convolved = Conv1D(32, 5, padding="same", activation="tanh")(convolved)

    convolved = Conv1D(32, 7, padding="same", activation="tanh")(convolved)

    return convolved

def base_model_plus_block(input_tensor, block_func):

    concatenated = block_func(input_tensor)

    if block_func == base_model:
        return concatenated

    convolved = base_model(concatenated)

    return convolved


# Multiscale block with residual connections
def multiscale_block_residual_connections(input_tensor):

    # convolutional layer with length 3 filters and residual connection
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(input_tensor)
    conv2 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
    residual = add([input_tensor, conv2])

    # convolutional layer with length 5 filters and residual connection
    conv3 = Conv1D(32, 5, activation='relu', padding='same')(residual)
    conv4 = Conv1D(32, 5, activation='relu', padding='same')(conv3)
    residual2 = add([residual, conv4])

    # convolutional layer with length 7 filters and residual connection
    conv5 = Conv1D(32, 7, activation='relu', padding='same')(residual2)
    conv6 = Conv1D(32, 7, activation='relu', padding='same')(conv5)
    residual3 = add([residual2, conv6])

    concat = concatenate([residual, residual2, residual3])

    return concat

# Multiscale block with residual connections deeper
def multiscale_block_residual_connections_deeper(input_tensor):
    # First set of convolutional layers with length 3 filters and residual connection
    conv1 = Conv1D(32, 3, padding='same')(input_tensor)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(32, 3, padding='same')(conv1)
    conv2 = Activation('relu')(conv2)
    residual = add([input_tensor, conv2])

    # Second set of convolutional layers with length 3 filters and residual connection
    conv3 = Conv1D(32, 3, padding='same')(residual)
    conv3 = Activation('relu')(conv3)
    conv4 = Conv1D(32, 3, padding='same')(conv3)
    conv4 = Activation('relu')(conv4)
    residual2 = add([residual, conv4])

    # Third set of convolutional layers with length 5 filters and residual connection
    conv5 = Conv1D(32, 5, padding='same')(residual2)
    conv5 = Activation('relu')(conv5)
    conv6 = Conv1D(32, 5, padding='same')(conv5)
    conv6 = Activation('relu')(conv6)
    residual3 = add([residual2, conv6])

    # Fourth set of convolutional layers with length 5 filters and residual connection
    conv7 = Conv1D(32, 5, padding='same')(residual3)
    conv7 = Activation('relu')(conv7)
    conv8 = Conv1D(32, 5, padding='same')(conv7)
    conv8 = Activation('relu')(conv8)
    residual4 = add([residual3, conv8])

    # Fifth set of convolutional layers with length 7 filters and residual connection
    conv9 = Conv1D(32, 7, padding='same')(residual4)
    conv9 = Activation('relu')(conv9)
    conv10 = Conv1D(32, 7, padding='same')(conv9)
    conv10 = Activation('relu')(conv10)
    residual5 = add([residual4, conv10])

    # Concatenate all residual connections
    concat = concatenate([residual, residual2, residual3, residual4, residual5])

    return concat


def residual_block(input_tensor, filters, kernel_size, strides=1):
    shortcut = input_tensor

    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if strides > 1 or input_tensor.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        shortcut = Activation('relu')(shortcut)

    output = add([x, shortcut])
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


# Multiscale block with residual connections deeper, dilated, batch-normalized, varying number of filters
def multiscale_block_residual_connections_deeper2(input_tensor):
    # First set of convolutional layers with length 3 filters and residual connection
    residual1 = residual_block(input_tensor, filters=16, kernel_size=3)

    # Second set of convolutional layers with length 3 filters and residual connection
    residual2 = residual_block(residual1, filters=32, kernel_size=3)

    # Third set of convolutional layers with length 5 filters and residual connection
    residual3 = residual_block(residual2, filters=64, kernel_size=5)

    # Fourth set of convolutional layers with length 5 filters and residual connection
    residual4 = residual_block(residual3, filters=128, kernel_size=5)

    # Fifth set of convolutional layers with length 7 filters and residual connection
    residual5 = residual_block(residual4, filters=256, kernel_size=7)

    # Concatenate all residual connections
    concat = concatenate([residual1, residual2, residual3, residual4, residual5])

    return concat

# Multiscale block with residual connections and bottleneck architecture
def multiscale_block_residual_connections_bottleneck_architecture(input_tensor):
    # convolutional layer with 1x1 filters and bottleneck architecture
    conv1 = Conv1D(8, 1)(input_tensor)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    # convolutional layer with length 3 filters and bottleneck architecture
    conv2 = Conv1D(16, 3, strides=2, padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    # convolutional layer with 1x1 filters and bottleneck architecture
    conv3 = Conv1D(32, 1)(act2)
    bn3 = BatchNormalization()(conv3)

    # residual connection
    residual = Conv1D(32, 1, strides=2)(input_tensor)
    bn_res = BatchNormalization()(residual)

    # concatenate the feature maps from the 2 layers
    concat = concatenate([bn3, bn_res])
    output = Activation('relu')(concat)

    return output


# Multiscale block with pyramid pooling
def multiscale_block_pyramid_pooling(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(input_tensor)

    # average pooling layers with different pool sizes
    pool1 = GlobalAveragePooling1D()(input_tensor)
    pool1 = Reshape((1, -1))(pool1)
    pool1 = Conv1D(32, 1, activation='relu')(pool1)

    pool2 = GlobalAveragePooling1D()(conv1)
    pool2 = Reshape((-1, 1))(pool2)
    pool2 = Conv1D(32, 1, activation='relu')(pool2)

    # repeat pool1 to match the length of conv1
    pool1 = RepeatVector(K.int_shape(conv1)[1])(pool1)

    # concatenate the feature maps from the 3 layers
    concat = concatenate([conv1, pool1, pool2], axis=-1)

    return concat


## Scale-Specific Filters
def scale_specific_filters(input_tensor):
    # Convolutional layers with scale-specific filters
    conv1_1 = Conv1D(32, 3, activation='relu', padding='same')(input_tensor)
    conv1_2 = Conv1D(32, 5, activation='relu', padding='same')(input_tensor)
    conv1_3 = Conv1D(32, 7, activation='relu', padding='same')(input_tensor)

    # Max pooling to reduce the spatial resolution
    pool1_1 = MaxPooling1D(pool_size=2)(conv1_1)
    pool1_2 = MaxPooling1D(pool_size=2)(conv1_2)
    pool1_3 = MaxPooling1D(pool_size=2)(conv1_3)

    # Concatenate the feature maps from each scale
    merged = concatenate([pool1_1, pool1_2, pool1_3])

    return merged

# Parallel Pathway CNN
def parallel_pathway(input_tensor):
    # First pathway
    conv1_1 = Conv1D(32, 3, activation='relu', padding='same')(input_tensor)
    pool1_1 = MaxPooling1D(pool_size=2)(conv1_1)

    # Second pathway
    conv1_2 = Conv1D(32, 5, activation='relu', padding='same')(input_tensor)
    pool1_2 = MaxPooling1D(pool_size=2)(conv1_2)

    # Third pathway
    conv1_3 = Conv1D(32, 7, activation='relu', padding='same')(input_tensor)
    pool1_3 = MaxPooling1D(pool_size=2)(conv1_3)

    # Concatenate the outputs from all three pathways
    merged = concatenate([pool1_1, pool1_2, pool1_3], axis=3)

    return merged

def efficientnet(tensor_input, expansion_factor=6, se_ratio=0.25):
    strides = 1

    # Initial 1x1 pointwise convolution
    x = Conv1D(filters=32 * expansion_factor, kernel_size=1, strides=1, padding='same')(tensor_input)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # Depthwise convolution
    x = DepthwiseConv1D(kernel_size=5, strides=1, padding='same', depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # Squeeze-and-Excitation (SE) module
    se = GlobalAveragePooling1D()(x)
    se = Reshape((1, se.shape[-1]))(se)
    se = Dense(se.shape[-1] * se_ratio, activation='swish')(se)
    se = Dense(x.shape[-1], activation='sigmoid')(se)
    x = Multiply()([x, se])

    # Final 1x1 pointwise convolution
    x = Conv1D(filters=32, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the residual connection
    if tensor_input.shape[-1] == 32 and strides == 1:
        x = Add()([x, tensor_input])

    return x


def efficientnet_v2(tensor_input, num_filters = 32, kernel_size = 3, strides = 1, expansion_factor=6, se_ratio=0.25, fused=True):
    # Fused-MBConv block
    if fused:
        x = Conv1D(filters=num_filters * expansion_factor, kernel_size=kernel_size, strides=strides, padding='same')(tensor_input)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
    else:
        # Initial 1x1 pointwise convolution
        x = Conv1D(filters=num_filters * expansion_factor, kernel_size=1, strides=1, padding='same')(tensor_input)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

        # Depthwise convolution
        x = DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding='same', depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

    # Squeeze-and-Excitation (SE) module
    se = GlobalAveragePooling1D()(x)
    se = Reshape((1, se.shape[-1]))(se)
    se = Dense(se.shape[-1] * se_ratio, activation='swish')(se)
    se = Dense(x.shape[-1], activation='sigmoid')(se)
    x = Multiply()([x, se])

    # Final 1x1 pointwise convolution
    x = Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the residual connection
    if tensor_input.shape[-1] == num_filters and strides == 1:
        x = Add()([x, tensor_input])

    return x


def build_efficientnet_v2(input_tensor, num_layers = 5):
    x = input_tensor

    # Increase the depth of the network by stacking layers
    for i in range(num_layers):
        x = efficientnet_v2(x, num_filters=32, kernel_size=3, strides=1, expansion_factor=6, se_ratio=0.25, fused=True)

    return x

