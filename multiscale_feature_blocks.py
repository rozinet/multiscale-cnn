from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Activation, BatchNormalization, GlobalAveragePooling2D, \
    Reshape, concatenate, add


# Multiscale block with different filter sizes:
def multiscale_block_different_filters(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)

    # convolutional layer with length 5 filters
    conv2 = Conv2D(32, 5, activation='relu', padding='same')(input_tensor)

    # convolutional layer with length 7 filters
    conv3 = Conv2D(32, 7, activation='relu', padding='same')(input_tensor)

    # concatenate the feature maps from the 3 layers
    concat = concatenate([conv1, conv2, conv3])

    return concat

# Multiscale block with different pooling sizes
def multiscale_block_different_pooling(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)

    # max pooling layer with 2x2 pool size
    pool1 = MaxPooling2D(2)(conv1)

    # convolutional layer with length 3 filters
    conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)

    # max pooling layer with length 3 pool size
    pool2 = MaxPooling2D(3)(conv2)

    # concatenate the feature maps from the 2 pooling layers
    concat = concatenate([pool1, pool2])

    return concat


# Multiscale block with dilated convolution
def multiscale_block_dilated_convoution(input_tensor):
    # convolutional layer with length 3 filters and dilated convolution
    conv1 = Conv2D(32, 3, activation='relu', padding='same', dilation_rate=(2, 2))(input_tensor)

    # convolutional layer with length 5 filters
    conv2 = Conv2D(32, 5, activation='relu', padding='same')(input_tensor)

    # concatenate the feature maps from the 2 layers
    concat = concatenate([conv1, conv2])

    return concat


## Multiscale block with depth-wise separable convolution
def multiscale_block_depthwise_convolution(input_tensor):
    # depthwise separable convolutional layer with length 3 filters
    conv1 = SeparableConv2D(32, 3, activation='relu', padding='same')(input_tensor)

    # depthwise separable convolutional layer with length 5 filters
    conv2 = SeparableConv2D(32, 5, activation='relu', padding='same')(input_tensor)

    # concatenate the feature maps from the 2 layers
    concat = concatenate([conv1, conv2])

    return concat


# Multiscale block with inception module
def multiscale_block_inception_convolution(input_tensor):
    # inception module with 1x1, length 3, and length 5 filters
    conv1 = Conv2D(16, (1, 1), activation='relu', padding='same')(input_tensor)
    conv3 = Conv2D(16, 3, activation='relu', padding='same')(input_tensor)
    conv5 = Conv2D(16, 5, activation='relu', padding='same')(input_tensor)

    # max pooling layer with length 3 pool size
    pool = MaxPooling2D(3, strides=(1, 1), padding='same')(input_tensor)

    # concatenate the feature maps from the 4 layers
    concat = concatenate([conv1, conv3, conv5, pool])

    return concat


# Multiscale block with deeper network
def multiscale_block_deeper_network(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)
    conv2 = Conv2D(32, 3, activation='relu', padding='same')(conv1)

    # convolutional layer with length 5 filters
    conv3 = Conv2D(32, 5, activation='relu', padding='same')(input_tensor)
    conv4 = Conv2D(32, 5, activation='relu', padding='same')(conv3)

    # max pooling layer with 2x2 pool size
    pool = MaxPooling2D((2, 2))(input_tensor)

    # concatenate the feature maps from the 4 layers
    concat = concatenate([conv1, conv2, conv3, conv4, pool])

    return concat


# Multiscale block with residual connections
def multiscale_block_residual_connections(input_tensor):
    # convolutional layer with length 3 filters and residual connection
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)
    conv2 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    residual = add([input_tensor, conv2])

    # convolutional layer with length 5 filters and residual connection
    conv3 = Conv2D(32, 5, activation='relu', padding='same')(residual)
    conv4 = Conv2D(32, 5, activation='relu', padding='same')(conv3)
    residual2 = add([residual, conv4])

    concat = concatenate([residual, residual2])

    return concat


# Multiscale block with residual connections and bottleneck architecture
def multiscale_block_residual_connections_bottleneck_architecture(input_tensor):
    # convolutional layer with 1x1 filters and bottleneck architecture
    conv1 = Conv2D(8, (1, 1))(input_tensor)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    # convolutional layer with length 3 filters and bottleneck architecture
    conv2 = Conv2D(16, 3, strides=(2, 2), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    # convolutional layer with 1x1 filters and bottleneck architecture
    conv3 = Conv2D(32, (1, 1))(act2)
    bn3 = BatchNormalization()(conv3)

    # residual connection
    residual = Conv2D(32, (1, 1), strides=(2, 2))(input_tensor)
    bn_res = BatchNormalization()(residual)

    # concatenate the feature maps from the 2 layers
    concat = concatenate([bn3, bn_res])
    output = Activation('relu')(concat)

    return output


# Multiscale block with pyramid pooling
def multiscale_block_pyramid_pooling(input_tensor):
    # convolutional layer with length 3 filters
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)

    # average pooling layers with different pool sizes
    pool1 = GlobalAveragePooling2D()(input_tensor)
    pool1 = Reshape((1, 1, -1))(pool1)
    pool1 = Conv2D(32, (1, 1), activation='relu')(pool1)

    pool2 = GlobalAveragePooling2D()(conv1)
    pool2 = Reshape((1, 1, -1))(pool2)
    pool2 = Conv2D(32, (1, 1), activation='relu')(pool2)

    # concatenate the feature maps from the 3 layers
    concat = concatenate([conv1, pool1, pool2])

    return concat


## Scale-Specific Filters
def scale_specific_filters(input_tensor):
    # Convolutional layers with scale-specific filters
    conv1_1 = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)
    conv1_2 = Conv2D(32, 5, activation='relu', padding='same')(input_tensor)
    conv1_3 = Conv2D(32, 7, activation='relu', padding='same')(input_tensor)

    # Max pooling to reduce the spatial resolution
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    pool1_3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)

    # Concatenate the feature maps from each scale
    merged = concatenate([pool1_1, pool1_2, pool1_3], axis=3)

    return merged

# Parallel Pathway CNN
def parallel_pathway(input_tensor):
    # First pathway
    conv1_1 = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    # Second pathway
    conv1_2 = Conv2D(32, 5, activation='relu', padding='same')(input_tensor)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    # Third pathway
    conv1_3 = Conv2D(32, 7, activation='relu', padding='same')(input_tensor)
    pool1_3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)

    # Concatenate the outputs from all three pathways
    merged = concatenate([pool1_1, pool1_2, pool1_3], axis=3)

    return merged
