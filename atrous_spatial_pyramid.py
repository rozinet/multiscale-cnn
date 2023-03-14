from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Dropout
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Concatenate, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.utils import plot_model

def atrous_spatial_pyramid_pooling(inputs):
    # Define the output feature map size based on the input size
    input_shape = int_shape(inputs)
    input_height = input_shape[1]
    input_width = input_shape[2]
    # Define the number of filters for the 1x1 convolution layers
    filters = 256
    # Define the dilation rates for the atrous convolutions
    dilation_rates = [1, 6, 12, 18]
    
    # Apply 1x1 convolution layer to the input
    x = Conv2D(filters, kernel_size=1, padding='same', activation='relu')(inputs)
    
    # Apply atrous convolutions with different dilation rates and concatenate the results
    convs = []
    for dilation_rate in dilation_rates:
        conv = Conv2D(filters, kernel_size=3, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
        convs.append(conv)
    x = Concatenate()(convs)
    
    # Apply a 1x1 convolution layer and average pooling to the concatenated feature map
    x = Conv2D(filters, kernel_size=1, padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(input_height, input_width))(x)
    
    # Apply a 1x1 convolution layer to the input and upsample the result
    y = Conv2D(filters, kernel_size=1, padding='same', activation='relu')(inputs)
    y = UpSampling2D(size=(input_height, input_width), interpolation='bilinear')(y)
    
    # Concatenate the two feature maps and apply a 1x1 convolution layer
    x = Concatenate()([x, y])
    x = Conv2D(filters, kernel_size=1, padding='same', activation='relu')(x)
    
    return x

# Define the input shape
input_shape = (256, 256, 3)

# Define the input layer
inputs = Input(input_shape)

# Apply the ASPP module to the input layer
aspp = atrous_spatial_pyramid_pooling(inputs)

# Define the output layer
outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(aspp)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Visualize the model architecture
plot_model(model, show_shapes=True, to_file='aspp.png')