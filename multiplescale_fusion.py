from keras.layers import Input, Dense, concatenate, Dropout
from keras.models import Model
from keras.utils import plot_model

# Define the input shape for both modalities
input_shape1 = (128,)
input_shape2 = (64, 64, 3)

# Define the inputs for both modalities
input1 = Input(input_shape1, name='input1')
input2 = Input(input_shape2, name='input2')

# Define the first hidden layer for input1
hidden1_1 = Dense(64, activation='relu')(input1)
dropout1_1 = Dropout(0.5)(hidden1_1)

# Define the second hidden layer for input1
hidden2_1 = Dense(32, activation='relu')(dropout1_1)

# Define the first convolutional layer for input2
conv1_2 = Conv2D(32, (3, 3), activation='relu')(input2)

# Define the second convolutional layer for input2
conv2_2 = Conv2D(64, (3, 3), activation='relu')(conv1_2)

# Define the third convolutional layer for input2
conv3_2 = Conv2D(128, (3, 3), activation='relu')(conv2_2)

# Flatten the output of the third convolutional layer for input2
flatten_2 = Flatten()(conv3_2)

# Concatenate the outputs of the second hidden layer for input1 and the flattened output of the third convolutional layer for input2
merged = concatenate([hidden2_1, flatten_2])

# Define the output layer
output = Dense(1, activation='sigmoid')(merged)

# Define the model with two inputs and one output
model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Visualize the model architecture
plot_model(model, show_shapes=True, to_file='multiple_fusion_learning.png')