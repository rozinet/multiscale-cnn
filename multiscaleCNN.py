import matplotlib.pyplot as plt
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D
from keras.models import Model

from typing import List
from keras.callbacks import EarlyStopping

# Utility module
from utils import load_data, get_multiscale_lengths

# Inputs downscale/donwsample module
from input_processor import downsample_bilateral, downsample_gaussian, downsample_mean, downsample_subsampling

(X_train, y_train) = load_data('AbnormalHeartbeat_TRAIN.arff')
(X_test, y_test) = load_data('AbnormalHeartbeat_TEST.arff')



# it takes a time series as an input, performs 1-D convolution, and returns it as an output ready for concatenation
def get_base_model(input_len, fsize):
    # the input is a time series of length n and width 1
    input_seq = Input(shape=(input_len, 1))

    # choose the number of convolution filters
    nb_filters = 10

    # 1-D convolution and global max-pooling
    convolved = Conv1D(nb_filters, fsize, padding="same", activation="tanh")(input_seq)
    processed = GlobalMaxPooling1D()(convolved)

    # dense layer with dropout regularization
    compressed = Dense(50, activation="tanh")(processed)
    compressed = Dropout(0.3)(compressed)
    model = Model(inputs=input_seq, outputs=compressed)
    return model

# this is the main model
# it takes the original time series and its down-sampled versions as an input, and returns the result of classification as an output
def main_model(inputs_lens: List[int], fsizes: List[int]):
    # the inputs to the branches are the original time series, and its down-sampled versions
    inputs = []
    branches = []
    for i, input_len in enumerate(inputs_lens):
        inputs.append(Input(shape=(input_len, 1)))
        base_net = get_base_model(input_len, fsizes[i])
        branches.append(base_net(inputs[-1]))

    # concatenate all the outputs
    merged = Concatenate()(branches)
    layer = Dense(50, activation='sigmoid')(merged)
    out = Dense(5, activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=out)
    return model


# the divisor of the size, 1 is mentioned for original size

inputs_functions = [downsample_subsampling] #, downsample_bilateral, downsample_gaussian, downsample_mean]

downsample_factors = [4, 2, 1] #, 6, 8]
fsizes = [8, 16, 24]

# Call each function in a loop
for input_func in inputs_functions:
    input_train_x = input_func(X_train, downsample_factors)
    input_test_x = input_func(X_test, downsample_factors)

    inputs_lens = get_multiscale_lengths(input_train_x)


    m = main_model(inputs_lens, fsizes)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    history = m.fit(input_train_x, y_train, validation_split=0.2, epochs=10000, callbacks=[early_stop])

    # plot metrics
    plt.plot(history.history['accuracy'])
    plt.show()

    score = m.evaluate(input_test_x, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])










