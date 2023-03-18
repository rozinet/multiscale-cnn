import matplotlib.pyplot as plt
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D
from keras.models import Model

from typing import List
from keras.callbacks import EarlyStopping

import statistics

# Utility module
from utils import load_data, get_multiscale_lengths

# Inputs downscale/donwsample module
from input_processor import downsample_guided, downsample_nl_means, downsample_wavelet, downsample_bilateral, \
    downsample_gaussian, downsample_mean, downsample_subsampling

# Multiscale features block
from multiscale_feature_blocks import base_model, multiscale_block_different_filters, multiscale_block_different_pooling, multiscale_block_dilated_convolution, \
    multiscale_block_depthwise_convolution, multiscale_block_inception_convolution, multiscale_block_deeper_network, \
    multiscale_block_residual_connections, multiscale_block_residual_connections_bottleneck_architecture, \
    multiscale_block_pyramid_pooling,  parallel_pathway\

(X_train, y_train) = load_data('AbnormalHeartbeat_TRAIN.arff')
(X_test, y_test) = load_data('AbnormalHeartbeat_TEST.arff')


# it takes a time series as an input, performs 1-D convolution, and returns it as an output ready for concatenation
def get_base_model(input_len, fsize, multiscale_func):

    # the input is a time series of length n and width 1
    input_seq = Input(shape=(input_len, 1))

    if multiscale_func == base_model:
        processed = base_model(input_seq, fsize)
    else:
        processed = multiscale_func(input_seq)

    # global-max-pooling
    processed = GlobalMaxPooling1D()(processed)

    # dense layer with dropout regularization
    compressed = Dense(50, activation="tanh")(processed)
    compressed = Dropout(0.3)(compressed)
    model = Model(inputs=input_seq, outputs=compressed)
    return model


# this is the main model
# it takes the original time series and its down-sampled versions as an input, and returns the result of classification as an output
def main_model(inputs_lens: List[int], fsizes: List[int], multiscale_func):
    # the inputs to the branches are the original time series, and its down-sampled versions
    inputs = []
    branches = []
    for i, input_len in enumerate(inputs_lens):
        inputs.append(Input(shape=(input_len, 1)))
        base_net = get_base_model(input_len, fsizes[i], multiscale_func)
        base_out = base_net(inputs[-1])
        branches.append(base_out)

    # concatenate all the outputs
    merged = Concatenate()(branches)
    layer = Dense(50, activation='sigmoid')(merged)
    out = Dense(5, activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=out)
    return model


# the divisor of the size, 1 is mentioned for original size

# multiscale_inputs_funcs = [downsample_guided, downsample_nl_means, downsample_wavelet, downsample_subsampling, downsample_bilateral,
#                           downsample_gaussian, downsample_mean]

multiscale_inputs_funcs = [downsample_guided, downsample_nl_means, downsample_wavelet, downsample_subsampling, downsample_bilateral,
                           downsample_gaussian, downsample_mean]

multiscale_features_funcs = [base_model, multiscale_block_residual_connections_bottleneck_architecture, multiscale_block_residual_connections, multiscale_block_inception_convolution, multiscale_block_depthwise_convolution, multiscale_block_dilated_convolution, multiscale_block_different_filters]

downsample_factors = [4, 2, 1] #, 6, 8]
fsizes = [8, 16, 24]

# Call each function in a loop
results = []
repetitions = 3
for input_func in multiscale_inputs_funcs:
    for multiscale_func in multiscale_features_funcs:
        max_loss = 0
        max_accuracy = 0
        std_loss = 0
        std_accuracy = 0
        loss_list = []
        accuracy_list = []

        print(f"Starting method:{input_func.__name__}+{multiscale_func.__name__}")

        for idx in range(repetitions):
            input_train_x = input_func(X_train, downsample_factors)
            input_test_x = input_func(X_test, downsample_factors)

            inputs_lens = get_multiscale_lengths(input_train_x)

            m = main_model(inputs_lens, fsizes, multiscale_func)
            m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Define the early stopping callback
            early_stop = EarlyStopping(monitor='val_loss', patience=10)

            history = m.fit(input_train_x, y_train, validation_split=0.2, epochs=10000, callbacks=[early_stop])

            # plot metrics
            #plt.plot(history.history['accuracy'])
            #plt.show()

            score = m.evaluate(input_test_x, y_test, verbose=0)

            loss_list.append(score[0])
            accuracy_list.append(score[1])

        # descriptive statistics
        max_loss = max(loss_list)
        max_accuracy = max(accuracy_list)
        std_loss = statistics.stdev(loss_list)
        std_accuracy = statistics.stdev(accuracy_list)

        msg = input_func.__name__ + ' + ' + multiscale_func.__name__ + '  loss:' + str(max_loss) + ' accuracy: ' + str(max_accuracy)
        print(msg)
        results.append(msg)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

# print experiment results
for res in results:
    print(res)










