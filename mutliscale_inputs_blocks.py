import pywt
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.models import Model

def wavelet_transform(x, wavelet='db1', level=1):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    return [c for c in coeffs]


def wavelatet_input(input_scale1, input_scale2, input_scale3):

    # Apply the wavelet transform to each input tensor
    wt1 = Lambda(lambda x: wavelet_transform(x, wavelet='db1', level=1))(input_scale1)
    wt2 = Lambda(lambda x: wavelet_transform(x, wavelet='db1', level=2))(input_scale2)
    wt3 = Lambda(lambda x: wavelet_transform(x, wavelet='db1', level=3))(input_scale3)

    # Define the convolutional layers for each scale
    conv1 = Conv1D(32, kernel_size=3, activation='relu')(wt1[0])
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(64, kernel_size=3, activation='relu')(wt2[0])
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(128, kernel_size=3, activation='relu')(wt3[0])
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    # Concatenate the output feature maps from each scale
    merged = concatenate([pool1, pool2, pool3])
    
    return merged