import pywt
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

from scipy.ndimage import gaussian_filter1d

def downsample_bilateral(x, downsample_factors, kernel_size=5, sigma_color=75, sigma_space=75):
    downsampled_x = []
    for factor in downsample_factors:
        # Compute the length of the downsampled time series
        length = int(x.shape[1] / factor)

        # Convert the time series DataFrame to a NumPy array of 32-bit float values
        x_array = x.values.astype(np.float32)

        # Downsample the time series using bilateral filtering
        downsampled = cv2.bilateralFilter(x_array, kernel_size, sigma_color, sigma_space)[:, ::factor]

        # Convert the downsampled array to a DataFrame
        downsampled = pd.DataFrame(downsampled, index=x.index, columns=x.columns[::factor])

        # Add the downsampled time series to the list
        downsampled_x.append(downsampled)

    return downsampled_x

def downsample_gaussian(x: pd.DataFrame, downsample_factors):
    downsampled_x = []
    for factor in downsample_factors:
        # Compute the length of the downsampled time series
        length = int(x.shape[1] / factor)

        # Downsample the time series using a Gaussian filter
        downsampled = gaussian_filter1d(x, sigma=factor, axis=1, truncate=1/factor)[:, ::factor]

        # Convert the downsampled array to a DataFrame
        downsampled = pd.DataFrame(downsampled, index=x.index, columns=x.columns[::factor])

        # Add the downsampled time series to the list
        downsampled_x.append(downsampled)

    return downsampled_x


def downsample_mean(x: pd.DataFrame, downsample_factors):
    downsampled_x = []
    for factor in downsample_factors:
        # Compute the length of the downsampled time series
        length = int(x.shape[1] / factor)

        # Downsample the time series using simple averaging
        downsampled = x.iloc[:, :(length * factor)].values.reshape(x.shape[0], length, factor).mean(axis=2)

        # Convert the downsampled array to a DataFrame
        downsampled = pd.DataFrame(downsampled, index=x.index, columns=x.columns[:length])

        # Add the downsampled time series to the list
        downsampled_x.append(downsampled)

    return downsampled_x

def downsample_subsampling(x: pd.DataFrame, donwsample_factors):
    downsampled_x = []
    for factor in donwsample_factors:
        sample = x.iloc[:, ::factor]
        downsampled_x.append(sample)
    return downsampled_x
