import pywt
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import pywt

from scipy.ndimage import gaussian_filter1d
from cv2.ximgproc import createGuidedFilter
from scipy.optimize import minimize
from skimage.restoration import denoise_nl_means

def downsample_bilateral(x: pd.DataFrame, downsample_factors, kernel_size=5, sigma_color=75, sigma_space=75):
    downsampled_x = []

    for factor in downsample_factors:
        # Convert the time series DataFrame to a NumPy array
        x_array = x.values.astype(np.float32)

        # Initialize the list to store the downsampled signals
        downsampled_signals = []

        # Process each row (1D signal) separately
        for row in x_array:
            # Reshape the row to a 1D "image"
            row_image = row.reshape(-1, 1)

            # Apply the bilateral filter to the row "image"
            denoised_row_image = cv2.bilateralFilter(row_image, kernel_size, sigma_color, sigma_space)

            # Flatten the denoised row "image" back to a 1D signal
            denoised_signal = denoised_row_image.flatten()

            # Downsample the denoised signal
            downsampled_signal = denoised_signal[::factor]

            # Add the downsampled signal to the list
            downsampled_signals.append(downsampled_signal)

        # Convert the downsampled signals to a DataFrame
        downsampled = pd.DataFrame(downsampled_signals, index=x.index, columns=x.columns[::factor])

        # Add the downsampled time series to the list
        downsampled_x.append(downsampled)

    return downsampled_x

def downsample_gaussian(x: pd.DataFrame, downsample_factors, sigma=1):
    downsampled_x = []

    for factor in downsample_factors:
        # Convert the time series DataFrame to a NumPy array
        x_array = x.values.astype(np.float64)

        # Initialize the list to store the downsampled signals
        downsampled_signals = []

        # Process each row (1D signal) separately
        for row in x_array:
            # Apply the Gaussian filter to the row
            denoised_signal = gaussian_filter1d(row, sigma=sigma)

            # Downsample the denoised signal
            downsampled_signal = denoised_signal[::factor]

            # Add the downsampled signal to the list
            downsampled_signals.append(downsampled_signal)

        # Convert the downsampled signals to a DataFrame
        downsampled = pd.DataFrame(downsampled_signals, index=x.index, columns=x.columns[::factor])

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

def downsample_wavelet(x: pd.DataFrame, downsample_factors, wavelet='db4', mode='soft'):
    downsampled_x = []

    for factor in downsample_factors:
        # Convert the time series DataFrame to a NumPy array
        x_array = x.values

        # Initialize the list to store the downsampled signals
        downsampled_signals = []

        # Process each row (1D signal) separately
        for row in x_array:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(row, wavelet)

            # Compute the threshold for denoising
            threshold = factor * np.median(np.abs(coeffs[-1]))

            # Apply thresholding to the wavelet coefficients
            denoised_coeffs = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]

            # Reconstruct the denoised signal
            denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

            # Downsample the denoised signal
            downsampled_signal = denoised_signal[::factor]

            # Add the downsampled signal to the list
            downsampled_signals.append(downsampled_signal)

        # Convert the downsampled signals to a DataFrame
        downsampled = pd.DataFrame(downsampled_signals, index=x.index, columns=x.columns[::factor])

        # Add the downsampled time series to the list
        downsampled_x.append(downsampled)

    return downsampled_x


# guided filter
def downsample_guided(x: pd.DataFrame, downsample_factors, radius=3, eps=0.001):
    downsampled_x = []

    for factor in downsample_factors:
        # Convert the time series DataFrame to a NumPy array
        x_array = x.values.astype(np.float32)

        # Initialize the list to store the downsampled signals
        downsampled_signals = []

        # Process each row (1D signal) separately
        for row in x_array:
            # Reshape the row to a 1D "image"
            row_image = row.reshape(-1, 1)

            # Create a guided filter instance
            gf = createGuidedFilter(row_image, radius, eps)

            # Apply the guided filter to the row "image"
            denoised_row_image = gf.filter(row_image)

            # Flatten the denoised row "image" back to a 1D signal
            denoised_signal = denoised_row_image.flatten()

            # Downsample the denoised signal
            downsampled_signal = denoised_signal[::factor]

            # Add the downsampled signal to the list
            downsampled_signals.append(downsampled_signal)

        # Convert the downsampled signals to a DataFrame
        downsampled = pd.DataFrame(downsampled_signals, index=x.index, columns=x.columns[::factor])

        # Add the downsampled time series to the list
        downsampled_x.append(downsampled)

    return downsampled_x



# non-local means filter
def downsample_nl_means(x: pd.DataFrame, downsample_factors, search_radius=3, h=0.1):
    downsampled_x = []

    for factor in downsample_factors:
        # Convert the time series DataFrame to a NumPy array
        x_array = x.values.astype(np.float64)

        # Initialize the list to store the downsampled signals
        downsampled_signals = []

        # Process each row (1D signal) separately
        for row in x_array:
            # Reshape the row to a 1D "image"
            row_image = row.reshape(-1, 1)

            # Apply the non-local means filter to the row "image"
            denoised_row_image = denoise_nl_means(row_image,  patch_distance=search_radius, h=h, fast_mode=True)

            # Flatten the denoised row "image" back to a 1D signal
            denoised_signal = denoised_row_image.flatten()

            # Downsample the denoised signal
            downsampled_signal = denoised_signal[::factor]

            # Add the downsampled signal to the list
            downsampled_signals.append(downsampled_signal)

        # Convert the downsampled signals to a DataFrame
        downsampled = pd.DataFrame(downsampled_signals, index=x.index, columns=x.columns[::factor])

        # Add the downsampled time series to the list
        downsampled_x.append(downsampled)

    return downsampled_x


def tv_denoise(x, lmbd):
    n = len(x)
    objective_function = lambda z: 0.5 * np.sum((z - x)**2) + lmbd * np.sum(np.abs(np.diff(z)))
    z0 = x.copy()
    bounds = [(None, None)] * n
    result = minimize(objective_function, z0, bounds=bounds, method='L-BFGS-B')
    return result.x



