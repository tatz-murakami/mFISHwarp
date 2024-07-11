import numpy as np
from scipy import fft


def create_3d_hann_window(window_size, array_size):
    """
    Create a 3D Hann window of specified size, centered in an array of the given size.

    Parameters:
    window_size : tuple of 3 integers
        The size of the Hann window (depth, height, width)
    array_size : tuple of 3 integers
        The size of the resulting array, typically the image size (depth, height, width)

    Returns:
    numpy.ndarray
        3D array of size array_size with a centered Hann window of size window_size
    """
    win_depth, win_height, win_width = window_size
    arr_depth, arr_height, arr_width = array_size

    # Create 1D Hann windows
    hann_d = np.hanning(win_depth)
    hann_h = np.hanning(win_height)
    hann_w = np.hanning(win_width)

    # Create 3D Hann window
    hann_3d = np.outer(hann_d, hann_h).reshape(win_depth, win_height, 1) * hann_w

    # Create an array of zeros with the desired output size
    output = np.zeros(array_size)

    # Calculate padding
    pad_d = (arr_depth - win_depth) // 2
    pad_h = (arr_height - win_height) // 2
    pad_w = (arr_width - win_width) // 2

    # Place the Hann window in the center of the output array
    output[pad_d:pad_d + win_depth,
    pad_h:pad_h + win_height,
    pad_w:pad_w + win_width] = hann_3d

    return output


def fft_filter(arr, mask, return_fft=False):
    """
    Apply FFT filter to the array given mask arr and mask should be the same size.
    Parameters:
    arr : numpy.ndarray
    mask : numpy.ndarray
        The values usually between 0 and 1 in real number. The size should be same as arr.

    Returns:
    numpy.ndarray
        Array with the same size as arr
    """
    if arr.shape != mask.shape:
        raise ValueError('The size of the mask should be same as arr')
    fft_ = fft.fftn(arr)
    fft_shifted = fft.fftshift(fft_)

    fft_filtered = fft_shifted * mask
    fft_filtered_shifted = fft.ifftshift(fft_filtered)

    # get only real part
    filtered_arr = np.real(fft.ifftn(fft_filtered_shifted))

    if return_fft:
        return filtered_arr, fft_filtered
    else:
        return filtered_arr
