from scipy import ndimage as ndi
import numpy as np
from skimage.filters import gaussian
from skimage.morphology import binary_dilation
from skimage.morphology import ball


def extract_largest_object_from_binary(binary_img):
    """
    input
        binary_img: ndarray. binarized image.
    return
        object_img: ndarray. binarized image of the largest object
    """
    # Find object and select the largest object.
    label_objects, nb_labels = ndi.label(binary_img)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0  # To remove index=0. because it is background.
    object_img = (label_objects == np.argmax(sizes)).astype(float)  # Get the largest objects. Making it a mask.

    return object_img


def mask_maker(img, background_value, sigma=3, dilation_radius=3, **kwargs):
    # segmentation of the object
    img_gauss = gaussian(img, sigma=sigma, mode='nearest', preserve_range=True, **kwargs)
    img_mask = (img_gauss > background_value)
    img_mask = binary_dilation(img_mask, ball(dilation_radius))

    # remove disconnected noises if exist
    img_mask = extract_largest_object_from_binary(img_mask)

    return img_mask
