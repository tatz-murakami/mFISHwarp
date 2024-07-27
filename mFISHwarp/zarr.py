from ome_zarr.scale import Scaler
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.dask_utils import downscale_nearest
import dask.array as da
import numpy as np
# to enable three dimensional scaler, I have to define my own class by inheriting the Scaler class
# https://github.com/ome/ome-zarr-py/blob/8fe43f2530282be557a318c8b2cc27d905ed62c3/ome_zarr/scale.py
from typing import Any, Tuple, Union
import mFISHwarp.utils

ArrayLike = Union[da.Array, np.ndarray]


class IsoScaler(Scaler):
    def resize_image(self, image: ArrayLike) -> ArrayLike:
        """
        Resize a numpy array OR a dask array to a smaller array (not pyramid)
        """
        if isinstance(image, da.Array):

            def _resize(image: ArrayLike, out_shape: Tuple, **kwargs: Any) -> ArrayLike:
                return dask_resize(image, out_shape, **kwargs)

        else:
            _resize = resize

        ### down-sample in all X, Y and Z
        new_shape = list(image.shape)
        new_shape[-1] = image.shape[-1] // self.downscale
        new_shape[-2] = image.shape[-2] // self.downscale
        new_shape[-3] = image.shape[-3] // self.downscale  ### this is a new line
        out_shape = tuple(new_shape)

        dtype = image.dtype
        image = _resize(
            image.astype(float), out_shape, order=1, mode="reflect", anti_aliasing=False
        )
        return image.astype(dtype)


def datasets_metadata_generator(physical_scale, downscale_factor=(1, 2, 2, 2), pyramid_level=1):
    """
    physical_scale (list or tuple): same size as the dimension of the image. if CZYX, it should have size of 4.
    downscale_factor (list or tuple): same size as the data_set_generator. it will be converted to int if it is not.
    """
    if len(physical_scale) != len(downscale_factor):
        raise ValueError("The size of physical scale should match with the size of downscale factor")

    downscale_factor = tuple(int(i) for i in downscale_factor)

    datasets = []
    for resolution in range(pyramid_level):
        scale = [i * (j ** resolution) for i, j in zip(physical_scale, downscale_factor)]
        datasets.append(
            {"path": str(int(resolution)),
             "coordinateTransformations": [
                 {
                     "scale": scale,
                     "type": "scale"
                 }
             ]}
        )

    return datasets


def pyramid_generator_from_dask(arr, downscale_factor=(1, 2, 2, 2), pyramid_level=5, chunk=(1, 256, 256, 256)):
    """
    Nearest neighbor downsampling to make pyramid format.
    arr (da.ndarray)
    downscale_factor (tuple or list): the size of the downscale_factor should be same as the dimension of arr.
    it will be converted to int if it is not.
    pyramid_level (int)
    """
    if arr.ndim != len(downscale_factor):
        raise ValueError("The size of downscale_factor should match with the dimension of arr")
    downscale_factor = tuple(int(i) for i in downscale_factor)

    if arr.ndim != len(chunk):
        raise ValueError("The size of chunk should match with the dimension of arr")

    pyramid_arr = []
    for resolution in range(pyramid_level):
        factors = tuple(i ** resolution for i in downscale_factor)
        pyramid_arr.append(da.rechunk(downscale_nearest(arr, factors), chunk))

    return pyramid_arr


def pyramid_from_dask_to_zarr(arr, zarr_root, downscale_factor=(1, 2, 2, 2), resolution_start=1, pyramid_level=5, chunk=(1, 256, 256, 256)):
    """
    Nearest neighbor downsampling to make pyramid format.
    arr (da.ndarray): full resolution image
    downscale_factor (tuple or list): the size of the downscale_factor should be same as the dimension of arr.
    it will be converted to int if it is not.
    resolution_start (int): which resolution to start saving in zarr. (2**resolution_start) * downscale_factor
    pyramid_level (int): which resolution to stop saving in zarr. 
    """
    if arr.ndim != len(downscale_factor):
        raise ValueError("The size of downscale_factor should match with the dimension of arr")
    downscale_factor = tuple(int(i) for i in downscale_factor)

    if arr.ndim != len(chunk):
        raise ValueError("The size of chunk should match with the dimension of arr")
        
    initial_downscale_factor = tuple(i ** resolution_start for i in downscale_factor)
    down_arr = da.rechunk(downscale_nearest(arr, initial_downscale_factor), chunk)
    p = zarr_root.create_dataset(str(resolution_start),shape=down_arr.shape,chunks=chunk,dtype=down_arr.dtype)
    down_arr.to_zarr(p,dimension_separator='/')
    arr = da.from_zarr(zarr_root[str(resolution_start)])
    
    for resolution in range(resolution_start+1, pyramid_level):
        down_arr = da.rechunk(downscale_nearest(arr, downscale_factor), chunk)
        p = zarr_root.create_dataset(str(resolution),shape=down_arr.shape,chunks=chunk,dtype=down_arr.dtype)
        down_arr.to_zarr(p,dimension_separator='/')
        arr = da.from_zarr(zarr_root[str(resolution)])

    return None

