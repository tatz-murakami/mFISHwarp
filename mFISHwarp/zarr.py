from ome_zarr.scale import Scaler
from ome_zarr.dask_utils import resize as dask_resize
import dask.array as da
import numpy as np
# to enable three dimensional scaler, I have to define my own class by inheriting the Scaler class
# https://github.com/ome/ome-zarr-py/blob/8fe43f2530282be557a318c8b2cc27d905ed62c3/ome_zarr/scale.py
from typing import Any, Tuple, Union

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
