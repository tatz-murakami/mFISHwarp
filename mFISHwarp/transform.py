import numpy as np
import SimpleITK as sitk
from skimage.transform import rescale
from scipy.ndimage import map_coordinates
import dask.array as da
import cupy as cp
from cupyx.scipy import ndimage
import cucim.skimage.transform


def position_grid(shape,position=None):
    """
    """
    coords = np.meshgrid(*[range(x) for x in shape], indexing='ij')
    coords = np.array(coords).astype(np.int16)
    if position is not None:
        for i, offset in enumerate(position):
            coords[i, ...] = coords[i, ...] + int(offset)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def displacement_itk2numpy(itk_displacement):
    """
    """
    relative_displacement = sitk.GetArrayFromImage(itk_displacement)
    relative_displacement = relative_displacement[...,
                            ::-1]  # very important. this comes from the fact ITK use xyz while numpy is zyx.

    return relative_displacement


def relative2positional(relative_displacement,position=None):
    """
    """
    # make meshgrid to generate displacement field
    grid_array = position_grid(relative_displacement.shape[:-1], position)
    positional_displacement = relative_displacement + grid_array

    return positional_displacement


def transform_block(displacement, mov, order=1):
    """
    displacement: ndarray
    mov: ndarray
    """
    if not isinstance(displacement, np.ndarray):
        displacement = np.asarray(displacement)

    # get moving data block coordinates
    s = np.floor(displacement.min(axis=(0, 1, 2))).astype(int)
    s = np.maximum(0, s)
    e = np.ceil(displacement.max(axis=(0, 1, 2))).astype(int) + 1
    e = np.maximum(0, e)
    coord = np.moveaxis(displacement - s, -1, 0)
    slc = tuple(slice(x, y) for x, y in zip(s, e))
    # slice data
    mov_block = mov[slc]
    # interpolate block (adjust transform to local origin)
    mov_block = map_coordinates(mov_block, coord, order=order, mode='constant')

    return mov_block


def transform_block_gpu(displacement, mov, order=1, size_limit=1024*1024*1024):
    """
    displacement: ndarray
    mov: ndarray
    """
    if not isinstance(displacement, np.ndarray):
        displacement = np.asarray(displacement)

    displacement = cp.array(displacement)
    # get moving data block coordinates
    s = cp.floor(displacement.min(axis=(0, 1, 2))).astype(int)  # min/max is very slow for GPU...
    s = cp.maximum(0, s)
    e = cp.ceil(displacement.max(axis=(0, 1, 2))).astype(int) + 1
    e = cp.maximum(0, e)
    coord = cp.moveaxis(displacement - s, -1, 0)
    s = s.get()
    e = e.get()
    slc = tuple(slice(x, y) for x, y in zip(s, e))
    size = np.prod(e - s)

    if size < size_limit:
        # slice data and move to cupy
        mov_block = cp.array(mov[slc])
        # interpolate block (adjust transform to local origin)
        mov_block = (ndimage.map_coordinates(mov_block, coord, order=order, mode='constant')).get()

        # del mov_block_cp
        del displacement
        del coord
        cp._default_memory_pool.free_all_blocks()

        return mov_block
    else:
        mov_block = np.zeros(displacement.shape[:-1], dtype=mov.dtype)
        del displacement
        del coord
        cp._default_memory_pool.free_all_blocks()

        return mov_block


def upscale_displacement_overlap(positional_displacement, rescale_constant, out_chunk_size=(512, 512, 512), out_overlap=(64,64,64)):
    """
    positional_displacement: ndarray
    rescale_constant: tuple
    out_chunk_size: tuple
    out_overlap = tuple
    """

    if (np.remainder(out_chunk_size,rescale_constant) != np.array([0,0,0])).any():
        raise ValueError('out_chunk_size is expected to be multiple of rescale_constant')
    if (np.remainder(out_overlap,rescale_constant)!= np.array([0,0,0])).any():
        raise ValueError('out_overlap is expected to be multiple of rescale_constant')

    in_chunk_size = tuple(x // y for x,y in zip(out_chunk_size, rescale_constant))
    in_overlap = tuple(x // y for x,y in zip(out_overlap, rescale_constant))

    def wrap_rescale(array, rescale_constant=rescale_constant):
        rescaled_array = rescale(array,
                                 rescale_constant + (1,),
                                 order=1,
                                 mode='edge')
        return rescaled_array

    def rescale_chunk_overlap(displacement_chunks, rescale_constant=rescale_constant, overlap=in_overlap):
        rescaled_chunks = []
        for i in range(3):
            rescaled_chunks.append(
                tuple((np.array(displacement_chunks[:-1][i]) + overlap[i] * 2)
                      * rescale_constant[i])
            )
        rescaled_chunks.append(displacement_chunks[-1])
        rescaled_chunks = tuple(rescaled_chunks)

        return rescaled_chunks

    # prepare up-scaling.
    for i, scale in enumerate(rescale_constant):
        positional_displacement[..., i] = positional_displacement[..., i] * scale
    # positional_displacement = positional_displacement * rescale_constant[0]

    # convert to dask for parallelization.
    if isinstance(positional_displacement, np.ndarray):
        displacement_da = da.from_array(positional_displacement, chunks=in_chunk_size + (1,))
    elif isinstance(positional_displacement, da.Array):
        displacement_da = da.rechunk(positional_displacement, chunks=in_chunk_size + (1,))

    displacement_rescale_overlap = da.map_overlap(
        wrap_rescale,
        displacement_da,
        depth=in_overlap,
        boundary='nearest',  # 'nearest' to avoid loading huge mov file
        trim=False,
        dtype=np.float32,
        chunks=rescale_chunk_overlap(displacement_da.chunks)
    )

    return displacement_rescale_overlap


# def upscale_displacement(positional_displacement, rescale_constant, out_chunk_size=(512, 512, 512), margin=(4, 4, 4)):
#     """
#     positional_displacement: ndarray
#     rescale_constant: tuple
#     out_chunk_size: tuple
#     margin: tuple. To avoid artifacts at the edge of the blocks
#     """
#
#     displacement_rescale_overlap = upscale_displacement_overlap(positional_displacement, rescale_constant, out_chunk_size, margin)
#     displacement_rescale = da.overlap.trim_overlap(displacement_rescale_overlap, margin + (0,), boundary="reflect")
#
#     return displacement_rescale

def upscale_displacement(displacement, rescale_constant, out_chunk_size=(512, 512, 512)):
    if displacement.ndim != (len(rescale_constant)+1):
        raise ValueError("dimension of displacement and rescale constant does not match")
    if (np.remainder(out_chunk_size,rescale_constant) != np.array([0,0,0])).any():
        raise ValueError('out_chunk_size is expected to be multiple of rescale_constant')

    def wrap_rescale(array, rescale_constant=rescale_constant):
        rescaled_array = rescale(array,
                                 rescale_constant + (1,),
                                 order=1,
                                 mode='edge')
        return rescaled_array

    # prepare up-scaling.
    for i, scale in enumerate(rescale_constant):
        displacement[..., i] = displacement[..., i] * scale

    in_chunk_size = tuple(x // y for x, y in zip(out_chunk_size, rescale_constant))

    if isinstance(displacement, np.ndarray):
        displacement_da = da.from_array(displacement, chunks=in_chunk_size + (1,))
    elif isinstance(displacement, da.Array):
        displacement_da = da.rechunk(displacement, chunks=in_chunk_size + (1,))


    rescaled_displacement = da.map_blocks(
        wrap_rescale,
        displacement_da,
        dtype=displacement_da.dtype,
        chunks=out_chunk_size+(1,)
    )

    return rescaled_displacement


def upscale_displacement_overlap_gpu(positional_displacement, rescale_constant, out_chunk_size=(512, 512, 512), out_overlap=(64,64,64)):
    """
    positional_displacement: ndarray
    rescale_constant: tuple
    out_chunk_size: tuple
    out_overlap = tuple
    """

    if (np.remainder(out_chunk_size,rescale_constant) != np.array([0,0,0])).any():
        raise ValueError('out_chunk_size is expected to be multiple of rescale_constant')
    if (np.remainder(out_overlap,rescale_constant)!= np.array([0,0,0])).any():
        raise ValueError('out_overlap is expected to be multiple of rescale_constant')

    in_chunk_size = tuple(x // y for x,y in zip(out_chunk_size, rescale_constant))
    in_overlap = tuple(x // y for x,y in zip(out_overlap, rescale_constant))

    def wrap_rescale(array, rescale_constant=rescale_constant):
        array = cp.array(array)
        rescaled_array = cucim.skimage.transform.rescale(array,
                                 rescale_constant + (1,),
                                 order=1,
                                 mode='edge')
        rescaled_array = rescaled_array.get()
        del array
        cp._default_memory_pool.free_all_blocks()

        return rescaled_array

    def rescale_chunk_overlap(displacement_chunks, rescale_constant=rescale_constant, overlap=in_overlap):
        rescaled_chunks = []
        for i in range(3):
            rescaled_chunks.append(
                tuple((np.array(displacement_chunks[:-1][i]) + overlap[i] * 2)
                      * rescale_constant[i])
            )
        rescaled_chunks.append(displacement_chunks[-1])
        rescaled_chunks = tuple(rescaled_chunks)

        return rescaled_chunks

    # prepare up-scaling.
    # for i, scale in enumerate(rescale_constant):
    #     positional_displacement[..., i] = positional_displacement[..., i] * scale
    positional_displacement = positional_displacement * rescale_constant[0]

    # convert to dask for parallelization.
    if isinstance(positional_displacement, da.Array):
        displacement_da = da.rechunk(positional_displacement, chunks=in_chunk_size + (1,))
    else:
        displacement_da = da.from_array(positional_displacement, chunks=in_chunk_size + (1,))

    displacement_rescale_overlap = da.map_overlap(
        wrap_rescale,  # rewrite to rescale of cucim
        displacement_da,
        depth=in_overlap,
        boundary='nearest',  # 'nearest' to avoid loading huge mov file
        trim=False,
        dtype=np.float32,
        chunks=rescale_chunk_overlap(displacement_da.chunks)
    )

    return displacement_rescale_overlap


def upscale_displacement_gpu(positional_displacement, rescale_constant, out_chunk_size=(512, 512, 512), margin=(4, 4, 4)):
    """
    positional_displacement: ndarray
    rescale_constant: tuple
    out_chunk_size: tuple
    margin: tuple. To avoid artifacts at the edge of the blocks
    """

    displacement_rescale_overlap = upscale_displacement_overlap_gpu(positional_displacement, rescale_constant, out_chunk_size, margin)
    displacement_rescale = da.overlap.trim_overlap(displacement_rescale_overlap, margin + (0,), boundary="reflect")

    return displacement_rescale



def relative2positional_gpu(relative_displacement):
    """
    convert relative displacement to positional displacement
    """
    # make grid in cupy
    shape = relative_displacement.shape[:-1]
    grid_array = cp.asarray(cp.meshgrid(*[cp.asarray(range(x)) for x in shape], indexing='ij')).astype(int)
    grid_array = cp.moveaxis(grid_array, 0, -1)
    relative_displacement = cp.asarray(relative_displacement)
    positional_displacement = relative_displacement + grid_array
    # return as numpy
    positional_displacement = positional_displacement.get()

    del relative_displacement
    del grid_array
    cp._default_memory_pool.free_all_blocks()

    return positional_displacement


def composite_displacement_gpu(disp1, disp2, order=1):
    """
    disp1,2: ndarray in same shape. (z,y,x,dispvector)
    """
    disp1 = cp.array(disp1)
    disp2 = cp.array(disp2)
    disp2 = cp.moveaxis(disp2, -1, 0)
    merged_disp = cp.empty_like(disp1)
    for i in range(disp1.shape[-1]):
        merged_disp[..., i] = ndimage.map_coordinates(disp1[..., i], disp2, order=order, mode='constant')

    # return as numpy
    merged_disp = merged_disp.get()
    del (disp1, disp2)
    cp._default_memory_pool.free_all_blocks()

    return merged_disp


def composite_displacement(disp1, disp2, order=1):
    """
    disp1,2: ndarray in same shape. (z,y,x,dispvector)
    """
    disp2 = np.moveaxis(disp2, -1, 0)
    merged_disp = np.empty_like(disp1)
    for i in range(disp1.shape[-1]):
        merged_disp[..., i] = map_coordinates(disp1[..., i], disp2, order=order, mode='constant')

    return merged_disp


def trim_array_to_size(arr, size):
    """
    This function only trim and does not pad.
    """
    if len(arr.shape) != len(size):
        raise ValueError('the dimension of target shape and array should be the same.')
    
    slice_list = []
    for i in range(len(arr.shape)):
        if arr.shape[i] > size[i]:
            slice_list.append(slice(0,size[i]))
        else:
            slice_list.append(slice(0,None))
    resized_array = arr[tuple(slice_list)]
    return resized_array

def pad_array_to_size(arr, size, *args, **kwargs):
    """
    This function only pad and does not trim. This function works with dask
    """
    if len(arr.shape) != len(size):
        raise ValueError('the dimension of target shape and array should be the same.')
        
    for i in range(len(arr.shape)):
        if arr.shape[i] < size[i]:
            pad_width = size[i] - arr.shape[i]
            pad_list = [[0,0] for i in range(len(arr.shape))]
            pad_list[i][1] = pad_width
            if isinstance(arr, np.ndarray):
                arr = np.pad(arr, pad_list, *args, **kwargs)
            elif isinstance(arr, da.Array):
                arr = da.pad(arr, pad_list, *args, **kwargs)
    return arr

def pad_trim_array_to_size(arr, target_shape, *args, **kwargs):
    """
    This function pads and trims the array to target shape.
    args, kwargs: parameters for padding. (mode='constant', constant_values=0) for zero padding, 
    """
    if len(arr.shape) != len(target_shape):
        raise ValueError('the dimension of target shape and array should be the same.')
        
    arr = trim_array_to_size(arr,target_shape)
    arr = pad_array_to_size(arr,target_shape, *args, **kwargs)
    
    return arr