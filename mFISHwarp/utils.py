import dask.array as da
import numpy as np
import functools
import operator


def chunks_from_dask(array):
    """
    get zarr style chunks from dask array
    """
    if not isinstance(array, da.core.Array):
        raise ValueError('input is limited to dask array')
    chunks = tuple(int(i[0]) for i in array.chunks)

    return chunks


def chunk_slicer(chunk_position, chunk_size):
    """
    """
    chunk_start = [chunk_size[i] * chunk_position[i] for i in range(len(chunk_position))]
    chunk_end = [chunk_size[i] * (chunk_position[i]+1) for i in range(len(chunk_position))]

    return tuple(slice(x, y) for x, y, in zip(chunk_start, chunk_end))


def flag_array_generator(target_chunk_size, target_shape, mask):
    """
    :param target_chunk_size: tuple
    :param target_shape: tuple as same size as target_chunk_size
    :param mask: 3d array
    :return: 3d array
    """
    mask_shape = mask.shape
    # because the downsampling factor is not always integer, making flag array needs a bit of work.
    down_factor = np.asarray(target_shape) / np.asarray(mask_shape)
    eval_range = []
    for i in range(3):
        intervals = np.arange(0 ,mask_shape[i], (target_chunk_size/down_factor)[i])
        if intervals[-1] < mask_shape[i]:
            intervals = np.array(tuple(intervals)+(mask_shape[i],))
        eval_range.append(intervals)
    eval_range_floor = [np.floor(intervals).astype(int) for intervals in eval_range]
    eval_range_ceil = [np.ceil(intervals).astype(int) for intervals in eval_range]

    flag_array=np.zeros(tuple(i.shape[0]-1 for i in eval_range_ceil), dtype=bool)
    for i in range(eval_range_ceil[0].size-1):
        eval_z_range = slice(eval_range_floor[0][i],eval_range_ceil[0][i+1])
        for j in range(eval_range_ceil[1].size-1):
            eval_y_range = slice(eval_range_floor[1][j],eval_range_ceil[1][j+1])
            for k in range(eval_range_ceil[2].size-1):
                eval_x_range = slice(eval_range_floor[2][k],eval_range_ceil[2][k+1])
                flag_array[i,j,k] = (mask[eval_z_range,eval_y_range,eval_x_range].sum()>0)

    # check if the flag array and number of chunks are consistent.
    return flag_array


def get_trimming_range(overlap_size, target_size, trim_ratio):
    """
    overlap_size: tuple
    target_size: tuple as same length as size1. Each element should be smaller than each element of size1
    trim_ratio: float. 0.0<=trim_ratio<=1.0. 0.0 to be trim none, 1.0 to be trim all overlaps.
    """
    overlaps = tuple(((i - j) // 2, (i - j) - ((i - j) // 2)) for i, j in zip(overlap_size, target_size))
    trimming_ranges = []
    trimmed_sizes = []
    for count, overlap in enumerate(overlaps):
        trimming_range = (round(overlap[0] * trim_ratio), round(overlap[1] * trim_ratio))
        trimmed_size = trimming_range[0] + trimming_range[1]
        trimming_ranges.append(trimming_range)
        trimmed_sizes.append(overlap_size[count] - trimmed_size)
    trimming_ranges = tuple(trimming_ranges)
    trimmed_sizes = tuple(trimmed_sizes)

    return trimming_ranges, trimmed_sizes


def obtain_chunk_slicer(chunks, chunk_position):
    """
    Assume chunks to be the da.chunks format. Not zarr.chunks
    """
    slicer_list = []
    for i, k in enumerate(chunk_position):
        accum_sum = int(np.asarray(chunks[i][:k]).sum())
        slicer_list.append(slice(accum_sum,accum_sum+chunks[i][k],None))
    return tuple(slicer_list)


def get_dask_index(image):
    index = list(np.ndindex(*image.numblocks))

    return index


def slicing_with_chunkidx(da_array, index):
    chunk_info = da_array.chunks
    p = slice(sum(chunk_info[0][:index[0]]),sum(chunk_info[0][:index[0]])+chunk_info[0][index[0]]) 
    q = slice(sum(chunk_info[1][:index[1]]),sum(chunk_info[1][:index[1]])+chunk_info[1][index[1]])
    r = slice(sum(chunk_info[2][:index[2]]),sum(chunk_info[2][:index[2]])+chunk_info[2][index[2]])
    
    return da_array[p, q, r]


def normalization_two_values(arr, lower, upper):
    """
    Normalize array so that the lower values to be 0 and upper values to be 1.
    """
    return (arr - lower) / (upper - lower)


def get_block_iter(image):
    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )
    return block_iter