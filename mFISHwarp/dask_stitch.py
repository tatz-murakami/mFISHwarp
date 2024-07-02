"""
MIT License

Copyright (c) 2021 GFleishman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import dask.array as da
import copy
from itertools import product
import mFISHwarp.transform

def weight_block(block, blocksize, overlap, block_info=None):
    """
    """

    # determine which faces need linear weighting
    core, pad_ones, pad_linear = [], [], []
    block_index = block_info[0]['chunk-location']
    block_grid = block_info[0]['num-chunks']
    for i in range(3):

        # get core shape and pad sizes
        o = overlap[i]  # shorthand
        c = blocksize[i] - 2*o + 2
        p_ones, p_linear = [0, 0], [2*o-1, 2*o-1]
        if block_index[i] == 0:# if chunk is edge at the start
            p_ones[0], p_linear[0] = 2*o-1, 0
        if block_index[i] == block_grid[i] - 1:# if chunk is edge at the end
            p_ones[1], p_linear[1] = 2*o-1, 0
        core.append(c)
        pad_ones.append(tuple(p_ones))
        pad_linear.append(tuple(p_linear))

    # create weights core
    weights = np.ones(core, dtype=np.float32)

    # extend weights
    weights = np.pad(
        weights, pad_ones, mode='constant', constant_values=1,
    )
    weights = np.pad(
        weights, pad_linear, mode='linear_ramp', end_values=0,
    )
    
    # trim the weight if the size is beyond the block[...,0].shape
    

    # block may be a vector field
    # conditional is too general, but works for now
    if weights.ndim != block.ndim:
        weights = mFISHwarp.transform.trim_array_to_size(weights, block[...,0].shape)
        weights = weights[..., None]# this is just to add more dimension.
    elif weights.ndim == block.ndim:
        weights = mFISHwarp.transform.trim_array_to_size(weights, block.shape)
    
    # multiply data by weights and return
    return np.multiply(block, weights)# np.multiply automaticaly auguments to the shape


def merge_overlaps(block, overlap, block_info=None):
    """
    """

    o = overlap  # shorthand
    core = [slice(2*x, -2*x) for x in o]
    result = np.require(block[tuple(core)], requirements='W') # ensure a writable array
    block_index = block_info[0]['chunk-location']
    block_grid = block_info[0]['num-chunks']

    # faces
    for ax in range(3):
        # the left side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(0, o[ax])
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(0, o[ax])
        block_to_add = block[tuple(slc2)]
        block_to_add = mFISHwarp.transform.pad_trim_array_to_size(block_to_add, result[tuple(slc1)].shape, mode='edge')
        result[tuple(slc1)] += block_to_add
        # the right side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(-1*o[ax], None)
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(-1*o[ax], None)
        block_to_add = block[tuple(slc2)]
        block_to_add = mFISHwarp.transform.pad_trim_array_to_size(block_to_add, result[tuple(slc1)].shape, mode='edge')
        result[tuple(slc1)] += block_to_add

    # edges
    for edge in product([0, 1], repeat=2):
        for ax in range(3):
            oo = np.delete(o, ax)
            left = [slice(None, oe) for oe in oo]
            right = [slice(-1*oe, None) for oe in oo]
            slc1 = [l if e == 0 else r for l, r, e in zip(left, right, edge)]
            slc2 = copy.deepcopy(slc1)
            slc1.insert(ax, slice(None, None))
            slc2.insert(ax, core[ax])
            block_to_add = block[tuple(slc2)]
            block_to_add = mFISHwarp.transform.pad_trim_array_to_size(block_to_add, result[tuple(slc1)].shape, mode='edge')
            result[tuple(slc1)] += block_to_add

    # corners
    for corner in product([0, 1], repeat=3):
        left = [slice(None, oe) for oe in o]
        right = [slice(-1*oe, None) for oe in o]
        slc = [l if c == 0 else r for l, r, c in zip(left, right, corner)]
        block_to_add = block[tuple(slc)]
        block_to_add = mFISHwarp.transform.pad_trim_array_to_size(block_to_add, result[tuple(slc)].shape, mode='edge')
        result[tuple(slc)] += block_to_add

    return result


def stitch_blocks(arr_da, blocksize, overlap, fullchunks):
    """
    """

    # blocks may be a vector fields
    # conditional is too general, but works for now
    if arr_da.ndim != len(blocksize):
        blocksize = list(blocksize) + [3,]
        overlap = tuple(overlap) + (0,)

    # weight block edges
    weighted_blocks = da.map_blocks(
        weight_block, arr_da,
        blocksize=blocksize[:3],
        overlap=overlap[:3],
        dtype=np.float32,
    )

    # stitch overlap regions
    return da.map_overlap(
        merge_overlaps, weighted_blocks,
        overlap=overlap[:3],
        depth=overlap,
        boundary=0.,
        trim=False,
        dtype=np.float32,
        chunks=fullchunks,
    )