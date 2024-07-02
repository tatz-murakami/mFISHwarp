"""
Copyright © 2020 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
"""
This script is modified from Cellpose project
"""

"""Segments detected regions in a chunked dask array.
Uses overlapping chunks during segmentation, and determines how to link segments
between neighboring chunks by examining the overlapping border regions.
Heavily based on dask_image.ndmeasure.label, which uses non-overlapping blocks
with a structuring element that links segments at the chunk boundaries.
"""

import functools
# import logging
import operator

import dask
import dask.array as da
import numpy as np

from cellpose import models

class DistSegError(Exception):
    """Error in image segmentation."""

try:
    from dask_image.ndmeasure._utils import _label
    from sklearn import metrics as sk_metrics
except ModuleNotFoundError as e:
    raise DistSegError("Install 'cellpose[distributed]' for distributed segmentation dependencies") from e


# logger = logging.getLogger(__name__)


def segment(
    image,
    channels,
    model_type,# path to model
    diameter,
    min_size=20,
    fast_mode=False,
    use_anisotropy=True,
    iou_depth=2,
    iou_threshold=0.7,
):
    """Use cellpose to segment nuclei in fluorescence data.
    Parameters
    ----------
    image : array of shape (z, y, x, channel)
        Image used for detection of objects
    channels : array of int with size 2
        See cellpose
    model_type : str
        path to model # "cyto" or "nuclei"
    diameter : tuple of size 3
        Approximate diameter (in pixels) of a segmented region, i.e. cell width
    fast_mode : bool
        In fast mode, network averaging, tiling, and augmentation are turned off.
    use_anisotropy : bool
        If true, use anisotropy parameter of cellpose
    iou_depth: dask depth parameter
        Number of pixels of overlap to use in intersection-over-union calculation when
        linking segments across neighboring, overlapping dask chunk regions.
    iou_threshold: float
        Minimum intersection-over-union in neighboring, overlapping dask chunk regions
        to be considered the same segment.  The region for calculating IOU is given by the
        iou_depth parameter.
    Returns:
        segments : array of int32 with same shape as input
            Each segmented cell is assigned a number and all its pixels contain that value (0 is background)
    """
    assert image.ndim == 4, image.ndim
    assert image.shape[-1] in {1, 2}, image.shape
    assert diameter[1] == diameter[2], diameter

    diameter_yx = diameter[1]
    anisotropy = diameter[0] / diameter[1] if use_anisotropy else None

    image = da.asarray(image)
    image = image.rechunk({-1: -1})  # color channel is chunked together

    depth = tuple(np.ceil(diameter).astype(np.int64))
    boundary = "reflect"

    # No chunking in channel direction
    image = da.overlap.overlap(image, depth + (0,), boundary)

    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )

    labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
    total = None
    for index, input_block in block_iter:
        labeled_block, n = dask.delayed(segment_chunk, nout=2)(
            input_block,
            channels,
            model_type,
            diameter_yx,
            anisotropy,
            fast_mode,
            index,
            min_size,
        )

        shape = input_block.shape[:-1]
        labeled_block = da.from_delayed(labeled_block, shape=shape, dtype=np.int32)

        n = dask.delayed(np.int32)(n)
        n = da.from_delayed(n, shape=(), dtype=np.int32)

        total = n if total is None else total + n

        block_label_offset = da.where(labeled_block > 0, total, np.int32(0))
        labeled_block += block_label_offset

        labeled_blocks[index[:-1]] = labeled_block
        total += n

    # Put all the blocks together
    block_labeled = da.block(labeled_blocks.tolist())

    depth = da.overlap.coerce_depth(len(depth), depth)

    if np.prod(block_labeled.numblocks) > 1:
        iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)

        if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
            raise DistSegError("iou_depth (%s) > depth (%s)" % (iou_depth, depth))

        trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
        block_labeled = da.overlap.trim_internal(
            block_labeled, trim_depth, boundary=boundary
        )
        block_labeled = link_labels(
            block_labeled,
            total,
            iou_depth,
            iou_threshold=iou_threshold,
        )

        block_labeled = da.overlap.trim_internal(
            block_labeled, iou_depth, boundary=boundary
        )

    else:
        block_labeled = da.overlap.trim_internal(
            block_labeled, depth, boundary=boundary
        )

    return block_labeled

def make_overlap(
    image,
    channels,
    diameter,
    use_anisotropy=True,
):
    """Use cellpose to segment nuclei in fluorescence data.
    Parameters
    ----------
    image : array of shape (z, y, x, channel)
        Image used for detection of objects
    channels : array of int with size 2
        See cellpose
    model_type : str
        path to model # "cyto" or "nuclei"
    diameter : tuple of size 3
        Approximate diameter (in pixels) of a segmented region, i.e. cell width
    fast_mode : bool
        In fast mode, network averaging, tiling, and augmentation are turned off.
    use_anisotropy : bool
        If true, use anisotropy parameter of cellpose
    Returns:
        segments : array of int32 with same shape as input
            Each segmented cell is assigned a number and all its pixels contain that value (0 is background)
    """
    assert image.ndim == 4, image.ndim
    assert image.shape[-1] in {1, 2}, image.shape
    assert diameter[1] == diameter[2], diameter

    diameter_yx = diameter[1]
    anisotropy = diameter[0] / diameter[1] if use_anisotropy else None

    image = da.asarray(image)
    image = image.rechunk({-1: -1})  # color channel is chunked together

    depth = tuple(np.ceil(diameter).astype(np.int64))
    boundary = "reflect"

    # No chunking in channel direction
    image = da.overlap.overlap(image, depth + (0,), boundary)
    
    return image


# def overlap_segmentation(
#     image,# image with overlap, dask array
#     channels,
#     model_type,# path to model
#     diameter,
#     min_size=20,
#     fast_mode=False,
#     use_anisotropy=True,
# ):
#     diameter_yx = diameter[1]
#     anisotropy = diameter[0] / diameter[1] if use_anisotropy else None
#     depth = tuple(np.ceil(diameter).astype(np.int64))
#     boundary = "reflect"

#     block_iter = zip(
#         np.ndindex(*image.numblocks),
#         map(
#             functools.partial(operator.getitem, image),
#             da.core.slices_from_chunks(image.chunks),
#         ),
#     )

#     labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
#     total = None
#     for index, input_block in block_iter:
#         labeled_block, n = dask.delayed(segment_chunk, nout=2)(
#             input_block,
#             channels,
#             model_type,
#             diameter_yx,
#             anisotropy,
#             fast_mode,
#             index,
#             min_size,
#         )

#         shape = input_block.shape[:-1]
#         labeled_block = da.from_delayed(labeled_block, shape=shape, dtype=np.int32)

#         n = dask.delayed(np.int32)(n)
#         n = da.from_delayed(n, shape=(), dtype=np.int32)

#         total = n if total is None else total + n

#         block_label_offset = da.where(labeled_block > 0, total, np.int32(0))
#         labeled_block += block_label_offset

#         labeled_blocks[index[:-1]] = labeled_block
#         total += n

#     # Put all the blocks together
#     block_labeled = da.block(labeled_blocks.tolist())

#     depth = da.overlap.coerce_depth(len(depth), depth)

#     return block_labeled, total, depth


# def overlap_segmentation_to_zarr(
#     image,# image with overlap, dask array
#     channels,
#     model_type,# path to model
#     diameter,
#     zarr_file,
#     min_size=20,
#     fast_mode=False,
#     use_anisotropy=True,
# ):
#     diameter_yx = diameter[1]
#     anisotropy = diameter[0] / diameter[1] if use_anisotropy else None
#     depth = tuple(np.ceil(diameter).astype(np.int64))
#     boundary = "reflect"

#     block_iter = zip(
#         np.ndindex(*image.numblocks),
#         map(
#             functools.partial(operator.getitem, image),
#             da.core.slices_from_chunks(image.chunks),
#         ),
#     )

#     # labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
#     total = None
#     chunk_info = image.chunks
    
#     for index, input_block in block_iter:
#         labeled_block, n = dask.delayed(segment_chunk, nout=2)(
#             input_block,
#             channels,
#             model_type,
#             diameter_yx,
#             anisotropy,
#             fast_mode,
#             index,
#             min_size,
#         )
#         # for zarr indexing
#         p = slice(sum(chunk_info[0][:index[0]]),sum(chunk_info[0][:index[0]])+chunk_info[0][index[0]]) 
#         q = slice(sum(chunk_info[1][:index[1]]),sum(chunk_info[1][:index[1]])+chunk_info[1][index[1]])
#         r = slice(sum(chunk_info[2][:index[2]]),sum(chunk_info[2][:index[2]])+chunk_info[2][index[2]])


#         shape = input_block.shape[:-1]
#         labeled_block = da.from_delayed(labeled_block, shape=shape, dtype=np.int32)

#         n = dask.delayed(np.int32)(n)
#         n = da.from_delayed(n, shape=(), dtype=np.int32)

#         total = n if total is None else total + n

#         block_label_offset = da.where(labeled_block > 0, total, np.int32(0))
#         labeled_block += block_label_offset

#         # labeled_blocks[index[:-1]] = labeled_block
#         zarr_file[p,q,r] = labeled_block
#         total += n

#     # Put all the blocks together
#     # block_labeled = da.block(labeled_blocks.tolist())

#     depth = da.overlap.coerce_depth(len(depth), depth)

#     return total, depth

def relabel_segmentation(
    block_labeled,# segmentation with overlaps
    total,
    depth,
    iou_depth=2,
    iou_threshold=0.7,
):
    boundary = "reflect"

    if np.prod(block_labeled.numblocks) > 1:
        iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)

        if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
            raise DistSegError("iou_depth (%s) > depth (%s)" % (iou_depth, depth))

        trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
        block_labeled = da.overlap.trim_internal(
            block_labeled, trim_depth, boundary=boundary
        )
        block_labeled = link_labels(
            block_labeled,
            total,
            iou_depth,
            iou_threshold=iou_threshold,
        )

        block_labeled = da.overlap.trim_internal(
            block_labeled, iou_depth, boundary=boundary
        )

    else:
        block_labeled = da.overlap.trim_internal(
            block_labeled, depth, boundary=boundary
        )

    return block_labeled


def segment_chunk(
    chunk,
    channels,
    model_type,
    diameter_yx,
    anisotropy,
    fast_mode,
    index,
    min_size,
):
    """Perform segmentation on an individual chunk."""
    # Cellpose seems to have some randomness, which is made deterministic by using the block
    # details as a random seed.
    # np.random.seed(index)

    # from cellpose import models

    # model = models.Cellpose(gpu=True, model_type=model_type, net_avg=not fast_mode)
    model = model_type# models.CellposeModel(gpu=True, pretrained_model=model_type, net_avg=not fast_mode)
    
    # logger.info("Evaluating model")
    segments, _, _ = model.eval(
        chunk,
        channels=channels,
        z_axis=0,
        channel_axis=None,
        diameter=diameter_yx,
        do_3D=True,
        anisotropy=anisotropy,
        net_avg=not fast_mode,
        augment=not fast_mode,
        tile=not fast_mode,
        min_size=min_size,
    )
    # logger.info("Done segmenting chunk")

    return segments.astype(np.int32)# , segments.max()


def link_labels(block_labeled, total, depth, iou_threshold=1):
    """
    Build a label connectivity graph that groups labels across blocks,
    use this graph to find connected components, and then relabel each
    block according to those.
    """
    label_groups = label_adjacency_graph(block_labeled, total, depth, iou_threshold)
    new_labeling = _label.connected_components_delayed(label_groups)
    return _label.relabel_blocks(block_labeled, new_labeling)


def label_adjacency_graph(labels, nlabels, depth, iou_threshold):
    all_mappings = [da.empty((2, 0), dtype=np.int32, chunks=1)]

    slices_and_axes = get_slices_and_axes(labels.chunks, labels.shape, depth)
    for face_slice, axis in slices_and_axes:
        face = labels[face_slice]
        mapped = _across_block_iou_delayed(face, axis, iou_threshold)
        all_mappings.append(mapped)

    i, j = da.concatenate(all_mappings, axis=1)
    result = _label._to_csr_matrix(i, j, nlabels + 1)
    return result


def _across_block_iou_delayed(face, axis, iou_threshold):
    """Delayed version of :func:`_across_block_label_grouping`."""
    _across_block_label_grouping_ = dask.delayed(_across_block_label_iou)
    grouped = _across_block_label_grouping_(face, axis, iou_threshold)
    return da.from_delayed(grouped, shape=(2, np.nan), dtype=np.int32)


def _across_block_label_iou(face, axis, iou_threshold):
    unique = np.unique(face)
    face0, face1 = np.split(face, 2, axis)

    intersection = sk_metrics.confusion_matrix(face0.reshape(-1), face1.reshape(-1))
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)

    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection

    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0)

    labels0, labels1 = np.nonzero(iou >= iou_threshold)

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    grouped = np.stack([labels0_orig, labels1_orig])

    valid = np.all(grouped != 0, axis=0)  # Discard any mappings with bg pixels
    return grouped[:, valid]


def get_slices_and_axes(chunks, shape, depth):
    ndim = len(shape)
    depth = da.overlap.coerce_depth(ndim, depth)
    slices = da.core.slices_from_chunks(chunks)
    slices_and_axes = []
    for ax in range(ndim):
        for sl in slices:
            if sl[ax].stop == shape[ax]:
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(
                sl[ax].stop - 2 * depth[ax], sl[ax].stop + 2 * depth[ax]
            )
            slices_and_axes.append((tuple(slice_to_append), ax))
    return slices_and_axes