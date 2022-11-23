# mFISHwarp
Image registration tool for large light-sheet data

Relevant paper: 
> Murakami and Heintz. Multiplexed and scalable cellular phenotyping toward the standardized three-dimensional human neuroanatomy. bioRxiv, 2022


### Software prerequisites 

Tested on Ubuntu 20.04 LST with the following versions of software.
- Cmake 3.21.3
- Python 3.8.12
- CUDA Toolkit 11.6
- [deform](https://github.com/simeks/deform)
- ISPC 1.16.1

### Source data prerequisites 

One pair of images to get registered. 
The pipeline assumes the input image format is in chunked format such as "zarr" or "hdf5" with pyramid resolution. 

This pipeline was inspired by [bigstream](https://github.com/GFleishman/bigstream) and depends on the library [dask_stitch](https://github.com/GFleishman/dask_stitch). 

### Installing python packages
```
conda env create -f environment.yml
```


### Usage
See the ipython notebooks.
