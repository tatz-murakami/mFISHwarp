import SimpleITK as sitk
import pydeform.sitk_api as pydeform
import dask.array as da
import zarr
import numpy as np
import mFISHwarp.transform
import mFISHwarp.utils


def affine_registration(array1, array2, initial_rotation=None, shrinking_factors=(16, 8, 4, 2), smoothing = (4, 2, 1, 0)):
    """
    Arguments:
        array1 (ndarray): fixed image
        array2 (ndarray): moving image
        initial_rotation: initial rotation degree in x, y, z.
            if you want to rotate around z for pi, (0,1,np.pi). In case of y, (0,2,np.pi). help(initial_transform.Rotate())
    Return:
        affine transformation (itk.AffineTransform):
    """

    fix_itk = sitk.Cast(sitk.GetImageFromArray(array1), sitk.sitkFloat32)
    mov_itk = sitk.Cast(sitk.GetImageFromArray(array2), sitk.sitkFloat32)

    initial_transform = sitk.AffineTransform(sitk.CenteredTransformInitializer(fix_itk,
                                                                               mov_itk,
                                                                               sitk.AffineTransform(3),
                                                                               sitk.CenteredTransformInitializerFilter.GEOMETRY))
    if initial_rotation is not None:
        initial_transform.Rotate(*initial_rotation)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinking_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    affine_transform = registration_method.Execute(fix_itk, mov_itk)
    affine_transform = sitk.AffineTransform(affine_transform)

    return affine_transform


def affine_warping(array1, array2, affine):
    """
    Arguments:
        array1 (ndarray): fixed image
        array2 (ndarray): moving image
        affine (itk.AffineTransform):
    Return:
        ndarray:
    """
    fix_itk = sitk.Cast(sitk.GetImageFromArray(array1), sitk.sitkFloat32)
    mov_itk = sitk.Cast(sitk.GetImageFromArray(array2), sitk.sitkFloat32)

    transformed = sitk.Resample(mov_itk, fix_itk, affine, sitk.sitkLinear, 0.0, mov_itk.GetPixelID())

    return sitk.GetArrayFromImage(transformed)


def deform_registration(array1, array2, settings, affine_transform=None, log_path=None, num_threads=-1, use_gpu=True, return_warped=False):
    """
    Arguments:
        array1 (ndarray): fixed image
        array2 (ndarray): moving image
        settings: settings for pydeform
    Return:
        displacement (simple itk Image object):

    """
    fix_itk = sitk.Cast(sitk.GetImageFromArray(array1), sitk.sitkFloat32)
    mov_itk = sitk.Cast(sitk.GetImageFromArray(array2), sitk.sitkFloat32)

    displacement = pydeform.register(
        fix_itk,
        mov_itk,
        settings=settings,
        log=log_path,
        log_level=pydeform.LogLevel.Verbose,
        num_threads=num_threads,
        affine_transform=affine_transform,
        use_gpu=use_gpu
    )

    if return_warped:
        mov_deformed = pydeform.transform(mov_itk, displacement)
        return displacement, sitk.GetArrayFromImage(mov_deformed)
    else:
        return displacement


def deform_warping(mov, displacement):
    """
    Arguments:
        mov (ndarray): moving image
        displacement (): displacement field
    Return:
        ndarray
    """
    mov_itk = sitk.Cast(sitk.GetImageFromArray(mov), sitk.sitkFloat32)
    mov_deformed = pydeform.transform(mov_itk, displacement)

    return sitk.GetArrayFromImage(mov_deformed)


def convert_disp2array(displacement):
    """
    Arguments:
        displacement (SimpleITK.Image):
    """
    return sitk.GetArrayFromImage(displacement)


def chunk_wise_registration(chunk_position, displacement_overlap, fix_overlap, mov_l, settings, displacement_zarr,
                            registered_mov_zarr, num_threads=1, use_gpu=True):
    """
    Arugments:
        chunk_position (tuple): the index of chunk
        displacement_overlap (dask array): overlapped displacement array
        fix_overlap (dask array): overlapped fixed array
        mov_l (zarr): zarr of moving image
        displacment_zarr (zarr): zarr to save displacement image
        regstered_mov_zarr (zarr): zarr to save registered moving image
    """

    chunk_size = mFISHwarp.utils.chunks_from_dask(displacement_overlap)[:-1]
    fix = fix_overlap[mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)].compute()
    disp = displacement_overlap[mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)].compute()

    mov = mFISHwarp.transform.transform_block_gpu(disp, mov_l)

    df_sitk, mov_deformed_overlap = mFISHwarp.register.deform_registration(fix, mov, settings, None, None,
                                                                        num_threads=num_threads, use_gpu=use_gpu,
                                                                        return_warped=True)

    # save deformed moving image
    if registered_mov_zarr is not None:
        shape = registered_mov_zarr.chunks
        crop = (np.asarray(mov_deformed_overlap.shape) - np.asarray(registered_mov_zarr.chunks)) // 2
        slicing1 = tuple(slice(i, i + j) for i, j in zip(crop, shape))
        mov_deformed = mov_deformed_overlap[slicing1].astype(np.uint16)
        slicing2 = tuple(slice(i * j, i * (j + 1)) for i, j in zip(shape, chunk_position))
        registered_mov_zarr[slicing2] = mov_deformed

    # convert relative displacement to positional displacement
    positional_df = mFISHwarp.transform.relative2positional_gpu(mFISHwarp.transform.displacement_itk2numpy(df_sitk))
    # composite two displacement field
    merged_displacement = mFISHwarp.transform.composite_displacement_gpu(disp, positional_df, order=1)

    # save dispalcement to zarr
    if displacement_zarr is not None:
        shape = displacement_zarr.chunks
        slicing = tuple(slice(i * j, i * (j + 1)) for i, j in zip(shape, chunk_position)) + (slice(None, None, None),)
        displacement_zarr[slicing] = merged_displacement


def chunk_wise_no_registration(chunk_position, displacement_overlap, displacement_zarr, registered_mov_zarr):
    """
    Arugments:
        chunk_position (tuple): the index of chunk
        displacement_overlap (dask array): overlapped displacement array
        displacment_zarr (zarr): zarr to save displacement image
        regstered_mov_zarr (zarr): zarr to save registered moving image
    """

    chunk_size = mFISHwarp.utils.chunks_from_dask(displacement_overlap)[:-1]

    disp = displacement_overlap[
        mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)
    ].compute()

    # save deformed moving image
    if registered_mov_zarr is not None:
        shape = registered_mov_zarr.chunks
        mov_deformed = np.zeros(registered_mov_zarr.chunks, dtype=np.uint16)
        slicing = tuple(slice(i * j, i * (j + 1)) for i, j in zip(shape, chunk_position))
        registered_mov_zarr[slicing] = mov_deformed

    # trim edges
    merged_displacement = disp

    # save dispalcement to zarr
    if displacement_zarr is not None:
        shape = displacement_zarr.chunks
        slicing = tuple(slice(i * j, i * (j + 1)) for i, j in zip(shape, chunk_position)) + (slice(None, None, None),)
        displacement_zarr[slicing] = merged_displacement