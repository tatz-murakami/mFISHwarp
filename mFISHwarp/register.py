import SimpleITK as sitk
import pydeform.sitk_api as pydeform
import dask.array as da
import zarr
import numpy as np
import mFISHwarp.transform
import mFISHwarp.utils


def affine_registration(array1, array2, initial_rotation=None, initial_scaling=None, shrinking_factors=(16, 8, 4, 2), smoothing = (4, 2, 1, 0), steplength=0.1, num_iter=25, model='affine'):
    """
    Arguments:
        array1 (ndarray): fixed image
        array2 (ndarray): moving image
        initial_rotation: initial rotation degree in x, y, z.
            if you want to rotate around z for pi, (0,1,np.pi). In case of y, (0,2,np.pi). help(initial_transform.Rotate())
        initial_scaling (float): initial scaling in x,y,z.
    Return:
        affine transformation (itk.AffineTransform):
    """
    if model != 'affine' and model != 'rigid' and model != 'similarity':
        raise ValueError("Model has to be either affine, rigid or similarity.") 

    fix_itk = sitk.Cast(sitk.GetImageFromArray(array1), sitk.sitkFloat32)
    mov_itk = sitk.Cast(sitk.GetImageFromArray(array2), sitk.sitkFloat32)
    
    if model=='affine':
        initial_transform = sitk.AffineTransform(fix_itk.GetDimension())
        if initial_scaling is not None:
            scaling_matrix = np.diag(initial_scaling)
            initial_transform.SetMatrix(scaling_matrix.flatten())
        initial_transform = sitk.AffineTransform(sitk.CenteredTransformInitializer(fix_itk,
                                                                                   mov_itk,
                                                                                   initial_transform,
                                                                                   sitk.CenteredTransformInitializerFilter.GEOMETRY))
        
    elif model == 'rigid':
        initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fix_itk,
                                                                                   mov_itk,
                                                                                   sitk.Euler3DTransform(),
                                                                                   sitk.CenteredTransformInitializerFilter.GEOMETRY))
    elif model == 'similarity':
        initial_transform = sitk.Similarity3DTransform()
        if initial_scaling is not None:
            scaling_matrix = np.diag(initial_scaling)
            initial_transform.SetMatrix(scaling_matrix.flatten())
        initial_transform = sitk.Similarity3DTransform(sitk.CenteredTransformInitializer(fix_itk,
                                                                                   mov_itk,
                                                                                   initial_transform,
                                                                                   sitk.CenteredTransformInitializerFilter.GEOMETRY))
        
    if initial_rotation is not None:
        initial_transform.Rotate(*initial_rotation)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsPowell(stepLength=steplength, numberOfIterations=num_iter)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinking_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    final_transform = registration_method.Execute(fix_itk, mov_itk)
    if model=='affine':
        final_transform = sitk.AffineTransform(final_transform)
    elif model =='rigid':
        final_transform = sitk.Euler3DTransform(final_transform)
    elif model =='similarity':
        final_transform = sitk.Similarity3DTransform(final_transform)

    return final_transform


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


def chunk_wise_registration(chunk_position, displacement_da, fix_da, mov_zarr, settings, displacement_zarr,
                            registered_mov_zarr, num_threads=1, use_gpu=True):
    """
    Arugments:
        chunk_position (tuple): the index of chunk
        displacement_da (dask array): positional displacement array. usually it is overlapped dask array.
        fix_da (dask array): fixed array. usually it is overlapped dask array.
        mov_zarr (zarr): zarr of moving image
        displacment_zarr (zarr): zarr to save displacement image
        regstered_mov_zarr (zarr): zarr to save registered moving image
    """

    chunk_size = mFISHwarp.utils.chunks_from_dask(displacement_da)[:-1]
    fix = fix_da[mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)].compute()
    target_shape = fix.shape
    disp = displacement_da[mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)].compute()
    disp = mFISHwarp.transform.pad_trim_array_to_size(disp, target_shape+(len(target_shape),), mode='edge')

    mov = mFISHwarp.transform.transform_block_gpu(disp, mov_zarr)

    # non-linear registration
    df_sitk, mov_deformed_overlap = mFISHwarp.register.deform_registration(fix, mov, settings, None, None,
                                                                        num_threads=num_threads, use_gpu=use_gpu,
                                                                        return_warped=True)

    # save deformed moving image
    if registered_mov_zarr is not None:
        # following assumes overlapped input of displacement_da nad fix_da.
        # this should work without overlap.
        chunks = da.from_zarr(registered_mov_zarr).chunks
        shape = [i[j] for i, j in zip(chunks, chunk_position)]
        crop = (np.asarray(mov_deformed_overlap.shape) - np.asarray(shape)) // 2 
        slicing1 = tuple(slice(i, i + j) for i, j in zip(crop, shape))
        mov_deformed = mov_deformed_overlap[slicing1].astype(np.uint16)
        
        slicing2 = mFISHwarp.utils.obtain_chunk_slicer(chunks, chunk_position)
        registered_mov_zarr[slicing2] = mov_deformed

    # convert relative displacement to positional displacement
    positional_df = mFISHwarp.transform.relative2positional_gpu(mFISHwarp.transform.displacement_itk2numpy(df_sitk))
    # composite two displacement field
    merged_displacement = mFISHwarp.transform.composite_displacement_gpu(disp, positional_df, order=1)

    # save dispalcement to zarr
    if displacement_zarr is not None:
        chunks = da.from_zarr(displacement_zarr).chunks
        slicing = mFISHwarp.utils.obtain_chunk_slicer(chunks,chunk_position) + (slice(None, None, None),)
        displacement_zarr[slicing] = merged_displacement
        
        
def chunk_wise_affine_deform_registration(chunk_position, displacement_da, fix_da, mov_zarr, settings, displacement_zarr,
                            registered_mov_zarr, *args, shrinking_factors=(32, 16, 8, 4), smoothing=(4, 4, 2, 1), num_threads=1, use_gpu=True, **kwargs):
    """
    Arugments:
        chunk_position (tuple): the index of chunk
        displacement_da (dask array): positional displacement array. usually it is overlapped dask array.
        fix_da (dask array): fixed array. usually it is overlapped dask array.
        mov_zarr (zarr): zarr of moving image
        displacment_zarr (zarr): zarr to save displacement image
        regstered_mov_zarr (zarr): zarr to save registered moving image
    """

    chunk_size = mFISHwarp.utils.chunks_from_dask(displacement_da)[:-1]
    fix = fix_da[mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)].compute()
    target_shape = fix.shape
    disp = displacement_da[mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)].compute()
    disp = mFISHwarp.transform.pad_trim_array_to_size(disp, target_shape+(len(target_shape),), mode='edge')

    mov = mFISHwarp.transform.transform_block_gpu(disp, mov_zarr)
    
    # affine registration
    affine_transform = mFISHwarp.register.affine_registration(
        fix, mov,
        initial_rotation=None,
        initial_scaling=None,
        shrinking_factors=shrinking_factors, # shrinking factor determine the resolution of the registration.
        smoothing=smoothing,
        model='affine'
    )
    
    # non-linear registration
    df_sitk, mov_deformed_overlap = mFISHwarp.register.deform_registration(fix, mov, settings, affine_transform, None,
                                                                        num_threads=num_threads, use_gpu=use_gpu,
                                                                        return_warped=True)

    # save deformed moving image
    if registered_mov_zarr is not None:
        # following assumes overlapped input of displacement_da nad fix_da.
        # this should work without overlap.
        chunks = da.from_zarr(registered_mov_zarr).chunks
        shape = [i[j] for i, j in zip(chunks, chunk_position)]
        crop = (np.asarray(mov_deformed_overlap.shape) - np.asarray(shape)) // 2 
        slicing1 = tuple(slice(i, i + j) for i, j in zip(crop, shape))
        mov_deformed = mov_deformed_overlap[slicing1].astype(np.uint16)
        
        slicing2 = mFISHwarp.utils.obtain_chunk_slicer(chunks, chunk_position)
        registered_mov_zarr[slicing2] = mov_deformed

    # convert relative displacement to positional displacement
    positional_df = mFISHwarp.transform.relative2positional_gpu(mFISHwarp.transform.displacement_itk2numpy(df_sitk))
    # composite two displacement field
    merged_displacement = mFISHwarp.transform.composite_displacement_gpu(disp, positional_df, order=1)

    # save dispalcement to zarr
    if displacement_zarr is not None:
        chunks = da.from_zarr(displacement_zarr).chunks
        slicing = mFISHwarp.utils.obtain_chunk_slicer(chunks,chunk_position) + (slice(None, None, None),)
        displacement_zarr[slicing] = merged_displacement


def chunk_wise_no_registration(chunk_position, displacement_da, fix_da, displacement_zarr, registered_mov_zarr):
    """
    Arugments:
        chunk_position (tuple): the index of chunk
        displacement_da (dask array): positional displacement array.
        fix_da (dask array): fixed array
        displacement_zarr (zarr): zarr to save displacement image
        regstered_mov_zarr (zarr): zarr to save registered moving image
    """

    chunk_size = mFISHwarp.utils.chunks_from_dask(displacement_da)[:-1]
    target_shape = fix_da[mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)].shape

    disp = displacement_da[
        mFISHwarp.utils.chunk_slicer(chunk_position, chunk_size)
    ].compute()
    
    disp = mFISHwarp.transform.pad_trim_array_to_size(disp, target_shape+(len(target_shape),), mode='edge')

    # save deformed moving image. Zero array.
    if registered_mov_zarr is not None:
        chunks = da.from_zarr(registered_mov_zarr).chunks
        slicing = mFISHwarp.utils.obtain_chunk_slicer(chunks,chunk_position)
        mov_deformed = np.zeros([i[j] for i,j in zip(chunks,chunk_position)], dtype=np.uint16)
        registered_mov_zarr[slicing] = mov_deformed

    # save dispalcement to zarr
    if displacement_zarr is not None:
        chunks = da.from_zarr(displacement_zarr).chunks
        slicing = mFISHwarp.utils.obtain_chunk_slicer(chunks,chunk_position) + (slice(None, None, None),)
        displacement_zarr[slicing] = disp