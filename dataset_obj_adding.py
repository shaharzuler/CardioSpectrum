import three_d_data_manager as dt_mng

# add 3D images:

def add_template_image_from_dicom(dataset, timestep_name, dicom_path):
    # add dicom
    template_dicom_data_creator = dt_mng.DicomDataCreator(source_path=dicom_path, sample_name=timestep_name, hirarchy_levels=2) 
    dataset.add_sample(template_dicom_data_creator)
    
    # create np arrays from dicom
    template_xyz_arr_data_creator = dt_mng.XYZArrDataCreator(source_path=None, sample_name=timestep_name, hirarchy_levels=2)
    dataset.add_sample(template_xyz_arr_data_creator)
    return dataset

def add_image_from_xyz_arr(dataset, timestep_name, xyz_arr_path):
    xyz_arr_data_creator = dt_mng.XYZArrDataCreator(source_path=xyz_arr_path, sample_name=timestep_name, hirarchy_levels=2)
    dataset.add_sample(xyz_arr_data_creator)
    return dataset


# add and smooth 3D masks:

def smooth_mask_spatially_and_add(dataset, timestep_name, voxels_mask_smoothing_args, mask_or_extra_mask):
    # smooth extra mask with voxels methods
    voxel_smoothing_args = dt_mng.VoxelSmoothingCreationArgs(voxels_mask_smoothing_args["opening_footprint_radius"], voxels_mask_smoothing_args["fill_holes_Area_threshold"],  voxels_mask_smoothing_args["closing_to_opening_ratio"])
    msk_data_creator = dt_mng.SmoothVoxelsExtraMaskDataCreator if mask_or_extra_mask=="extra_mask" else dt_mng.SmoothVoxelsMaskDataCreator
    smooth_voxel_extra_mask_data_creator = msk_data_creator(source_path=None, sample_name=timestep_name, hirarchy_levels=2, creation_args=voxel_smoothing_args)
    dataset.add_sample(smooth_voxel_extra_mask_data_creator)
    return dataset

def add_mask_from_xyz_arr(dataset, timestep_name, xyz_voxels_mask_arr_path, voxels_mask_smoothing_args, mask_or_extra_mask):
    # add segmentation mask
    msk_data_creator = dt_mng.XYZVoxelsExtraMaskDataCreator if mask_or_extra_mask=="extra_mask" else dt_mng.XYZVoxelsMaskDataCreator
    xyz_voxels_mask_data_creator = msk_data_creator(source_path=xyz_voxels_mask_arr_path, sample_name=timestep_name, hirarchy_levels=2)
    dataset.add_sample(xyz_voxels_mask_data_creator) 
    dataset = smooth_mask_spatially_and_add(dataset, timestep_name, voxels_mask_smoothing_args, mask_or_extra_mask=mask_or_extra_mask)
    return dataset

def add_mask_from_zxy_arr(dataset, timestep_name, zxy_voxels_mask_arr_path, voxels_mask_smoothing_args, mask_or_extra_mask):
    # add segmentation mask
    zxy_msk_data_creator = dt_mng.ZXYVoxelsExtraMaskDataCreator if mask_or_extra_mask == "extra_mask" else dt_mng.ZXYVoxelsMaskDataCreator
    zxy_voxels_mask_data_creator = zxy_msk_data_creator(source_path=zxy_voxels_mask_arr_path, sample_name=timestep_name, hirarchy_levels=2)
    dataset.add_sample(zxy_voxels_mask_data_creator)

    # zxy to xyz
    xyz_msk_data_creator = dt_mng.XYZVoxelsExtraMaskDataCreator if mask_or_extra_mask=="extra_mask" else dt_mng.XYZVoxelsMaskDataCreator
    xyz_voxels_mask_data_creator = xyz_msk_data_creator(source_path=None, sample_name=timestep_name, hirarchy_levels=2)
    dataset.add_sample(xyz_voxels_mask_data_creator) 
    dataset = smooth_mask_spatially_and_add(dataset, timestep_name, voxels_mask_smoothing_args, mask_or_extra_mask=mask_or_extra_mask)
    return dataset