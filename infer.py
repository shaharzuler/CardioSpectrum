import os
import time
from argparse import ArgumentParser
import time
import shutil

from easydict import EasyDict
import torch
import numpy as np

import cardio_volume_skewer
import four_d_ct_cost_unrolling
import three_d_data_manager as dt_mng
import flow_n_corr_utils
import p2p_correspondence

from utils import json2dict, handle_pred_flow
import dataset_obj_adding as doa
from exp_managing.drawings.drawing_teaser_main_paper import draw_img_w_gt_and_preds


parser = ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs/config_templates/general_config_template.json") 
parser.add_argument("--cuda_device", type=str, default=0)#None)

args = parser.parse_args()

config = json2dict(args.config_path) 
if args.cuda_device is not None: # override json cuda device with cmd line cuda device
    config.general.cuda_device = int(args.cuda_device)

top_output_dir = config.general.top_output_dir 
outputs_path = os.path.join(top_output_dir, f"outputs_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(outputs_path, exist_ok=True)
print(f"Output directory: {outputs_path}")
dataset_target_path = os.path.join(top_output_dir, config.dataset.dataset_target_folder_name)   
template_timestep_name = config.dataset.template_timestep_name 
unlabeled_timestep_name = "28" 

dt_mng.write_config_file(outputs_path, "main_train", config)

torch.set_num_threads(config.general.torch_num_threads)

print("Init dataset")
dataset = dt_mng.Dataset(target_root_dir=dataset_target_path, file_paths=dt_mng.read_filepaths(dataset_target_path))
if config.dataset.template_dicom_path is not None: # magix
    abs_template_dicom_path = os.path.join(os.getcwd(), config.dataset.template_dicom_path)
    dataset = doa.add_template_image_from_dicom(dataset, template_timestep_name, abs_template_dicom_path) 
else: # mm-whs, 3d_slicer
    dataset = doa.add_image_from_xyz_arr(dataset, template_timestep_name, config.dataset.template_xyz_arr_path)
if config.dataset.template_zxy_voxels_mask_arr_path is not None: # magix
    dataset = doa.add_mask_from_zxy_arr(dataset, template_timestep_name, config.dataset.template_zxy_voxels_mask_arr_path, config.dataset.voxels_mask_smoothing, mask_or_extra_mask="mask")
    dataset = doa.add_mask_from_zxy_arr(dataset, template_timestep_name, config.dataset.template_zxy_voxels_extra_mask_arr_path, config.dataset.voxels_mask_smoothing, mask_or_extra_mask="extra_mask")
else: # mm-whs, 3d_slicer
    dataset = doa.add_mask_from_xyz_arr(dataset, template_timestep_name, os.path.join(os.getcwd(),config.dataset.template_xyz_voxels_mask_arr_path), config.dataset.voxels_mask_smoothing, mask_or_extra_mask="mask") 
    dataset = doa.add_mask_from_xyz_arr(dataset, template_timestep_name, os.path.join(os.getcwd(),config.dataset.template_xyz_voxels_extra_mask_arr_path), config.dataset.voxels_mask_smoothing, mask_or_extra_mask="extra_mask")

template_3dimg_path = dataset.file_paths.xyz_arr[template_timestep_name] 
template_mask_path = dataset.file_paths.xyz_voxels_mask_smooth[template_timestep_name]
template_extra_mask_path = dataset.file_paths.xyz_voxels_extra_mask_smooth[template_timestep_name]

print("Generating synthetic dataset")
template_synthetic_img_path, unlabeled_synthetic_img_path, \
    template_synthetic_mask_path, unlabeled_synthetic_mask_path, \
        template_synthetic_extra_mask_path, unlabeled_synthetic_extra_mask_path, synthetic_flow_path, \
        error_radial_coordinates_path, error_circumferential_coordinates_path, error_longitudinal_coordinates_path = \
    cardio_volume_skewer.create_skewed_sequences(
        r1s_end=config.synthetic_data.r1, 
        r2s_end=config.synthetic_data.r2, 
        theta1s_end=config.synthetic_data.theta1, 
        theta2s_end=config.synthetic_data.theta2, 
        hs_end=config.synthetic_data.h, 
        output_dir=dataset_target_path, 
        template_3dimage_path=template_3dimg_path, 
        template_mask_path=template_mask_path,
        template_extra_mask_path=template_extra_mask_path,
        num_frames=6,
        zero_outside_mask=config.synthetic_data.zero_outside_mask,
        blur_around_mask_radious=config.synthetic_data.blur_around_mask_radious,
        theta_distribution_method=config.synthetic_data.theta_distribution_method,
        scale_down_by=1
        ) 

print("Processing synthetic dataset")
dataset = doa.add_image_from_xyz_arr(dataset, unlabeled_timestep_name, unlabeled_synthetic_img_path)
dataset = doa.add_mask_from_xyz_arr( dataset, unlabeled_timestep_name, unlabeled_synthetic_mask_path, config.dataset.voxels_mask_smoothing, mask_or_extra_mask="mask")
dataset = doa.add_mask_from_xyz_arr( dataset, unlabeled_timestep_name, unlabeled_synthetic_extra_mask_path, config.dataset.voxels_mask_smoothing, mask_or_extra_mask="extra_mask")

unlabeled_3dimg_path = dataset.file_paths.xyz_arr[unlabeled_timestep_name]
unlabeled_mask_path = dataset.file_paths.xyz_voxels_mask_smooth[unlabeled_timestep_name]
unlabeled_extra_mask_path = dataset.file_paths.xyz_voxels_extra_mask_smooth[unlabeled_timestep_name] 

print("Creating meshes")
for timestep_name in unlabeled_timestep_name, template_timestep_name:
    print("Creating mesh")
    mesh_creation_args = dt_mng.MeshSmoothingCreationArgs(marching_cubes_step_size=2) #1) 
    mesh_data_creator = dt_mng.MeshDataCreator(source_path=None, sample_name=timestep_name, hirarchy_levels=2, creation_args=mesh_creation_args)
    dataset.add_sample(mesh_data_creator) 

    print("Creating LBOs from mesh")
    lbo_creation_args = dt_mng.LBOCreationArgs(num_LBOs=config.dataset.num_lbos, is_point_cloud=False, geometry_path=dataset.file_paths.mesh[timestep_name], orig_geometry_name="mesh", use_torch=True)
    lbos_data_creator = dt_mng.LBOsDataCreator(source_path=None, sample_name=timestep_name, hirarchy_levels=2, creation_args=lbo_creation_args)
    dataset.add_sample(lbos_data_creator)
    if config.dataset.mesh_smoothing:
        print("Smoothing mesh with lbos")
        smooth_mesh_creation_args = dt_mng.SmoothMeshCreationArgs(lbos_path=dataset.file_paths.mesh_lbo_data[timestep_name])
        smooth_lbo_mesh_data_creator = dt_mng.SmoothLBOMeshDataCreator(source_path=None, sample_name=timestep_name, hirarchy_levels=2, creation_args=smooth_mesh_creation_args)
        dataset.add_sample(smooth_lbo_mesh_data_creator)
    print("Computing vertex normals")
    vertex_normals_creation_args = dt_mng.VertexNormalsCreationArgs(geometry_path=dataset.file_paths.mesh_smooth[timestep_name], orig_geometry_name="mesh_smooth" if config.dataset.mesh_smoothing else "mesh") 
    vertex_normals_data_creator = dt_mng.VertexNormalsDataCreator(source_path=None, sample_name=timestep_name, hirarchy_levels=2, creation_args=vertex_normals_creation_args)
    dataset.add_sample(vertex_normals_data_creator)

mesh_filename = "smooth_mesh" if config['dataset'].mesh_smoothing else "mesh"
template_mesh_path = os.path.join(dataset_target_path, template_timestep_name, f"orig/meshes/{mesh_filename}.off") 
unlabeled_mesh_path = os.path.join(dataset_target_path, unlabeled_timestep_name, f"orig/meshes/{mesh_filename}.off")
template_normals_path = os.path.join(dataset_target_path, template_timestep_name, f"orig/vertices_normals/vertices_normals_from_{'mesh_smooth' if config['dataset'].mesh_smoothing else 'mesh'}.npy") 

print("Using ZoomOut")
config_zoomout = p2p_correspondence.get_default_config()
config_zoomout["plots"] = True
config_zoomout["main_output_dir"] = os.path.join(outputs_path, "zoomout_output_dir")
config_zoomout["default_output_subdir"] = config.dataset.dataset_target_folder_name
config_zoomout["process_params"]["descr_type"] = config.zoomout.descriptor_type
config_zoomout["process_params"]["n_ev"] = config.zoomout.num_eigenvectors
config_zoomout["process_params"]["n_descr"] = config.zoomout.num_preprocess_descriptors
config_zoomout["fm_fit_params"]["optinit"] = config.zoomout.optinit
config_zoomout["fm_fit_params"]["w_descr"] = config.zoomout.w_descr
config_zoomout["fm_fit_params"]["w_lap"] = config.zoomout.w_lap
config_zoomout["zoomout_refine_params"]["nit"] = config.zoomout.num_zoomout_iters
config_zoomout["preprocess"]["normalize_meshes_area"] = config.zoomout.normalize_meshes_area
config_zoomout["validation"]["mean_l1_flow_th"] = config.zoomout.mean_l1_flow_th

corr_infer_output_path, valid_flow = p2p_correspondence.get_correspondence(
    mesh1_path=template_mesh_path, 
    mesh2_path=unlabeled_mesh_path,
    config=EasyDict(config_zoomout)
    )
if not(valid_flow):
    print("Retrying ZoomOut with more eigenvectors")
    config_zoomout["process_params"]["n_ev"] = config.zoomout.num_eigenvectors_for_2nd_try
    corr_infer_output_path, valid_flow = p2p_correspondence.get_correspondence(
        mesh1_path=template_mesh_path, 
        mesh2_path=unlabeled_mesh_path,
        config=EasyDict(config_zoomout)
        )

print("Converting correspondence to constraints")
sample_shape = dataset.get_xyz_arr(template_timestep_name).shape
config.constraints_creation.confidence_matrix_manipulations_config["plot_folder"] = outputs_path

two_d_constraints_path = flow_n_corr_utils.convert_corr_to_constraints(
    correspondence_h5_path=os.path.join(corr_infer_output_path, "model_inference.hdf5"),
    k_nn=config.constraints_creation.k_smooth_constraints_nn,
    output_folder_path=outputs_path,
    output_constraints_shape=(*sample_shape, 3),
    k_interpolate_sparse_constraints_nn=config.constraints_creation.k_interpolate_sparse_constraints_nn,
    confidence_matrix_manipulations_config=config.constraints_creation.confidence_matrix_manipulations_config
    )

voxelized_normals_path = flow_n_corr_utils.voxelize_and_visualize_3d_vecs(
    vectors_cloud=np.load(template_normals_path), 
    point_cloud=dt_mng.read_off(template_mesh_path)[0], 
    output_shape=(*sample_shape, 3), 
    text_vis="normals", 
    output_arr_filename="normals", 
    output_folder=outputs_path
)

print("Training without constraints")
config_backbone = four_d_ct_cost_unrolling.get_default_backbone_config()
config_backbone["save_iter"] = 2
config_backbone["inference_args"]["inference_flow_median_filter_size"] = False
config_backbone["epochs"] = config.fourD_ct_cost_unrolling.backbone.early_stopping.epochs 
config_backbone["valid_type"] = "synthetic+basic"
config_backbone["w_sm_scales"] = config.fourD_ct_cost_unrolling.backbone["w_sm_scales"]
config_backbone["output_root"] = os.path.join(outputs_path, config_backbone["output_root"])
config_backbone["visualization_arrow_scale_factor"] = 1
config_backbone["cuda_device"] = config.general.cuda_device 
config_backbone["scale_down_by"] = config.fourD_ct_cost_unrolling.backbone.scale_down_by
config_backbone["metric_for_early_stopping"] = config.fourD_ct_cost_unrolling.backbone.early_stopping.metric_for_early_stopping 
config_backbone["max_metric_not_dropping_patience"] = config.fourD_ct_cost_unrolling.backbone.early_stopping.max_metric_not_dropping_patience 

backbone_model_output_path = four_d_ct_cost_unrolling.overfit_backbone(
    template_image_path=unlabeled_synthetic_img_path,
    unlabeled_image_path=template_synthetic_img_path,
    template_LV_seg_path=unlabeled_synthetic_mask_path,
    unlabeled_LV_seg_path=template_synthetic_mask_path,
    template_shell_seg_path=unlabeled_synthetic_extra_mask_path,
    unlabeled_shell_seg_path=template_synthetic_extra_mask_path,
    flows_gt_path=synthetic_flow_path,
    error_radial_coordinates_path=error_radial_coordinates_path,
    error_circumferential_coordinates_path=error_circumferential_coordinates_path,
    error_longitudinal_coordinates_path=error_longitudinal_coordinates_path,  
    voxelized_normals_path=voxelized_normals_path, 
    args=EasyDict(config_backbone)
    )

print("Training with constraints loss")
config_constraints = four_d_ct_cost_unrolling.get_default_w_constraints_config()
config_constraints["save_iter"] = 2
config_constraints["inference_args"]["inference_flow_median_filter_size"] = False
config_constraints["epochs"] = config.fourD_ct_cost_unrolling.w_constraints.early_stopping.epochs 
config_constraints["valid_type"] = "synthetic+basic"
config_constraints["w_sm_scales"] = config.fourD_ct_cost_unrolling.w_constraints["w_sm_scales"]
config_constraints["output_root"] = os.path.join(outputs_path, config_constraints["output_root"])
config_constraints["visualization_arrow_scale_factor"] = 1
config_constraints["w_constraints_scales"] = [100.0, 100.0, 100.0, 100.0, 100.0]
config_constraints["cuda_device"] = config.general.cuda_device
config_constraints["scale_down_by"] = config.fourD_ct_cost_unrolling.w_constraints.scale_down_by
config_constraints["metric_for_early_stopping"] = config.fourD_ct_cost_unrolling.w_constraints.early_stopping.metric_for_early_stopping 
config_constraints["max_metric_not_dropping_patience"] = config.fourD_ct_cost_unrolling.w_constraints.early_stopping.max_metric_not_dropping_patience 
config_constraints["load"] = four_d_ct_cost_unrolling.get_checkpoints_path(backbone_model_output_path)

constraints_model_output_path = four_d_ct_cost_unrolling.overfit_w_constraints(
    template_image_path=unlabeled_synthetic_img_path,
    unlabeled_image_path=template_synthetic_img_path,
    template_LV_seg_path=unlabeled_synthetic_mask_path,
    unlabeled_LV_seg_path=template_synthetic_mask_path,
    template_shell_seg_path=unlabeled_synthetic_extra_mask_path,
    unlabeled_shell_seg_path=template_synthetic_extra_mask_path,
    two_d_constraints_path=two_d_constraints_path,
    flows_gt_path=synthetic_flow_path,        
    error_radial_coordinates_path=error_radial_coordinates_path,
    error_circumferential_coordinates_path=error_circumferential_coordinates_path,
    error_longitudinal_coordinates_path=error_longitudinal_coordinates_path,  
    voxelized_normals_path=voxelized_normals_path,
    args=EasyDict(config_constraints)
    )
print(f"Constraints model output path: {constraints_model_output_path}")
config_constraints["load"] = four_d_ct_cost_unrolling.get_checkpoints_path(constraints_model_output_path)

infer_constraints_model_output_path = four_d_ct_cost_unrolling.infer_w_constraints(
    template_image_path=unlabeled_synthetic_img_path,
    unlabeled_image_path=template_synthetic_img_path,
    template_LV_seg_path=unlabeled_synthetic_mask_path,
    unlabeled_LV_seg_path=template_synthetic_mask_path,
    template_shell_seg_path=unlabeled_synthetic_extra_mask_path,
    unlabeled_shell_seg_path=template_synthetic_extra_mask_path,
    two_d_constraints_path=two_d_constraints_path,
    flows_gt_path=synthetic_flow_path,  
    save_mask=True,      
    args=EasyDict(config_constraints)
)

print("Drawing")
for_drawing_path = os.path.join(outputs_path, "for_drawing")
os.makedirs(for_drawing_path, exist_ok=True)
handle_pred_flow(config.fourD_ct_cost_unrolling.w_constraints.scale_down_by, infer_constraints_model_output_path, for_drawing_path) # shape 3xyz

shutil.copyfile(synthetic_flow_path, os.path.join(for_drawing_path,"ground_truth_flow.npy")) # shape xyz3
shutil.copyfile(template_synthetic_img_path, os.path.join(for_drawing_path,"template_img.npy")) # shape xyz
shutil.copyfile(unlabeled_synthetic_img_path, os.path.join(for_drawing_path,"unlabeled_img.npy")) # shape xyz
shutil.copyfile(template_synthetic_mask_path, os.path.join(for_drawing_path,"template_mask.npy")) # shape xyz
shutil.copyfile(unlabeled_synthetic_mask_path, os.path.join(for_drawing_path,"unlabeled_mask.npy")) # shape xyz
draw_img_w_gt_and_preds(
    bb_pred_path=None, 
    constraints_pred_path=os.path.join(for_drawing_path, "pred_flow.npy"), 
    gt_path=os.path.join(for_drawing_path,"ground_truth_flow.npy"), 
    unlabeled_img_path=os.path.join(for_drawing_path,"unlabeled_img.npy"), 
    unlabeled_mask_path=os.path.join(for_drawing_path,"unlabeled_mask.npy"),
    output_root=for_drawing_path
    )

print("Outputs:")
print(os.path.join(for_drawing_path, "w_constraints_pred_vs_gt.png"))
print(os.path.join(for_drawing_path, "pred_flow.npy"))