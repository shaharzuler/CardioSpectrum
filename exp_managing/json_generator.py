import json
import os

tot_torsions = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0]

def generate_scan_jsons(top_jsons_dir, template_json_path):
    top_ds_name = "" if "magix" in template_json_path else "mm_whs"
    with open(template_json_path) as file:
        scan_json = json.load(file)
    orig_output_dir = scan_json["general"]["top_output_dir"]
    for tot_torsion in tot_torsions:
        torsion_versions = [0] if tot_torsion == 0.0 else [0, 1, 2, 3, 4]
        for torsion_version in torsion_versions:
            exp_name = f"tot_torsion_{int(tot_torsion)}_torsion_version_{torsion_version}"
            scan_json["general"]["cuda_device"] = torsion_version
            scan_json["general"]["top_output_dir"] = os.path.join(orig_output_dir, exp_name)
            scan_json["dataset"]["dataset_target_folder_name"] = f"dataset_{exp_name}"
            scan_json["dataset"]["num_lbos"] = 200
            if torsion_version == 0:
                scan_json["synthetic_data"]["theta1"] = 0.0
                scan_json["synthetic_data"]["theta2"] = -tot_torsion
            elif torsion_version == 1:
                scan_json["synthetic_data"]["theta1"] = tot_torsion * 0.25
                scan_json["synthetic_data"]["theta2"] = -tot_torsion * 0.75
            elif torsion_version == 2:
                scan_json["synthetic_data"]["theta1"] = tot_torsion/2
                scan_json["synthetic_data"]["theta2"] = -tot_torsion/2
            elif torsion_version == 3:
                scan_json["synthetic_data"]["theta1"] = tot_torsion * 0.75
                scan_json["synthetic_data"]["theta2"] = -tot_torsion * 0.25
            elif torsion_version == 4:
                scan_json["synthetic_data"]["theta1"] = tot_torsion
                scan_json["synthetic_data"]["theta2"] = 0.0

            sample_name = top_jsons_dir.split("/")[-1]
            actual_exp_path = os.path.join("/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments", top_ds_name, sample_name, exp_name)
            prefix = f"{actual_exp_path}/dataset_{exp_name}/thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1"
            scan_json["levels"]["generate_synth_dataset_args"]["unlabeled_synthetic_img_path"]           = f"{prefix}/image_skewed_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["template_synthetic_img_path"]            = f"{prefix}/image_orig_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["unlabeled_synthetic_mask_path"]          = f"{prefix}/mask_skewed_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["template_synthetic_mask_path"]           = f"{prefix}/mask_orig_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["unlabeled_synthetic_extra_mask_path"]    = f"{prefix}/extra_mask_skewed_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["template_synthetic_extra_mask_path"]     = f"{prefix}/extra_mask_orig_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["synthetic_flow_path"]                    = f"{prefix}/flow_for_image_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["error_radial_coordinates_path"]          = f"{prefix}/error_radial_coordinates_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["error_circumferential_coordinates_path"] = f"{prefix}/error_circumferential_coordinates_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            scan_json["levels"]["generate_synth_dataset_args"]["error_longitudinal_coordinates_path"]    = f"{prefix}/error_longitudinal_coordinates_thetas_{scan_json['synthetic_data']['theta1']}_{scan_json['synthetic_data']['theta2']}_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
            if os.path.isdir(actual_exp_path):
                outs = [d for d in os.listdir(actual_exp_path) if "outputs" in d]
                outs.sort()
                outs = outs[::-1]
                constraints_flag = False
                bb_flag = False
                normals_flag = False
                for o in outs:
                    if not(constraints_flag):
                        if os.path.isfile(os.path.join(actual_exp_path, o, "constraints.npy")):
                            scan_json["levels"]["corr2constraints_args"]["two_d_constraints_path"] = os.path.join(actual_exp_path, o, "constraints.npy")
                            constraints_flag = True
                    if not(bb_flag):
                        for dd in os.listdir(os.path.join(actual_exp_path, o)):
                            if "backbone_training" in dd:
                                if os.path.isfile( os.path.join(actual_exp_path, o, dd, "checkpoints", "4dct_costunrolling_model_best.pth.tar")):
                                    scan_json["levels"]["four_d_ct_cost_unrolling_backbone_train_args"]["backbone_model_output_path"] = os.path.join(actual_exp_path, o, dd )
                                    bb_flag = True
                    if not(normals_flag): 
                         if os.path.isfile(os.path.join(actual_exp_path, o, "normals.npy")):
                            scan_json["levels"]["cont_normals2discrete_args"]["voxelized_normals_path"] = os.path.join(actual_exp_path, o, "normals.npy")
                            normals_flag = True
            else:
                print(f"prev exp doesnt exist, {actual_exp_path}")

            constraints_and_zoomout_only = False
            if constraints_and_zoomout_only:
                scan_json["levels"]["generate_synth_dataset"] = False
                scan_json["levels"]["process_synth_dataset"] = False
                scan_json["levels"]["four_d_ct_cost_unrolling_backbone_train"] = False
                scan_json["levels"]["four_d_ct_cost_unrolling_w_segmentation_train"] = False
            
            eval_only = False
            if eval_only:
                scan_json["levels"]["generate_synth_dataset"] = False
                scan_json["levels"]["process_synth_dataset"] = False    
                scan_json["levels"]["use_zoomout"] = False
                scan_json["levels"]["corr2constraints"] = False
                scan_json["levels"]["cont_normals2discrete"] = False
                scan_json["levels"]["four_d_ct_cost_unrolling_backbone_train"] = True
                scan_json["levels"]["four_d_ct_cost_unrolling_w_segmentation_train"] = True

            with open(os.path.join(top_jsons_dir, f"{exp_name}.json"), "w") as f:
                f.write(json.dumps(scan_json, indent=4) )

top_evals_jsons_dir = "config_files/eval_config_files"
top_templates_jsons_dir = "config_files/template_config_files"


# MAGIX        
generate_scan_jsons(
    top_jsons_dir=os.path.join(top_evals_jsons_dir, "magix_auto_generated_configs", "magix"),
    template_json_path=os.path.join(top_templates_jsons_dir, "magix_template_config_server.json")
)

# 1001
generate_scan_jsons(
    top_jsons_dir=os.path.join(top_evals_jsons_dir, "mm_whs_auto_generated_configs", "1001"),
    template_json_path=os.path.join(top_templates_jsons_dir, "mm_whs_1001_template_config_server.json")
)

# 1003
generate_scan_jsons(
    top_jsons_dir=os.path.join(top_evals_jsons_dir, "mm_whs_auto_generated_configs", "1003"),
    template_json_path=os.path.join(top_templates_jsons_dir, "mm_whs_1003_template_config_server.json")
)

# 1016
generate_scan_jsons(
    top_jsons_dir=os.path.join(top_evals_jsons_dir, "mm_whs_auto_generated_configs", "1016"),
    template_json_path=os.path.join(top_templates_jsons_dir, "mm_whs_1016_template_config_server.json")
)

# 1020
generate_scan_jsons(
    top_jsons_dir=os.path.join(top_evals_jsons_dir, "mm_whs_auto_generated_configs", "1020"),
    template_json_path=os.path.join(top_templates_jsons_dir, "mm_whs_1020_template_config_server.json")
)