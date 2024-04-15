import os

import numpy as np
import cv2

from flow_n_corr_utils import disp_flow_as_arrows, xyz3_to_3xyz


def min_max_norm(img, min_, max_):
    return (img-min_)/(max_-min_)

def draw_img_w_gt(gt_path, template_img_path, unlabeled_img_path, template_mask_path, unlabeled_mask_path, output_path):
    gt = np.nan_to_num(np.load(gt_path))
    template_img = np.load(template_img_path)
    unlabeled_img = np.load(unlabeled_img_path)
    template_mask = np.load(template_mask_path)
    unlabeled_mask = np.load(unlabeled_mask_path)

    min_ = min(unlabeled_img.min(), unlabeled_img.min())
    max_ = max(unlabeled_img.max(), unlabeled_img.max())
    unlabeled_img = min_max_norm(unlabeled_img, min_, max_)
    template_img = min_max_norm(template_img, min_, max_)

    a=20
    b=-10
    c=-40
    gt = gt[a:b, a:b, :c, :]
    template_img = template_img[a:b, a:b, :c]
    unlabeled_img = unlabeled_img[a:b, a:b, :c]
    template_mask = template_mask[a:b, a:b, :c]
    unlabeled_mask = unlabeled_mask[a:b, a:b, :c]

    img_w_gt = disp_flow_as_arrows(img=np.zeros_like(unlabeled_img), seg=unlabeled_mask, flow=xyz3_to_3xyz(gt), arrow_color=(33/255, 33/255, 255/255), circle_color=(0.5, 0, 0.5), thickness=1, paper_vis_config=True)  #BGR  
    img_unlabeled = disp_flow_as_arrows(img=unlabeled_img, seg=unlabeled_mask, flow=np.zeros_like(xyz3_to_3xyz(gt)), arrow_color=(0.3, 0.3, 0.3), circle_color=(0.3, 0.3, 0.3), paper_vis_config=True)    
    img_template = disp_flow_as_arrows(img=template_img, seg=template_mask, flow=np.zeros_like(xyz3_to_3xyz(gt)), arrow_color=(0.3, 0.3, 0.3), circle_color=(0.3, 0.3, 0.3), paper_vis_config=True)    

    cv2.imwrite(os.path.join(output_path, "unlabeled_img.png"), np.clip((255*np.transpose(img_unlabeled[0], (1,2,0))).astype(int),0,255)) 
    cv2.imwrite(os.path.join(output_path, "template_img.png"), np.clip((255*np.transpose(img_template[0], (1,2,0))).astype(int),0,255)) 

    alpha_red = 1.0 #0.45 #
    ii = img_unlabeled.copy()
    ii[img_w_gt != 0] = (1-alpha_red)*ii[img_w_gt != 0] + alpha_red * img_w_gt[img_w_gt != 0]
    iii=np.transpose(ii[0], (1,2,0))
    cv2.imwrite(os.path.join(output_path, "w_gt.png"), np.clip((255*iii).astype(int),0,255)) 


if __name__ == "__main__":
    # gt_path = np.nan_to_num(np.load("/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/mm_whs/1014/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"))
    # template_img_path = np.load("/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/mm_whs/1014/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/01/orig/voxels/xyz_arr_raw.npy")#"/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/mm_whs/1014/tot_torsion_80_torsion_version_3/dataset_tot_torsion_80_torsion_version_3/thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_0.0_0.0_rs_1.0_1.0_h_1.0_linear_mask_True_blur_radious_1.npy")
    # unlabeled_img_path = np.load("/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/mm_whs/1014/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy")
    # template_mask_path = np.load("/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/mm_whs/1014/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy")
    # unlabeled_mask_path = np.load("/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/mm_whs/1014/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1//mask_skewed_thetas_80.0_0.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy")

    gt_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/slicer/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/flow_for_image_thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
    template_img_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/slicer/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/01/orig/voxels/xyz_arr_raw.npy" #"/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/mm_whs/1014/tot_torsion_80_torsion_version_3/dataset_tot_torsion_80_torsion_version_3/thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_orig_thetas_0.0_0.0_rs_1.0_1.0_h_1.0_linear_mask_True_blur_radious_1.npy")
    unlabeled_img_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/slicer/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/image_skewed_thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
    template_mask_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/slicer/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1/mask_orig_thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"
    unlabeled_mask_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/slicer/tot_torsion_80_torsion_version_4/dataset_tot_torsion_80_torsion_version_4/thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1//mask_skewed_thetas_60.0_-20.0_rs_0.9_0.9_h_0.91_linear_mask_True_blur_radious_1.npy"

    draw_img_w_gt(gt_path, template_img_path, unlabeled_img_path, template_mask_path, unlabeled_mask_path, output_path='/home/shahar/home/shahar/projects/complete_constrained_cardiac_temporal_correspondence_project/complete_constrained_cardiac_temporal_correspondence/complete_constrained_cardiac_temporal_correspondence/src/sample_scan/output/slicer')
    print(1)