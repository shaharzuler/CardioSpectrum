import os

import numpy as np
import cv2

from flow_n_corr_utils import disp_flow_as_arrows, xyz3_to_3xyz

def min_max_norm(img, min_, max_):
    return (img-min_)/(max_-min_)

def _make_legend_bkg(r1,g1,b1,r2,g2,b2,a,b,c, min1,max1,min2,max2,min3,max3):
    legend = 255*np.ones((a,b,c), dtype=np.uint8)

    legend[min1:max1, -min2:-max2, 0] = r1 
    legend[min1:max1, -min2:-max2, 1] = g1
    legend[min1:max1, -min2:-max2, 2] = b1 

    legend[-min3:-max3, -min2:-max2, 0] = r2
    legend[-min3:-max3, -min2:-max2, 1] = g2 
    legend[-min3:-max3, -min2:-max2, 2] = b2  

    return legend

def draw_img_w_gt_and_preds(bb_pred_path, constraints_pred_path, gt_path, unlabeled_img_path, unlabeled_mask_path, output_root, crop=False):
    if bb_pred_path is not None:
        bb_pred = np.load(bb_pred_path)
        print(bb_pred.shape)
    constraints_pred = np.load(constraints_pred_path)
    gt = np.nan_to_num(np.load(gt_path))
    unlabeled_img = np.load(unlabeled_img_path)
    unlabeled_mask = np.load(unlabeled_mask_path)

    min_ = min(unlabeled_img.min(), unlabeled_img.min())
    max_ = max(unlabeled_img.max(), unlabeled_img.max())
    unlabeled_img = min_max_norm(unlabeled_img, min_, max_)

    if bb_pred_path is not None:
        img_w_bb_arrows = disp_flow_as_arrows(img=np.zeros_like(unlabeled_img), seg=unlabeled_mask, flow=bb_pred, arrow_color=(0.001, 192/255, 0.9999), circle_color=(0.5, 0.5, 0), thickness=1, paper_vis_config=True) #BGR
        if crop:
            img_w_bb_arrows = img_w_bb_arrows[:,:,40:-35,:]
    
    img_w_gt = disp_flow_as_arrows(img=np.zeros_like(unlabeled_img), seg=unlabeled_mask, flow=xyz3_to_3xyz(gt), arrow_color=(33/255, 33/255, 255/255), circle_color=(0.5, 0, 0.5), thickness=2, paper_vis_config=True) #BGR  
    img_w_constraints_arrows = disp_flow_as_arrows(img=np.zeros_like(unlabeled_img), seg=unlabeled_mask, flow=constraints_pred, arrow_color=(236/255, 231/255, 32/255), circle_color=(0., 0.99, 0.99), thickness=1, paper_vis_config=True) #BGR
    img = disp_flow_as_arrows(img=unlabeled_img, seg=unlabeled_mask, flow=np.zeros_like(constraints_pred), arrow_color=(0., 0., 0.), circle_color=(0., 0., 0.), paper_vis_config=True)   

    if crop:
        img = img[:,:,40:-35,:]
        img_w_gt = img_w_gt[:,:,40:-35,:]
        img_w_constraints_arrows = img_w_constraints_arrows[:,:,40:-35,:]

    a, b, c, l_a, l_b, l_c, l_d, l_e, l_f = 42, 155, 3, 11, 13, 10, 6, 40, 5

    legend_pred = _make_legend_bkg(236, 231, 32, 33, 33, 255, a, b, c, l_a, l_b, l_e, l_f, l_c, l_d)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend_pred, 'CardioSpectrum', org=(5, 13), fontFace=font, fontScale=0.35, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(legend_pred, 'Ground Truth', org=(5, 33), fontFace=font, fontScale=0.35, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    alpha_red = 1.0 #0.45 #
    ii = img.copy()
    ii[img_w_gt != 0] = (1-alpha_red)*ii[img_w_gt != 0] + alpha_red * img_w_gt[img_w_gt != 0]
    ii[img_w_constraints_arrows != 0] = img_w_constraints_arrows[img_w_constraints_arrows != 0]
    iii=np.transpose(ii[0], (1,2,0))
    iii[:legend_pred.shape[0], -legend_pred.shape[1]:, :] = legend_pred/255 #top right

    cv2.imwrite(os.path.join(output_root, "w_constraints_pred_vs_gt.png"), np.clip((255*iii).astype(int),0,255)) 

    if bb_pred_path is not None:
        legend_baseline = _make_legend_bkg(0, 192,255, 33,33,255, a, b, c, l_a, l_b, l_e, l_f, l_c, l_d)
        cv2.putText(legend_baseline, 'Baseline', org=(5, 13), fontFace=font, fontScale=0.35, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(legend_baseline, 'Ground Truth', org=(5, 33), fontFace=font, fontScale=0.35, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        ii[img_w_bb_arrows != 0] = img_w_bb_arrows[img_w_bb_arrows != 0]
        iii=np.transpose(ii[0], (1,2,0))
        cv2.imwrite(os.path.join(output_root, "all.png"), np.clip((255*iii).astype(int),0,255)) 

        ii = img.copy()
        ii[img_w_gt != 0] = (1-alpha_red)*ii[img_w_gt != 0] + alpha_red*img_w_gt[img_w_gt != 0]
        ii[img_w_bb_arrows != 0] = img_w_bb_arrows[img_w_bb_arrows != 0]
        iii=np.transpose(ii[0], (1,2,0))

        iii[:legend_baseline.shape[0], -legend_baseline.shape[1]:, :] = legend_baseline/255
        cv2.imwrite(os.path.join(output_root, "baseline_vs_gt.png"), np.clip((255*iii).astype(int),0,255))

if __name__ == "__main__":
    bb_pred_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_110_torsion_version_0/outputs_20240129_110323/for_drawing/bb_pred.npy"
    constraints_pred_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_110_torsion_version_0/outputs_20240129_110323/for_drawing/constraints_pred.npy"
    gt_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_110_torsion_version_0/outputs_20240129_110323/for_drawing/gt.npy"
    unlabeled_img_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_110_torsion_version_0/outputs_20240129_110323/for_drawing/unlabeled_img.npy"
    unlabeled_mask_path = "/mnt/storage/datasets/shahar_user_data/complete_cnstrained_outputs/miccai_experiments/magix/tot_torsion_110_torsion_version_0/outputs_20240129_110323/for_drawing/unlabeled_mask.npy"

    draw_img_w_gt_and_preds(bb_pred_path, constraints_pred_path, gt_path, unlabeled_img_path, unlabeled_mask_path, output_root=".")
    print("done")