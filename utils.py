import os
import json

import numpy as np
from easydict import EasyDict
import scipy

from three_d_data_manager import extract_segmentation_envelope
from flow_n_corr_utils import interpolate_from_flow_in_axis, attach_flow_between_segs, xyz3_to_3xyz, t3xyz_to_xyz3



def gt_to_fake_constraints(gt:np.ndarray, mask:np.ndarray, interp_knn:int=1, smooth_mask=None) -> np.ndarray: #TODO use smooth voxelized mask instead of the accurate mask
    print("generating fake constraints from ground truth")
    fake_constraints = np.zeros_like(gt)
    fake_constraints[:] = np.nan
    segmentation_envelope = extract_segmentation_envelope(mask)
    fake_constraints[segmentation_envelope] = gt[segmentation_envelope]

    if smooth_mask is not None:
        smooth_constraints = attach_flow_between_segs(sparse_flows_arr=fake_constraints, seg_arr=smooth_mask)
        fake_constraints = attach_flow_between_segs(sparse_flows_arr=smooth_constraints, seg_arr=mask)

    if interp_knn > 1:
        for axis in range(fake_constraints.shape[-1]):
            fake_constraints = interpolate_from_flow_in_axis(interp_knn, fake_constraints, axis)
    return fake_constraints


def remove_non_floats_from_dict(dct):
    return {k: v for k, v in dct.items() if isinstance(v, float)}


def json2dict(file_path:str) -> dict:
    with open(file_path) as file:
        dct = EasyDict(json.load(file))
    return dct

def handle_pred_flow(scale, infer_constraints_model_output_path, for_drawing_path):
    pred_flow = np.load(os.path.join(infer_constraints_model_output_path , "warped_seg_maps", "flow_18_to_28.npy"))[0]
    reorganized_pred_flow = xyz3_to_3xyz(scipy.ndimage.zoom(t3xyz_to_xyz3(pred_flow)*scale, (scale,scale,scale,1)))
    np.save(os.path.join(for_drawing_path, "pred_flow.npy"), reorganized_pred_flow)


