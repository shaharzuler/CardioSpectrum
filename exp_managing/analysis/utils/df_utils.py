import numpy as np
import pandas as pd


def get_metric_for_nn_type(df, metric_name, nn_type, use_scale_down): 
    abs_epe_pixels = np.array(df.loc[(slice(None), nn_type, slice(None), slice(None), metric_name), "Value"], dtype=float)
    if use_scale_down:
        scale_down_factor = np.array(df.loc[(slice(None), nn_type, slice(None), slice(None), "scale_down_factor"), "Value"], dtype=float)
        abs_epe_mm = abs_epe_pixels * scale_down_factor
        epe = abs_epe_mm
    else:
        epe = abs_epe_pixels
    return epe

def get_metric_for_tot_torsion_and_nn_type(df, metric_name, tot_torsion, nn_type, use_scale_down_factor): 
    epe_pixels = np.array(df.loc[(slice(None), nn_type, tot_torsion, slice(None), metric_name), "Value"], dtype=float)
    actual_tot_torsion = np.array(df.loc[(slice(None), nn_type, tot_torsion, slice(None), "actual_tot_torsion"), "Value"], dtype=float)
    if use_scale_down_factor:
        scale_down_factor = np.array(df.loc[(slice(None), nn_type, tot_torsion, slice(None), "scale_down_factor"), "Value"], dtype=float)
        epe_mm = epe_pixels * scale_down_factor
        epe = epe_mm
    else:
        epe = epe_pixels
    return epe, actual_tot_torsion

def get_metric_for_tot_torsion_and_nn_type_and_sample(df, sample_name, metric_name, tot_torsion, nn_type, use_scale_down_factor): 
    epe_pixels = np.array(df.loc[(sample_name, nn_type, tot_torsion, slice(None), metric_name), "Value"], dtype=float)
    actual_tot_torsion = np.array(df.loc[(sample_name, nn_type, tot_torsion, slice(None), "actual_tot_torsion"), "Value"], dtype=float)
    if use_scale_down_factor:
        scale_down_factor = np.array(df.loc[(sample_name, nn_type, tot_torsion, slice(None), "scale_down_factor"), "Value"], dtype=float)
        epe_mm = epe_pixels * scale_down_factor
        epe = epe_mm
    else:
        epe = epe_pixels
    return epe, actual_tot_torsion

def get_metric_for_tot_torsion_and_nn_type_and_sample_and_torsion_version(df, sample_name, torsion_version, metric_name, tot_torsion, nn_type, use_scale_down_factor): 
    epe_pixels = np.array(df.loc[(sample_name, nn_type, tot_torsion, torsion_version, metric_name), "Value"], dtype=float)
    actual_tot_torsion = np.array(df.loc[(sample_name, nn_type, tot_torsion, torsion_version, "actual_tot_torsion"), "Value"], dtype=float)
    if use_scale_down_factor:
        scale_down_factor = np.array(df.loc[(sample_name, nn_type, tot_torsion, torsion_version, "scale_down_factor"), "Value"], dtype=float)
        epe_mm = epe_pixels * scale_down_factor
        epe = epe_mm
    else:
        epe = epe_pixels
    return epe, actual_tot_torsion

def set_metric_from_errors_dct_to_df(sample_name, nn_type, tot_torsion, torsion_version, metric:str, df:pd.DataFrame, errs_dct):
    df.loc[(sample_name, nn_type, tot_torsion, torsion_version, metric), "Value"] = errs_dct[nn_type][metric]
    return df