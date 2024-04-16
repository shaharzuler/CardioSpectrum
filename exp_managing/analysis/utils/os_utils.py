import os
from datetime import datetime
import json

import four_d_ct_cost_unrolling

from experiment_analyzer import ExperimentAnalyzer


def _extract_actual_tot_torsion(exp_dir):
    for folder in os.listdir(exp_dir):
        if "dataset" in folder:
            dataset_path =  os.path.join(exp_dir, folder)
            for subfolder in os.listdir(dataset_path):
                if "thetas" in subfolder:
                    actual_tot_torsion_file_path = os.path.join(dataset_path, subfolder, "total_torsion.txt")
                    with open(actual_tot_torsion_file_path) as f:
                        txt = f.read()
                    actual_tot_torsion = abs(float(txt.split("Total torsion: ")[-1]))
                    return actual_tot_torsion
    return None

def _get_all_subdirs_containing(exp_dir, nn_type):
    all_subsubdirectories = []
    for root, dirs, files in os.walk(exp_dir):
        for subdir in dirs:
            if "outputs" in subdir:
                subdirectory_path = os.path.join(root, subdir)
                for typedir in os.listdir(subdirectory_path):
                    if nn_type in typedir:
                        all_subsubdirectories.append(os.path.join(subdirectory_path, typedir))
    return all_subsubdirectories

def extract_date_time(all_subsubdirectories):
    date_str, time_str = all_subsubdirectories.split("/")[-1].split('_')[-2:]
    return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

def _get_most_recent_subsubdorectory(all_subsubdirectories):
    if len(all_subsubdirectories) == 0:
        return None
    elif len(all_subsubdirectories) == 1:
        return all_subsubdirectories[0]
    elif len(all_subsubdirectories)>1:
        return max(all_subsubdirectories, key=extract_date_time)

def _get_all_exps_paths(exp_dir):
    exp_names = os.listdir(exp_dir)
    return [os.path.join(exp_dir, exp_name) for exp_name in exp_names]

def get_torsion_data(exp_dir): 
    exp_parts = exp_dir.split("/")[-1].split("_")
    tot_torsion = exp_parts[2]
    torsion_version = exp_parts[-1]
    actual_tot_torsion = _extract_actual_tot_torsion(exp_dir)
    return int(tot_torsion), int(torsion_version), actual_tot_torsion

def get_most_recent_checkpoints(exp_dir, nn_type, minimal_date="20230101"):
    minimal_full_date = f"outputs_{minimal_date}_000000"
    all_subsubdirectories = _get_all_subdirs_containing(exp_dir, nn_type)
    while len(all_subsubdirectories)>0:
        exp_most_recent_dir = _get_most_recent_subsubdorectory(all_subsubdirectories)
        checkpoints_path = four_d_ct_cost_unrolling.get_checkpoints_path(exp_most_recent_dir)
        if os.path.isfile(checkpoints_path):
            exp_full_date = exp_most_recent_dir.split("/")[-2]
            if max([exp_full_date, minimal_full_date], key=extract_date_time) != minimal_full_date:
                return checkpoints_path
        all_subsubdirectories.remove(exp_most_recent_dir)
    return None

def get_most_recent_errors_path(exp_dir, minimal_date):
    minimal_full_date = f"outputs_{minimal_date}_000000"
    all_subsubdirectories = [d for d in os.listdir(exp_dir) if "outputs" in d]
    while len(all_subsubdirectories)>0:
        exp_most_recent_dir = _get_most_recent_subsubdorectory(all_subsubdirectories)
        errs_file_path = os.path.join(exp_dir, exp_most_recent_dir, "errors.json")
        if os.path.isfile(errs_file_path):
            exp_full_date = exp_most_recent_dir.split("/")[-1]
            if max([exp_full_date, minimal_full_date], key=extract_date_time) != minimal_full_date:
                return errs_file_path
        all_subsubdirectories.remove(exp_most_recent_dir)
    return None

def get_summary_path(exp_dir, nn_type, minimal_date="20230101"): 
    minimal_full_date = f"outputs_{minimal_date}_000000"
    all_subsubdirectories = _get_all_subdirs_containing(exp_dir, nn_type)
    while len(all_subsubdirectories) > 0:
        exp_most_recent_dir = _get_most_recent_subsubdorectory(all_subsubdirectories)
        if max([exp_most_recent_dir, minimal_full_date], key=extract_date_time) == minimal_full_date:
            pass
        else:
            summary_dir_path = os.path.join(exp_most_recent_dir, "filtered_summary")
            if os.path.isdir(summary_dir_path) and os.path.isfile(four_d_ct_cost_unrolling.get_checkpoints_path(exp_most_recent_dir)):
                summary_filename = os.listdir(summary_dir_path)[0]
                summary_path = os.path.join(summary_dir_path, summary_filename)
                return summary_path
        all_subsubdirectories.remove(exp_most_recent_dir)
    return None

def get_scale_down_factor(exp_dir, nn_type): 
    all_subsubdirectories = _get_all_subdirs_containing(exp_dir, nn_type)
    exp_most_recent_dir = _get_most_recent_subsubdorectory(all_subsubdirectories)
    if exp_most_recent_dir is None:
        return None
    else:
        config_path = os.path.join(exp_most_recent_dir, "training_config.json")
        with open(config_path) as f:
            config = json.load(f)
        scale_down_factor = config["scale_down_by"]
        return scale_down_factor

def get_sample_exp_dirs(sample_name): 
    with open("exp_managing/analysis/utils/exp_paths.json", "r") as f:
        return _get_all_exps_paths(json.load(f)[sample_name])

def get_epoch_num(minimal_date, df, sample_name, exp_dir, tot_torsion, torsion_version, nn_type, min_save_iter=10):
    summary_path = get_summary_path(exp_dir, nn_type, minimal_date=minimal_date) 
    if summary_path is not None:
        df.loc[(sample_name, nn_type, tot_torsion, torsion_version, "summary_path"), "Value"] = summary_path
        experiment_analyzer = ExperimentAnalyzer(tb_summary_path=summary_path, min_n_epochs=1000)
        if experiment_analyzer.is_valid:
            epoch_w_min_shell_epe = experiment_analyzer.get_minimal_epoch_and_metric_val(determing_metric="shell_volume_error", min_save_iter=10)
            return epoch_w_min_shell_epe, df
    return None, df


print(1)