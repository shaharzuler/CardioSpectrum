import json

import pandas as pd

from utils.os_utils import get_sample_exp_dirs, get_scale_down_factor, get_torsion_data, get_most_recent_errors_path, get_epoch_num
from utils.df_utils import set_metric_from_errors_dct_to_df
from plotter import Plotter

minimal_date = "20240112"

analysis_outputs_path = "sample_scan/analysis_outputs"

sample_names = ["magix", "mm_whs_1001", "mm_whs_1003", "mm_whs_1016", "mm_whs_1020"]
nn_types = ["backbone_training", "segmentation_training", "constraints_training"]
tot_torsions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
torsion_versions = [0, 1, 2, 3, 4]

metrics_to_read = ["shell_volume_error", "rel_shell_volume_error","relative_surface_error","surface_error",\
    "shell_error_globally_radial","shell_error_globally_longitudinal", "shell_error_globally_circumferential",\
        "shell_rel_error_globally_radial","shell_rel_error_globally_longitudinal","shell_rel_error_globally_circumferential",\
            "surface_error_locally_radial","rel_surface_error_locally_radial","surface_error_locally_tangential","rel_surface_error_locally_tangential",\
                "shell_angular_error", "surface_angular_error"]

data = ["epoch", "errors_path", "summary_path", "actual_tot_torsion", "scale_down_factor"] + metrics_to_read

errs_names = ["errors_backbone", "errors_w_seg", "errors_w_constraints"]

midx = pd.MultiIndex.from_product([sample_names, errs_names, tot_torsions, torsion_versions, data], names=["Sample", "NN_Type", "Tot_Torsion", "Torsion_Version", "Data"])
df = pd.DataFrame(index=midx, columns=["Value"])

do_get_epoch_num = True

redo_df = True

if redo_df:
    corrupted_exps = []
    for sample_name in sample_names:
        exp_dirs = get_sample_exp_dirs(sample_name)
        for exp_dir in exp_dirs: 
            tot_torsion, torsion_version, actual_tot_torsion = get_torsion_data(exp_dir)
            for nn_type, errs_name in zip(nn_types, errs_names):
                errors_path = get_most_recent_errors_path(exp_dir, minimal_date=minimal_date)
                scale_down_factor = get_scale_down_factor(exp_dir, nn_type) 
                if do_get_epoch_num:
                    epoch, df = get_epoch_num(minimal_date, df, sample_name, exp_dir, tot_torsion, torsion_version, nn_type, min_save_iter=10)
                    if epoch is not None:
                        df.loc[(sample_name, nn_type, tot_torsion, torsion_version, "epoch"), "Value"] = epoch
                    else:
                        print(exp_dir, nn_type, "epoch num unreachable")
                        corrupted_exps.append((exp_dir, nn_type))
                if errors_path is not None:
                    with open(errors_path) as j:
                        errors = json.load(j)
                    df.loc[(sample_name, errs_name, tot_torsion, torsion_version, "errors_path"       ), "Value"] = errors_path
                    df.loc[(sample_name, errs_name, tot_torsion, torsion_version, "actual_tot_torsion"), "Value"] = actual_tot_torsion
                    df.loc[(sample_name, errs_name, tot_torsion, torsion_version, "scale_down_factor" ), "Value"] = scale_down_factor

                    for metric in metrics_to_read:
                        df = set_metric_from_errors_dct_to_df(sample_name, errs_name, tot_torsion, torsion_version, metric, df, errors)

                else:
                    print(exp_dir, nn_type, "is corrupted, summary doesn't exist")
                    corrupted_exps.append((exp_dir, nn_type))

        print(corrupted_exps)
    df.to_pickle(f"{analysis_outputs_path}/general_df.pkl")
else:
    df = pd.read_pickle(f"{analysis_outputs_path}/general_df.pkl")
            
plotter = Plotter(df, tot_torsions, sample_names, nn_types, errs_names, analysis_outputs_path)

short=False
if short:
    metrics_to_read = [\
        "shell_volume_error", "shell_error_globally_radial","shell_error_globally_longitudinal", "shell_error_globally_circumferential",\
            "surface_error_locally_radial","surface_error_locally_tangential","shell_angular_error"\
                ]
       
    for metric in metrics_to_read :
        plotter.plot_nn_type_metric_vs_tot_torsion_general_mean_x(metric) 
else:
    for metric in metrics_to_read :
        plotter.plot_nn_type_metric_vs_tot_torsion_general(metric) 
        plotter.plot_nn_type_metric_vs_tot_torsion_general_mean_x(metric) 
        plotter.plot_nn_type_all_metric_vs_tot_torsion_general(metric)
        for sample_name in sample_names:
            plotter.plot_nn_type_metric_vs_tot_torsion_sample(sample_name, metric)
            plotter.plot_nn_type_all_metric_vs_tot_torsion_sample(sample_name, metric)
            for torsion_version in torsion_versions:
                plotter.plot_nn_type_all_metric_vs_tot_torsion_sample_torsion_version(sample_name, torsion_version, metric)


print("done")







