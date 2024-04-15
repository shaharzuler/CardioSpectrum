plots_strs = {
    "shell_volume_error": "mEPE",
    "shell_error_globally_radial": "mEPE - Radial",
    "shell_error_globally_circumferential": "mEPE - Circumferential",
    "shell_error_globally_longitudinal": "mEPE - Longitudinal",
    "shell_angular_error": "Mean Angular Error",
    "surface_error_locally_radial": "mEPE - Locally-Radial",
    "surface_error_locally_tangential": "mEPE - Locally-Tangential",
    "x_axis": "Torsion [degrees]",
    "backbone_training": "3D ARFlow",
    "segmentation_training": "3D ARFlow + Anatomical Loss",
    "constraints_training": "CardioSpectrum (Ours)",
}

def _get_use_scale_down_factor(metric_name):
    if "angular_error" in metric_name:
        return False
    elif "rel" in metric_name:
        return False
    else:
        return True

def _get_y_ax_factor(metric_name):
    if "rel" in metric_name:
        return 100. # for percents
    else:
        return 1.

def _get_units_from_metric_name(metric_name):
    if "angular_error" in metric_name:
        return "degrees"
    elif "rel" in metric_name:
        return "%"
    else:
        return "mm"

def get_plot_properties(metric_name):
    use_scale_down_factor = _get_use_scale_down_factor(metric_name)
    y_ax_factor = _get_y_ax_factor(metric_name)
    y_ax_units = _get_units_from_metric_name(metric_name)
    return use_scale_down_factor, y_ax_factor, y_ax_units
