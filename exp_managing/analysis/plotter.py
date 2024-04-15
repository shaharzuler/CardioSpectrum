import numpy as np
from matplotlib import pyplot as plt

from utils.plot_properties import get_plot_properties, plots_strs
from utils.df_utils import get_metric_for_tot_torsion_and_nn_type_and_sample_and_torsion_version, get_metric_for_tot_torsion_and_nn_type_and_sample, get_metric_for_tot_torsion_and_nn_type
from utils.stat_utils import nansem, nanstd

class Plotter:
    def __init__(self, df, tot_torsions, sample_names, nn_types, errs_names, analysis_outputs_path) -> None:
        self.df = df
        self.tot_torsions = tot_torsions
        self.sample_names = sample_names
        self.nn_types = nn_types
        self.errs_names = errs_names
        self.analysis_outputs_path = analysis_outputs_path

    def plot_nn_type_metric_vs_tot_torsion_general_mean_x(self, metric_name): 
        fontsize=18 ##16
        legend_fontsize=12
        use_scale_down_factor, y_ax_factor, y_ax_units = get_plot_properties(metric_name)
        plt.close()
        for_paper_table = []
        for nn_type, errs_name, color, fmt in zip(self.nn_types, self.errs_names, ('tab:blue','tab:orange','tab:red'), ('o','o','o')):
            for_paper_table.append(str((nn_type, errs_name)))
            actual_tot_torsions_means = []
            epes_means = []
            epes_sems = []
            epes_stds = []
            for tot_torsion in self.tot_torsions:
                epes = []
                actual_tot_torsions = []
                for sample_name in self.sample_names:
                    specific_tot_torsion_epes, actual_tot_torsion = get_metric_for_tot_torsion_and_nn_type_and_sample(self.df, sample_name, metric_name, tot_torsion, errs_name, use_scale_down_factor)
                    specific_tot_torsion_epes *= y_ax_factor
                    actual_tot_torsions = np.append(actual_tot_torsions, actual_tot_torsion)
                    epes = np.append(epes, specific_tot_torsion_epes)
                actual_tot_torsions_means.append(np.nanmean(actual_tot_torsions))
                epes_means.append(np.nanmean(epes))
                epes_sems.append(nansem(epes))
                epes_stds.append(nanstd(epes))
            plt.plot(actual_tot_torsions_means, epes_means, color=color)
            for_paper_table.append(str(("actual_tot_torsions_means",actual_tot_torsions_means, "epes_means",epes_means, "epes_stds",epes_stds)))
            plt.errorbar(actual_tot_torsions_means, epes_means, label=f"{plots_strs[nn_type]}",  yerr=epes_sems, fmt=fmt, ms=6, capsize=6, color=color)

        plt.tick_params(axis='both', which='major', labelsize=fontsize)  
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(plots_strs["x_axis"], fontsize=fontsize) 
        if metric_name in plots_strs.keys():
            plt.ylabel(f"{plots_strs[metric_name]} [{y_ax_units}]", fontsize=fontsize)   
        else:
            plt.ylabel(f"{metric_name} [{y_ax_units}]", fontsize=fontsize)   
        plt.show()
        plt.savefig(f"{self.analysis_outputs_path}/png/general_{metric_name}_actual_tot_torsion_3.png", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/pdf/general_{metric_name}_actual_tot_torsion_3.pdf", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/eps/general_{metric_name}_actual_tot_torsion_3.eps", bbox_inches='tight', format="eps")
        with open(f"{self.analysis_outputs_path}/eps/general_{metric_name}_actual_tot_torsion_3.txt", 'w') as fp:
            for item in for_paper_table:
                fp.write("%s\n" % item)

    def plot_nn_type_metric_vs_tot_torsion_general(self, metric_name): 
        use_scale_down_factor, y_ax_factor, y_ax_units = get_plot_properties(metric_name)

        plt.close()
        for nn_type, errs_name, color, fmt in zip(self.nn_types, self.errs_names, ('tab:blue','tab:orange','tab:red'), ('^','s','o')):
            for tot_torsion in self.tot_torsions:
                epes = []
                actual_tot_torsions = []
                sems = []
                for sample_name in self.sample_names:
                    specific_tot_torsion_epes, actual_tot_torsion = get_metric_for_tot_torsion_and_nn_type_and_sample(self.df, sample_name, metric_name, tot_torsion, errs_name, use_scale_down_factor)
                    actual_tot_torsions.append(np.nanmean(actual_tot_torsion))
                    specific_tot_torsion_epes *= y_ax_factor
                    epes.append(np.nanmean(specific_tot_torsion_epes))
                    sems.append(nansem(specific_tot_torsion_epes))

                plt.errorbar(actual_tot_torsions, epes, label=f"{nn_type} with standard error" if tot_torsion<1 else None,  yerr=sems, fmt=fmt, ms=6, capsize=4, color=color)

        plt.legend()
        plt.xlabel(f'total_torsion [degrees]') 
        plt.ylabel(f"{metric_name} [{y_ax_units}]")   
        plt.title(f'{metric_name} vs total (actual) torsion')
        plt.show()
        plt.savefig(f"{self.analysis_outputs_path}/png/general_{metric_name}_actual_tot_torsion_1.png", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/pdf/general_{metric_name}_actual_tot_torsion_1.pdf", bbox_inches='tight')

    def plot_nn_type_metric_vs_tot_torsion_sample(self, sample_name, metric_name): 
        use_scale_down_factor, y_ax_factor, y_ax_units = get_plot_properties(metric_name)

        plt.close()
        for nn_type, errs_name, color, fmt in zip(self.nn_types, self.errs_names, ('tab:blue','tab:orange','tab:red'), ('^','s','o')):
            epes = []
            actual_tot_torsions = []
            sems = []

            for tot_torsion in self.tot_torsions:
                specific_tot_torsion_epes, actual_tot_torsion = get_metric_for_tot_torsion_and_nn_type_and_sample(self.df, sample_name, metric_name, tot_torsion, errs_name, use_scale_down_factor)
                actual_tot_torsions.append(np.nanmean(actual_tot_torsion))
                specific_tot_torsion_epes *= y_ax_factor
                epes.append(np.nanmean(specific_tot_torsion_epes))
                sems.append(nansem(specific_tot_torsion_epes))

            plt.plot(actual_tot_torsions, epes, color=color) 
            plt.errorbar(actual_tot_torsions, epes, label=nn_type,  yerr=sems, fmt=fmt, ms=6, capsize=4, color=color)
        plt.legend()
        plt.xlabel(f'total_torsion [degrees]') 
        plt.ylabel(f"{metric_name} [{y_ax_units}]")   
        plt.title(f'{metric_name} vs total (actual) torsion {sample_name}')
        plt.show()
        plt.savefig(f"{self.analysis_outputs_path}/png/{sample_name}_{metric_name}_actual_tot_torsion_1.png", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/pdf/{sample_name}_{metric_name}_actual_tot_torsion_1.pdf", bbox_inches='tight')

    def plot_nn_type_metric_vs_tot_torsion_sample_torsion_version(self, sample_name, torsion_version, metric_name): 
        use_scale_down_factor, y_ax_factor, y_ax_units = get_plot_properties(metric_name)

        plt.close()
        for nn_type, errs_name, color, fmt in zip(self.nn_types, self.errs_names, ('tab:blue','tab:orange','tab:red'), ('^','s','o')):
            epes = []
            actual_tot_torsions = []
            sems = []

            for tot_torsion in self.tot_torsions:
                specific_tot_torsion_epes, actual_tot_torsion = get_metric_for_tot_torsion_and_nn_type_and_sample_and_torsion_version(self.df, sample_name, torsion_version, metric_name, tot_torsion, errs_name, use_scale_down_factor)
                actual_tot_torsions.append(np.nanmean(actual_tot_torsion))
                specific_tot_torsion_epes *= y_ax_factor
                epes.append(np.nanmean(specific_tot_torsion_epes))
                sems.append(nansem(specific_tot_torsion_epes))

            plt.plot(actual_tot_torsions, epes, color=color) 
            plt.errorbar(actual_tot_torsions, epes, label=nn_type,  yerr=sems, fmt=fmt, ms=6, capsize=4, color=color)
        plt.legend()
        plt.xlabel(f'total_torsion [degrees]') 
        plt.ylabel(f"{metric_name} [{y_ax_units}]")
        plt.title(f'{metric_name} vs total (actual) torsion {sample_name} torsion version {torsion_version}')
        plt.show()
        plt.savefig(f"{self.analysis_outputs_path}/png/{sample_name}_{torsion_version}_{metric_name}_actual_tot_torsion_1.png", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/pdf/{sample_name}_{torsion_version}_{metric_name}_actual_tot_torsion_1.pdf", bbox_inches='tight')

    def plot_nn_type_all_metric_vs_tot_torsion_general(self, metric_name): 
        use_scale_down_factor, y_ax_factor, y_ax_units = get_plot_properties(metric_name)

        plt.close()
        for nn_type, errs_name, color, fmt in zip(self.nn_types, self.errs_names, ('tab:blue','tab:orange','tab:red'), ('^','s','o')):
            for tot_torsion in self.tot_torsions:
                specific_tot_torsion_epes, actual_tot_torsions = get_metric_for_tot_torsion_and_nn_type(self.df, metric_name, tot_torsion, errs_name, use_scale_down_factor)
                specific_tot_torsion_epes *= y_ax_factor
                plt.scatter(actual_tot_torsions, specific_tot_torsion_epes, label=nn_type if tot_torsion==0 else None, marker=fmt, s=36, color=color)
        plt.legend()
        plt.xlabel(f'total_torsion [degrees]') 
        plt.ylabel(f"{metric_name} [{y_ax_units}]")    
        plt.title(f'{metric_name} vs total (actual) torsion')
        plt.show()
        plt.savefig(f"{self.analysis_outputs_path}/png/general_{metric_name}_actual_tot_torsion_2.png", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/pdf/general_{metric_name}_actual_tot_torsion_2.pdf", bbox_inches='tight')

    def plot_nn_type_all_metric_vs_tot_torsion_sample(self, sample_name, metric_name): 
        use_scale_down_factor, y_ax_factor, y_ax_units = get_plot_properties(metric_name)

        plt.close()
        for nn_type, errs_name, color, fmt in zip(self.nn_types, self.errs_names, ('tab:blue','tab:orange','tab:red'), ('^','s','o')):
            for tot_torsion in self.tot_torsions:
                specific_tot_torsion_epes, actual_tot_torsions = get_metric_for_tot_torsion_and_nn_type_and_sample(self.df, sample_name, metric_name, tot_torsion, errs_name, use_scale_down_factor)
                specific_tot_torsion_epes *= y_ax_factor
                for n, (actual_tot_torsion, specific_tot_torsion_epe) in enumerate(zip(actual_tot_torsions, specific_tot_torsion_epes)):
                    if ~np.isnan(actual_tot_torsion) and ~np.isnan(specific_tot_torsion_epe):
                        plt.scatter(actual_tot_torsion, specific_tot_torsion_epe, label=nn_type if (n==0 and tot_torsion==0) else None, marker=fmt, s=36, color=color)
                        plt.text(actual_tot_torsion+.14, specific_tot_torsion_epe-0.01, str(n), fontsize=9, color=color)
        plt.legend()
        plt.xlabel(f'total_torsion [degrees]') 
        plt.ylabel(f"{metric_name} [{y_ax_units}]")   
        plt.title(f'{metric_name} vs total (actual) torsion {sample_name}')
        plt.show()
        plt.savefig(f"{self.analysis_outputs_path}/png/{sample_name}_{metric_name}_actual_tot_torsion_2.png", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/pdf/{sample_name}_{metric_name}_actual_tot_torsion_2.pdf", bbox_inches='tight')

    def plot_nn_type_all_metric_vs_tot_torsion_sample_torsion_version(self, sample_name, torsion_version, metric_name): 
        use_scale_down_factor, y_ax_factor, y_ax_units = get_plot_properties(metric_name)

        plt.close()
        for nn_type, errs_name, color, fmt in zip(self.nn_types, self.errs_names, ('tab:blue','tab:orange','tab:red'), ('^','s','o')):
            for tot_torsion in self.tot_torsions:
                specific_tot_torsion_epes, actual_tot_torsions = get_metric_for_tot_torsion_and_nn_type_and_sample_and_torsion_version(self.df, sample_name, torsion_version, metric_name, tot_torsion, errs_name, use_scale_down_factor)
                specific_tot_torsion_epes *= y_ax_factor
                if ~np.isnan(actual_tot_torsions) and ~np.isnan(specific_tot_torsion_epes):
                    plt.scatter(actual_tot_torsions.item(), specific_tot_torsion_epes.item(), label=nn_type if tot_torsion==10 else None, marker=fmt, s=36, color=color) #the 10 is on porpose!
                    plt.text(actual_tot_torsions.item()+.14, specific_tot_torsion_epes.item() - 0.01, str(torsion_version), fontsize=9, color=color)
        plt.legend()
        plt.xlabel(f'total_torsion [degrees]') 
        plt.ylabel(f"{metric_name} [{y_ax_units}]")   
        plt.title(f'{metric_name} vs total (actual) torsion {sample_name}')
        plt.show()
        plt.savefig(f"{self.analysis_outputs_path}/png/{sample_name}_{torsion_version}_{metric_name}_actual_tot_torsion_2.png", bbox_inches='tight')
        plt.savefig(f"{self.analysis_outputs_path}/pdf/{sample_name}_{torsion_version}_{metric_name}_actual_tot_torsion_2.pdf", bbox_inches='tight')

