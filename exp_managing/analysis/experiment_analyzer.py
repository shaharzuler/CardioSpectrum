from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np



class ExperimentAnalyzer:
    def __init__(self, tb_summary_path, min_n_epochs=0) -> None:
        self.ea = event_accumulator.EventAccumulator(tb_summary_path, 
        size_guidance={ 
        event_accumulator.COMPRESSED_HISTOGRAMS: 500000,
        event_accumulator.IMAGES: 400000,
        event_accumulator.AUDIO: 400000,
        event_accumulator.SCALARS: 100000,
        event_accumulator.HISTOGRAMS: 10000
        }
        )  
        self.ea.Reload()  
        self.is_valid = True
        self.min_n_epochs = min_n_epochs
        if self.min_n_epochs > 0:
            if 'shell_volume_error' in self.ea.Tags()["scalars"]:
                self.num_actual_epochs = len(self.ea.Scalars('shell_volume_error')) 
                if self.num_actual_epochs < self.min_n_epochs:
                    self.is_valid = False
            else:
                self.is_valid = False

    def get_minimal_epoch(self, determing_metric, min_save_iter=10):
        determing_metric_vals = np.array(pd.DataFrame(self.ea.Scalars(determing_metric)))
        epoch_w_min_val = np.argmin(determing_metric_vals[min_save_iter:,2]) + min_save_iter
        return int(epoch_w_min_val)

    def get_metric_meas_by_epoch(self, metric_name, epoch):
        metric_vals = np.array(pd.DataFrame(self.ea.Scalars(metric_name)))
        metric_val = metric_vals[epoch, 2]
        print(f"{metric_vals[np.argmin(metric_vals[10:,2]) + 10, 2]} vs {metric_val}, epochs {np.argmin(metric_vals[10:,2]) + 10} vs {epoch}")
        print(f"diff of {100*((metric_val - metric_vals[np.argmin(metric_vals[10:,2]) + 10, 2])/(metric_vals[np.argmin(metric_vals[10:,2]) + 10, 2]))} %")
        return metric_val

    def get_metric_at_epoch(self, metric_name, epoch):
        metric = np.array(pd.DataFrame(self.ea.Scalars(metric_name)))[epoch, 2]
        return metric