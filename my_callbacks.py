import time
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback


class TrainingStatsCallback(Callback):
    def __init__(self, stats_dict):
        self.stats = stats_dict

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            self.stats["peak_vram_bytes"] = max(
                self.stats.get("peak_vram_bytes", 0),
                torch.cuda.max_memory_allocated(),
            )

            self.stats["peak_reserved_bytes"] = max(
                self.stats.get("peak_reserved_bytes", 0),
                torch.cuda.max_memory_reserved(),
            )

    def on_fit_end(self, trainer, pl_module):
        total_time = time.perf_counter() - self.start_time

        self.stats["train_time_sec"] = total_time
        self.stats["train_time_min"] = total_time / 60

        if torch.cuda.is_available():
            self.stats["peak_vram_gb"] = (
                self.stats["peak_vram_bytes"] / 1024**3
            )

            self.stats["peak_reserved_gb"] = (
                self.stats["peak_reserved_bytes"] / 1024**3
            )