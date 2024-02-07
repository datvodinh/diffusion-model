from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os


class ModelCallback:
    def __init__(
        self,
        root_path: str,
        ckpt_monitor: str = "loss",
        ckpt_mode: str = "min",
        es_monitor: str = "loss",
        es_mode: str = "min"
    ):
        ckpt_path = os.path.join(os.path.join(root_path, "model/"))
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        self.ckpt_callback = ModelCheckpoint(
            monitor=ckpt_monitor,
            dirpath=ckpt_path,
            filename="model",
            save_top_k=1,
            mode=ckpt_mode
        )

        self.lr_callback = LearningRateMonitor("step")

        self.early_stop_callback = EarlyStopping(
            monitor=es_monitor,
            patience=15,
            verbose=True,
            mode=es_mode
        )

    def get_callback(self):
        return [
            self.ckpt_callback, self.lr_callback, self.early_stop_callback
        ]