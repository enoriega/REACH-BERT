from typing import cast

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from data_loaders import ReachDataset

@CALLBACK_REGISTRY
class ShiftMasksCallback(Callback):
    """ Will update the data loader of the trainer to change the
        version of the pre-masked data (see RoBERTa for details) """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Fetch the dataloader from the trainer

        # Should be enough to do it on one data loader as the datamodule uses the same ref to the dataset for
        # all dataloaders
        dataset = cast(ReachDataset, trainer.train_dataloader.dataset.datasets.dataset) # This is just for type hint

        # Increment the mask version, but if fails as an out-of-bounds index, circle back to the first one
        try:
            dataset.masked_index += 1
        except IndexError:
            dataset.masked_index = 1