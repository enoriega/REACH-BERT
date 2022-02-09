from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from data_loaders import ReachDataset


class ReachDataModule(LightningDataModule):
    """ Encapsulates the different dataloaders using lightning """

    def __init__(self, dataset: ReachDataset):
        """
        Creates instance of `ReachDataModule`
        :param dataset: to use for train/dev/test
        """
        super().__init__()
        self._dataset = dataset
        self._train = DataLoader(self._dataset.train_dataset())
        self._test = DataLoader(self._dataset.test_dataset())
        self._dev = DataLoader(self._dataset.dev_dataset())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dev

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()