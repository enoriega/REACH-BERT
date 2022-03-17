from dataclasses import dataclass
from typing import Optional, Callable, List, Any, Mapping

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import DataLoader

from data_loaders import ReachDataset
from data_utils import InputSequence


@dataclass
class ReachBertInput:
    features: Mapping[str, torch.FloatTensor]
    tags: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]

class ReachDataModule(LightningDataModule):
    """ Encapsulates the different dataloaders using lightning """

    def __init__(self,
                 data_dir: str,
                 masked_data_dir: str,
                 tokenizer_model_name: str,
                 num_workers: int = 1,
                 batch_size: int = 1,
                 max_seq_len: int = 256,
                 overwrite_dataset_index: bool = False,
                 debug: bool = False
                 ):
        """
        Creates instance of `ReachDataModule`
        :param dataset: to use for train/dev/test
        """
        super().__init__()
        # Load the data module and query the parameters
        self._dataset = ReachDataset(data_dir=data_dir,
                               masked_data_dir=masked_data_dir,
                               overwrite_index=overwrite_dataset_index)

        # Load the tokenizer model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._num_workers = num_workers
        self._train = DataLoader(self._dataset.train_dataset(), collate_fn=self._collator, batch_size=batch_size, num_workers= num_workers)
        self._test = DataLoader(self._dataset.test_dataset(), collate_fn=self._collator, batch_size=batch_size)
        self._dev = DataLoader(self._dataset.dev_dataset(), collate_fn=self._collator, batch_size=batch_size, num_workers = num_workers)

    @property
    def num_interactions(self):
        return self._dataset.num_interactions

    @property
    def num_tags(self):
        return self._dataset.num_tags

    def _collator(self, instances: List[InputSequence]) -> ReachBertInput:
        """ Converts a batch of `InputSequence`s into a suitable format for pytorch """

        ds_index = self._dataset.index
        max_seq_len = self._max_seq_len

        # Tokenize the input words
        tokenizer = self._tokenizer

        encoding = tokenizer([i.masked_words for i in instances], padding='max_length', max_length=max_seq_len, is_split_into_words= True, return_tensors='pt', truncation=True)

        # Map the labels to subword unit space
        tags  = torch.zeros((len(instances), max_seq_len, ds_index.num_tags), dtype=torch.float)
        interactions = torch.zeros((len(instances), ds_index.num_interactions), dtype=torch.float)

        for ix, instance in enumerate(instances):
            for jx, t in enumerate(instance.tags):
                tags[ix, encoding.word_to_tokens(jx), ds_index.tag_codes[t]] = 1

            for l in instance.event_labels:
                interactions[ix, ds_index.interaction_codes[l]] = 1


        return ReachBertInput(features=encoding.data, tags=tags,  labels=interactions)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dev

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()