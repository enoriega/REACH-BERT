from dataclasses import dataclass
from typing import Optional, Callable, List, Any, Mapping

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import PreTrainedTokenizer
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

    def __init__(self, dataset: ReachDataset, tokenizer: PreTrainedTokenizer, batch_size: int = 1):
        """
        Creates instance of `ReachDataModule`
        :param dataset: to use for train/dev/test
        """
        super().__init__()
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._train = DataLoader(self._dataset.train_dataset(), collate_fn=self._collator, batch_size=batch_size)
        self._test = DataLoader(self._dataset.test_dataset(), collate_fn=self._collator, batch_size=batch_size)
        self._dev = DataLoader(self._dataset.dev_dataset(), collate_fn=self._collator, batch_size=batch_size)


    def _collator(self, instances: List[InputSequence]) -> ReachBertInput:
        """ Converts a batch of `InputSequence`s into a suitable format for pytorch """

        ds_index = self._dataset.index

        # Tokenize the input words
        tokenizer = self._tokenizer

        encoding = tokenizer([i.masked_words for i in instances], padding='max_length', max_length=512, is_split_into_words= True, return_tensors='pt')

        # Map the labels to subword unit space
        tags  = torch.full((len(instances), 512), ds_index.tag_codes['O'], dtype=torch.int)
        interactions = torch.zeros((len(instances), ds_index.num_interactions), dtype=torch.int)

        for ix, instance in enumerate(instances):
            for jx, t in enumerate(instance.tags):
                tags[ix, encoding.word_to_tokens(jx)] =  ds_index.tag_codes[t]

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