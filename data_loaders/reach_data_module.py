import functools
from dataclasses import dataclass
from typing import Optional, Mapping, List

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from torch.utils.data import DataLoader

from data_loaders import ReachDataset, DatasetIndex
from data_loaders.utils import collapse_labels
from data_utils import InputSequence


@dataclass
class ReachBertInput:
    encoding: BatchEncoding
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

        # TODO maybe make this configurable??
        interaction_labels_hook = collapse_labels

        # Load the data module and query the parameters
        self._dataset = ReachDataset(data_dir=data_dir,
                               masked_data_dir=masked_data_dir,
                               overwrite_index=overwrite_dataset_index,
                               data_hook= interaction_labels_hook)

        # Load the tokenizer model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._num_workers = num_workers

        collate_fn = functools.partial(collator,
                                       max_seq_len=self._max_seq_len,
                                       ds_index=self._dataset.index,
                                       tokenizer=self._tokenizer)

        train_collate_fn = functools.partial(collate_fn, train=True)
        eval_collate_fn = functools.partial(collate_fn, train=False)

        self._train = DataLoader(self._dataset.train_dataset(), collate_fn=train_collate_fn, batch_size=batch_size, num_workers= num_workers)
        self._test = DataLoader(self._dataset.test_dataset(), collate_fn=eval_collate_fn, batch_size=batch_size, num_workers= num_workers)
        self._dev = DataLoader(self._dataset.dev_dataset(), collate_fn=eval_collate_fn, batch_size=batch_size, num_workers = num_workers)

    @property
    def num_interactions(self):
        return self._dataset.num_interactions

    @property
    def interaction_weights(self):
        return self._dataset.interaction_weights

    @property
    def tag_weights(self):
        return self._dataset.tag_weights

    @property
    def num_tags(self):
        return self._dataset.num_tags

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._test

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dev

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()


def collator(instances: List[InputSequence],
             max_seq_len: int,
             ds_index: DatasetIndex,
             tokenizer: PreTrainedTokenizer,
             train:bool = False
             ) -> ReachBertInput:
    """ Converts a batch of `InputSequence`s into a suitable format for pytorch """

    encoding = tokenizer([i.masked_words if train else i.words for i in instances], padding='max_length', max_length=max_seq_len, is_split_into_words= True, return_tensors='pt', truncation=True)

    # Map the labels to subword unit space
    tags  = torch.zeros((len(instances), max_seq_len, ds_index.num_tags), dtype=torch.float)
    interactions = torch.zeros((len(instances), ds_index.num_interactions), dtype=torch.float)

    for ix, instance in enumerate(instances):
        for jx, t in enumerate(instance.tags):
            tags[ix, encoding.word_to_tokens(jx), ds_index.tag_codes[t]] = 1

        for l in instance.event_labels:
            interactions[ix, ds_index.interaction_codes[l]] = 1


    return ReachBertInput(encoding=encoding, tags=tags,  labels=interactions)