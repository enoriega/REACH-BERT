from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from data_loaders import DatasetIndex
from data_loaders.utils import split_dataset
from data_utils import parse_input_file, InputSequence


class ReachDataset(Dataset):
    """ Dataset implementation over a directory with training data and optionally a directory with masked data """

    def __init__(self, data_dir:str,
                 masked_data_dir:Optional[str] = None,
                 overwrite_index:bool = False,
                 ) -> None:
        """
        Builds an instance from the parameters. If there is no index, it created one by default, otherwise it loads it
        :param data_dir: with original,unmasked files
        :param masked_data_dir: with masked variants of the files in `data_dir`
        """

        # Creates or loads an index on data_dir
        self.index = self.create_or_load_index(Path(data_dir), Path(masked_data_dir) if masked_data_dir else None, overwrite_index)

        # Sets the current masked index to 1 if available, else to zero to return the original version
        self.__masked_index = 1 if self.index.num_masked_instances > 0 else 0

    def __len__(self) -> int:
        """ Number of data in this dataset instance """
        return len(self.index.file_map)

    def __getitem__(self, item: int) -> InputSequence:
        """ Return Input Sequence by index """

        # Resolve the location of the item
        file, local_ix =  self.index.file_map[item]

        # Fetch original InputSeq
        items  = parse_input_file(self.index.data_dir / f"{file}.txt")

        orig = items[local_ix]

        # If masked index selected, fetch that too
        if self.masked_index > 0:
            masked_items = parse_input_file(self.index.masked_data_dir / f"{str(file)}_{self.masked_index}.txt")
            masked  = masked_items[local_ix]
        # Otherwise, consider orig == masked
        else:
            masked = orig

        # Merge both sequences together
        ret = InputSequence(
            event_labels= orig.event_labels,
            tags= orig.tags,
            words= orig.words,
            masked_words= masked.words
        )

        return ret

    # Overriding this to be able to use lru cache with the __getitem__ method efficiently.
    # Don't care about this instance's hash
    # def __hash__(self):
    #     return 0

    @property
    def num_masked_instances(self) -> int:
        """ Number of masked instances available in this dataset """
        return self.index.num_masked_instances

    @property
    def masked_index(self) -> int:
        """ Currently selected mask index """
        return self.__masked_index

    @masked_index.setter
    def masked_index(self, val) -> None:
        """ Set the mask index"""
        assert 0 <= val <= self.masked_index, f"Masked index out of bounds: {val} not within [0, {self.num_masked_instances}]"

        self.__masked_index = val

    @staticmethod
    def create_index(data_dir:  Path, masked_data_dir: Optional[Path]) -> DatasetIndex:
        """
        Builds an index of the data set for random access retrieval
        """

        seen = set()
        labels = list() # Keep track of the labels for stratification
        file_map = dict()
        index = 0
        for file_path in tqdm(data_dir.iterdir(), desc= "Creating dataset index", unit=" files"):
            file_name = str(file_path.stem)
            data = parse_input_file(file_path)
            for local_ix, datum in enumerate(data):
                # Filter duplicate phrases
                d_hash = hash(" ".join(datum.words))
                if d_hash not in seen:
                    file_map[index] = (file_name, local_ix)
                    index += 1
                    labels.append(datum.event_labels)
                    seen.add(d_hash)

        assert len(file_map) > 0, f"Empty data directory:{str(data_dir)}"

        if masked_data_dir:
            # Compute the number of masked instances by inspecting the copies of the first file
            file_name = file_map[0][0]
            num_masked_instances = len(list(masked_data_dir.glob(f"{file_name}_*.txt")))

        else:
            num_masked_instances = 0

        # Split the data set
        train, dev, test  = split_dataset(np.arange(index), labels, num_test=100_000, num_dev=100_00)

        index = DatasetIndex(
            data_dir= data_dir,
            file_map= file_map,
            masked_data_dir= masked_data_dir,
            num_masked_instances= num_masked_instances,
            train_indices= train,
            dev_indices= dev,
            test_indices= test
        )

        return index

    def create_or_load_index(self, data_dir: Path, masked_data_dir: Optional[Path], overwrite_index: bool = False) -> DatasetIndex:

        # If the index file is present, read it
        index_path = data_dir / "index.json"
        if not overwrite_index and index_path.exists():
            index = DatasetIndex.from_json_file(index_path)

        # Otherwise, create it and return it
        else:
            index = self.create_index(data_dir, masked_data_dir)
            with index_path.open('w') as f:
                f.write(index.to_json())

        return index

    def train_dataset(self) ->  Dataset:
        """ Generates training dataset view from current dataset """
        return Subset(self, self.index.train_indices)

    def test_dataset(self) -> Dataset:
        """ Generates testing dataset view from current dataset """
        return Subset(self, self.index.test_indices)

    def dev_dataset(self) -> Dataset:
        """ Generates development dataset view from current dataset """
        return Subset(self, self.index.dev_indices)


# Test case
if __name__ == "__main__":
    ds = ReachDataset("/media/evo870/data/reach-bert-data/bert_files",
                      "/media/evo870/data/reach-bert-data/masked_data",
                      False)

    print(len(ds.train_dataset()))

    # from numpy.random import default_rng
    # rng = default_rng(1024)
    # indices = rng.choice(len(ds), (1_000_000,))
    #
    # for ix in tqdm(indices):
    #     ds[ix]