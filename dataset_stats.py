""" Reads the input files and prints statistics about it """

import itertools as it
from collections import Counter
from pathlib import Path
from pprint import pprint

import plac
from tqdm import tqdm
import pandas as pd

from mask_data_files import parse_input_file


@plac.pos("input_dir_paths", help="Directory or directories with input files")
def main(*input_dir_paths) -> None:
    """
    Parses all the input files in the specified directories and prints useful stats
    """

    input_dir_paths = (Path(d) for d in input_dir_paths)
    input_files_paths = it.chain.from_iterable(d.iterdir() for d in input_dir_paths)

    # Keep counts to generate stats
    doc_level = Counter()
    token_level = Counter()
    tag_level =  Counter()
    label_level = Counter()
    total_seqs = 0

    # Count the data
    for input_path in tqdm(input_files_paths, desc="Parsing input files", unit=" files"):
        sequences = parse_input_file(input_path)
        total_seqs += len(sequences)
        doc_level[len(sequences)] += 1
        tag_level.update(it.chain.from_iterable(s.tags for s in sequences))
        token_level.update(it.chain.from_iterable(s.words for s in sequences))
        label_level.update(it.chain.from_iterable(s.event_labels for s in sequences))

    label_level = pd.Series(label_level).sort_values(ascending=False)
    tag_level = pd.Series(tag_level).sort_values(ascending=False)
    token_level = pd.Series(token_level).sort_values(ascending=False)
    doc_level = pd.Series(doc_level).sort_values(ascending=False)

    # Print stats
    print(f"Total documents: {doc_level[doc_level.index > 0].sum()}")
    print(f"Total input sequences: {total_seqs}")
    print(f"Total tokens: {token_level.sum()}")
    print(f"Total events: {label_level.sum()}")
    print("Tag distribution:")
    pprint(tag_level / tag_level.sum())
    print("Event distribution:")
    pprint(label_level / label_level.sum())



if __name__ == "__main__":
    plac.call(main)