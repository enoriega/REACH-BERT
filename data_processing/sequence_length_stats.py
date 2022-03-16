""" Tokenizes all the input sequences in the dataset and keeps track of the lenghts to help tune the batch size """
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import parse_input_file
import itertools as it
from collections import Counter

def main(inpuit_data_dir: Path, tokenizer_model_name: str) -> None:

    # Keep track of the length of the sequences here
    lengths = list()

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    for file_path in tqdm(inpuit_data_dir.iterdir(), desc="Parsing data ...", unit=" files"):
        # Read the file
        instances = parse_input_file(file_path)
        if len(instances) > 0:
            # Tokenize the words
            encoding = tokenizer([i.words for i in instances],
                                 is_split_into_words=True, truncation=False)
            # Keep track of the lenghts
            for seq in encoding.data['input_ids']:
                lengths.append(len(seq))

    # Print the quantiles
    lengths = pd.Series(lengths, name='Sequence Lengths')

    print("Quantile\tSequence Length")
    for q in [.5, .75, .8, .9, .95,  .99]:
        print(f"{q*100}%:\t{lengths.quantile(q)}")

if __name__ == "__main__":
    main(Path("/media/evo870/data/reach-bert-data/bert_files_w_negatives"), "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")