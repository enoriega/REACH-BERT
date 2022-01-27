""" Generate masked input files for REACH BERT inspired by the approached used by RoBERTa """
import itertools as it
from pathlib import Path
from typing import NamedTuple, List, Optional, Set

import plac
import numpy as np
from tqdm import tqdm

rng: np.random.Generator # This will be the random number generator
vocab: List[str] # Holds the vocabulary used for masking

@plac.pos("input_path", type= Path, help="Input file or directory containing files to be processed")
@plac.opt("output_dir", type= Path, help="Output directory")
@plac.opt("num_copies", type= int, help="Number of masked copies to generate per input file", abbrev="n")
@plac.opt("seed", type = int, help= "Random seed used to generate the masks", abbrev="s")
def main(input_path:Path, output_dir: Optional[Path], num_copies:int = 10, seed: int = 0) -> None:
    """
    Main function of the app.
    Dispatches each individual file to the processing function

    :param output_dir: optional directory where to save the masked files. Defaults to CWD
    :param seed: Random seed used to generate the masks
    :param input_path: Input file or directory with input files
    :param num_copies: Number of masked copies to generate per input file
    """

    # Configure the global rng
    global rng, vocab
    rng = np.random.default_rng(seed)

    # Set the default value if no output_dir was provided
    if not output_dir:
        output_dir = Path()

    # If the output dir doesn't exist, create it
    if not output_dir.exists():
        output_dir.mkdir(parents= True)

    # Generate the vocabulary from the input files
    vocab = build_vocab(input_path)

    # Process all files in the input directory
    if input_path.is_dir():
        mask_directory(input_path, num_copies, output_dir)
    # Process the single input file
    else:
        mask_file(input_path, num_copies, output_dir)

class InputSequence(NamedTuple):
    """ Represents and input data point """
    event_labels: List[str]
    tags: List[str]
    words: List[str]

    # Convenienc method to return the lenght of the sequence
    def __len__(self):
        return len(self.tags)

def build_vocab(input_path: Path) ->  List[str]:
    """
    Builds the vocabulary from ``input_path``
    :param input_path: to file or directory with files
    :return: vocabulary present in inputs
    """

    if input_path.is_file():
        paths = [input_path]
    else:
        paths = input_path.iterdir()

    x= np.asarray(list(set(it.chain.from_iterable(
        seq.words for
        seq in it.chain.from_iterable(
            parse_input_file(path) for path  in tqdm(paths, desc="Building vocab", unit=" files"))))))

    return x


def parse_input_file(path:Path) -> List[InputSequence]:
    """
    Parses the contents of :path: into a list of InputSequence instances

    :param path: to the input file
    :return: list of Input Sequence instances contained in the input file
    """

    ret = list()

    buffer = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not buffer:
                buffer = list()
            if line:
                buffer.append(line)
            else:
                ret.append(InputSequence(buffer[0].split('\t'), buffer[1].split('\t'), buffer[2].split('\t')))
                buffer = None

    # In case there is lingering data in the  buffer that hasn't been cleared out
    if buffer and len(buffer) == 3:
        ret.append(InputSequence(buffer[0].split('\t'), buffer[1].split('\t'), buffer[2].split('\t')))

    return ret

def mask_file(input_file: Path, num_copies: int, output_dir: Path) -> None:
    """
    Generates ``num_copies`` masked variations of each ``InputSequence`` containes in ``input_file``

    :param output_dir: to save the masked files
    :param input_file: path that contains the input sequences to mask
    :param num_copies: number of masked copies to generate
    :return: Nested list of masked variations. The outer level indexes the original input sequence,
    the inner lists contain the ``num_copies`` variations
    """

    def make_variation_path(orig_path: Path, variation_num: int) -> Path:
        """
        Generates a new Path object for the specified variation number

        :param orig_path:
        :param variation_num: number tu use on the returned path
        :return:
        """

        base_name = orig_path.stem
        suffix = orig_path.suffix

        return output_dir / f"{base_name}_{variation_num}{suffix}"

    sequences = parse_input_file(input_file)

    for copy_n in range(num_copies):
        variant_path = make_variation_path(input_file, copy_n + 1)
        with variant_path.open('w') as f:
            for sequence in sequences:
                try:
                    variation = mask_sequence(sequence)
                    # Save the variation to disk
                    f.write('\t'.join(variation.event_labels) + "\n")
                    f.write('\t'.join(variation.tags) + "\n")
                    f.write('\t'.join(variation.words) + "\n")
                    f.write("\n")
                except Exception as ex:
                    print(ex)

def mask_directory(input_directory: Path, num_copies: int, output_dir: Path) -> None:
    """
    Generates `num_copies` masked variants of each file in `input_directory`
    :param output_dir: to save the masked files
    :param input_directory: path containing input files
    :param num_copies: to be generated of the data in each file
    """

    # Count the number of files ahead of time to display progress
    num_files = 0
    for _ in input_directory.iterdir():
        num_files += 1

    for input_file in tqdm(input_directory.iterdir(), desc="Generating masked files", unit=" files", total= num_files):
        if input_file.is_file():
            mask_file(input_file, num_copies, output_dir)


def mask_sequence(original_seq: InputSequence) -> InputSequence:
    """
    Efficiently masks an input acording to the BERT masking algorithm

    :param original_seq: to mask
    :return: A masked instance
    """

    # Sequence lenght
    seq_len = len(original_seq)
    # Numpy array with the original words in the input
    original_words = np.asarray(original_seq.words)
    # First sample a mask that selects 15% of the words
    choices = rng.choice([0, 1], (seq_len,), p=[.85, .15])
    # Choose the masking type
    masking_type = rng.choice([1, 2, 3], (seq_len,), p=[.8, .1, .1])
    # Sample tokens randomly from the vocabulary
    random_tokens = rng.choice(vocab, seq_len)
    # Array of mask tokens
    mask_tokens = np.full((seq_len,), "[MASK]")

    # Efficient masking using numpy operations
    new_words = np.where(choices == 0,
                         original_words,
                         np.where(masking_type == 2,
                          random_tokens,
                          np.where(masking_type == 1,
                                   mask_tokens,
                                   original_words)
                          )
                         )

    return InputSequence(
            event_labels=original_seq.event_labels,
            tags= original_seq.tags,
            words= list(new_words)
        )


if __name__ == "__main__":
    plac.call(main)