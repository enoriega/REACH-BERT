from pathlib import Path
from typing import NamedTuple, List, Optional, Sequence, Mapping

from torch import Tensor


class InputSequence(NamedTuple):
    """ Represents and input data point """
    event_labels: List[str]
    tags: List[str]
    words: List[str]
    masked_words: Optional[List[str]]

    # Convenienc method to return the lenght of the sequence
    def __len__(self):
        return len(self.tags)

class TransformerInput(NamedTuple):
    """ Represents an input sequence encoded by a tokenizer ready to pass through a transformer """
    event_labels: List[str]
    tags: List[str]
    tensor: Tensor # Instead of words, we have a tensor (matrix) with the embedded words


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
            line = line.strip('\n')
            if not buffer:
                buffer = list()
            if line:
                buffer.append(line)
            else:
                ret.append(InputSequence(buffer[0].split('\t'), buffer[1].split('\t'), buffer[2].split('\t'), None))
                buffer = None

    # In case there is lingering data in the  buffer that hasn't been cleared out
    if buffer and len(buffer) == 3:
        ret.append(InputSequence(buffer[0].split('\t'), buffer[1].split('\t'), buffer[2].split('\t'), None))

    return ret

def marshall_inputs(input_sequences:Sequence[InputSequence]) -> Sequence[Mapping[str, Tensor]]:
    # Map to the labels using BatchEncoding.word_to_token
    # call the tokenizer using is_split_into_words = True kwarg
    pass