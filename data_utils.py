from pathlib import Path
from typing import NamedTuple, List, Optional, Sequence, Mapping

from tokenizers import Tokenizer
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

def _buffer_to_sequence(buffer: List[str]) -> InputSequence:
    return InputSequence([l for l in buffer[0].strip('\t').split('\t') if l], buffer[1].strip('\t').split('\t'), buffer[2].split('\t'), None)

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
                ret.append(_buffer_to_sequence(buffer))
                buffer = None

    # In case there is lingering data in the  buffer that hasn't been cleared out
    if buffer and len(buffer) == 3:
        ret.append(_buffer_to_sequence(buffer))

    return ret