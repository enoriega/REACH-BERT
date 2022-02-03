from pathlib import Path
from typing import NamedTuple, List, Optional


class InputSequence(NamedTuple):
    """ Represents and input data point """
    event_labels: List[str]
    tags: List[str]
    words: List[str]
    masked_words: Optional[List[str]]

    # Convenienc method to return the lenght of the sequence
    def __len__(self):
        return len(self.tags)


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