from isal import igzip
from typing import Sequence, Type, Mapping, Iterable, Literal

def open_gzipped(path, mode="rt", gzipped: bool | None = None, **kw):
    if gzipped is None:
        gzipped = path.endswith(".gz")
    if gzipped:
        open_ = igzip.open
        return open_(path, mode)
    else:
        open_ = open
    return open_(path, mode, **kw)

def parse_fasta(filepath: str) -> Iterable[tuple[str, str]]:
    """
    A generator function to parse a FASTA file.

    Parameters:
    filename (str): The path to the FASTA file.

    Yields:
    tuple: A tuple containing the sequence header and the sequence.
    """
    name = None
    sequence = []

    with open_gzipped(filepath, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    yield (name, "".join(sequence))
                name = line.split(" ")[0][1:]  # Exclude comments and the '>' character
                sequence = []
            else:
                sequence.append(line)

        if name:
            yield (name, "".join(sequence))