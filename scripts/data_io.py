#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import List, Optional, Iterable
import gzip
from typing import (
    Mapping,
    Sequence,
    Optional,
    Collection,
    MutableMapping,
    Optional,
    Iterable,
    Iterator,
    overload,
    Literal,
    BinaryIO,
    TextIO,
    IO,
    Any,
    Generator,
    TypeVar,
    Generic,
    cast,
)
from dataclasses import dataclass, field
from isal import igzip
T = TypeVar("T")
def get_fwd_id(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def get_sibling_id(x: int) -> int:
    if x % 2 == 0:
        return x + 1
    else:
        return x - 1


def is_fwd_id(x: int) -> bool:
    return x % 2 == 0


def open_gzipped(path, mode="rt", gzipped: Optional[bool] = None, **kw):
    if gzipped is None:
        gzipped = path.endswith(".gz")
    if gzipped:
        open_ = igzip.open
        return open_(path, mode)
    else:
        open_ = open
    return open_(path, mode, **kw)

class FastxRecord:
    name: str
    sequence: str

class FastaRecord(FastxRecord):
    name: str
    sequence: str

    def __post_init__(self):
        self.sequence = self.sequence.upper()

    @staticmethod
    def from_lines(lines: Sequence[str]) -> 'FastaRecord':
        name = lines[0][1:-1].split()[0]
        sequence = ''.join(line.strip() for line in lines[1:])  # 合并多行序列
        return FastaRecord(name, sequence)

@dataclass
class _DataLoader(Generic[T]):
    file_path: str

    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError
    
    def open(self):
        return self


class FastaLoader(_DataLoader[FastaRecord]):
    @staticmethod
    def _read_item(file_obj: TextIO) -> Iterator[Sequence[str]]:
        item = []
        for line in file_obj:
            if line.startswith('>'):
                if item:  # 如果已经有数据，先返回
                    yield item
                    item = []
                item.append(line)  # 添加 header
            else:
                item.append(line)  # 添加序列行
        if item:  # 返回最后一个记录
            yield item


    @staticmethod
    def _parse_item(item: Sequence[str]) -> FastaRecord:
        return FastaRecord.from_lines(item)

    def __iter__(self) -> Iterator[FastaRecord]:
        with open_gzipped(self.file_path, 'rt') as f:
            for item in self._read_item(f):
                yield self._parse_item(item)

class PAFRecord:
    def __init__(self, line: str) -> None:
        fields = line.strip().split('\t')
        self.query_name: str = fields[0]
        self.query_length: int = int(fields[1])
        self.query_start: int = int(fields[2])
        self.query_end: int = int(fields[3])
        self.strand: str = fields[4]
        self.target_name: str = fields[5]
        self.target_length: int = int(fields[6])
        self.target_start: int = int(fields[7])
        self.target_end: int = int(fields[8])
        self.num_matching_bases: int = int(fields[9])
        self.alignment_block_length: int = int(fields[10])
        self.mapping_quality: int = int(fields[11])
        self.optional_fields: Optional[List[str]] = fields[12:] if len(fields) > 12 else []

    def __str__(self) -> str:
        return (f"Query: {self.query_name}, Target: {self.target_name}, "
                f"Strand: {self.strand}, Mapping Quality: {self.mapping_quality}")


def parse_paf_file(file_path: str) -> Iterable[PAFRecord]:
    if file_path.endswith(".gz"):
        open_ = gzip.open
    else:
        open_ = open
    with open_(file_path, 'rt') as file:
        for line in file:
            if line.strip():
                record = PAFRecord(line)
                yield record




