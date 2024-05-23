#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import List, Optional, Iterable
import gzip

def get_fwd_id(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def get_sibling_id(x: int) -> int:
    if x % 2 == 0:
        return x + 1
    else:
        return x - 1


def is_fwd_id(x: int) -> bool:
    return x % 2 == 0



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




