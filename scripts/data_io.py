#!/usr/bin/python3
# -*- coding: utf-8 -*-

import gzip, re
import collections
from dataclasses import dataclass
from typing import Sequence, Mapping

from Bio import SeqIO
import numpy as np



def get_id(index):
    return index + 1


def get_fwd_id(x):
    return x if x >= 0 else -x


def get_sibling_id(x):
    return -x


def is_fwd_id(x):
    return x >= 0


def load_reads(fasta_path):
    read_indices = {}
    read_sizes = {}
    with gzip.open(fasta_path, "rt") as f:
        for i, record in enumerate(SeqIO.parse(f, "fasta")):
            read_id = get_id(i)
            read_sizes[read_id] = len(record.seq)
            read_indices[record.id] = read_id
    return read_sizes, read_indices
