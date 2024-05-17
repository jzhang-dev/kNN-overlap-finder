#!/usr/bin/env python
# coding: utf-8

"""
This module is designed to be imported in Snakemake job scripts to suppress type checking errors. 

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *
"""

from typing import Mapping, Any

class SnakemakeContext:
    input: Mapping[str, str] = {}
    output: Mapping[str, str] = {}
    wildcards: Mapping[str, str] = {}
    params: Mapping[str, Any] = {}
    threads: int = 1
    resources: Mapping[str, Any] = {}

snakemake = SnakemakeContext()