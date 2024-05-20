#!/usr/bin/python3
# -*- coding: utf-8 -*-



def get_fwd_id(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def get_sibling_id(x: int) -> int:
    if x % 2 == 0:
        return x + 1
    else:
        return x - 1


def is_fwd_id(x: int) -> bool:
    return x % 2 == 0
