import argparse

import pytest

from visual_encoder.llm_probe import parse_layers


def test_parse_layers_is_sorted_and_unique():
    assert parse_layers("8,1,3,1") == [1, 3, 8]


def test_parse_layers_rejects_negative_values():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_layers("0,-1")
