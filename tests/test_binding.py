import numpy as np
import torch

from visual_encoder.latent_port import (
    content_vs_order,
    damerau_levenshtein,
    warmstart_targets,
)


def test_mean_warmstart_is_permutation_invariant():
    # The confound the external review flagged: z(AB) == z(BA) under order="mean".
    table = torch.randn(32, 8)
    ab = warmstart_targets(["AB"], slots=1, char_table=table, order="mean")
    ba = warmstart_targets(["BA"], slots=1, char_table=table, order="mean")
    assert torch.allclose(ab, ba, atol=1e-6)


def test_role_warmstart_breaks_permutation_symmetry():
    table = torch.randn(32, 64)
    ab = warmstart_targets(["AB"], slots=1, char_table=table, order="role")
    ba = warmstart_targets(["BA"], slots=1, char_table=table, order="role")
    assert not torch.allclose(ab, ba, atol=1e-4)


def test_damerau_counts_transposition_as_one():
    assert damerau_levenshtein("CED", "CDE") == 1   # one adjacent swap
    assert damerau_levenshtein("abc", "abc") == 0


def test_content_vs_order_isolates_transposition():
    # decoded == payload with two chars swapped inside each slot: content perfect,
    # order imperfect.
    payload = "ABCD"
    decoded = "BADC"  # swap within each 2-char slot
    split = content_vs_order(payload, decoded, slots=2)
    assert split["unordered_char_accuracy"] == 1.0
    assert split["order_error_share"] > 0.0
