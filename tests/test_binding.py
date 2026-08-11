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


def test_transposition_flagged_by_dl_vs_lev():
    # A pure adjacent transposition: DL=1 < Levenshtein=2, so transposition_share>0
    # (this is the clean transposition signal; global NW alignment is indel-robust).
    split = content_vs_order("ABCD", "BACD", slots=2)  # swap first pair only
    assert split["transposition_share"] > 0.0
    assert split["unordered_char_accuracy"] >= 0.5


def test_content_vs_order_empty_decode_scores_zero_not_crash():
    # The bug the review flagged: empty decode must return order_error_share, not crash.
    split = content_vs_order("ABCD", "", slots=2)
    assert split["unordered_char_accuracy"] == 0.0
    assert split["order_error_share"] == 0.0


def test_content_vs_order_insertion_does_not_masquerade():
    # A single early insertion shifts everything; alignment should keep content ~perfect.
    payload = "ABCDEF"
    decoded = "XABCDEF"  # one inserted char at the front
    split = content_vs_order(payload, decoded, slots=3)
    assert split["unordered_char_accuracy"] >= 5 / 6  # not near-zero


def test_blend_warmstart_interpolates():
    table = torch.randn(32, 64)
    payloads = ["AB"]
    mean_t = warmstart_targets(payloads, 1, table, order="mean")
    blend_lo = warmstart_targets(payloads, 1, table, order="blend", eps=0.1)
    # small eps stays close to the on-manifold mean target
    assert (blend_lo - mean_t).norm() < (warmstart_targets(payloads, 1, table, order="role") - mean_t).norm()
