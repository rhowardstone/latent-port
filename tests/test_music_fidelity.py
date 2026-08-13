"""CPU regression for the LP-6 musical-accuracy metric. Skips if music21 is absent
(CI is CPU-only and may not install it) — this is helper-level coverage, like the
rest of the suite; the headline music numbers live in runs/music_port.json."""

import pytest

pytest.importorskip("music21")

from visual_encoder.music_port import music_fidelity  # noqa: E402

H = "X:1\nL:1/8\nM:4/4\nK:Cmaj\n"
TUNE = H + "|: G2 AB c2 BA | G2 E2 C4 :|"


def test_identical_is_perfect():
    f = music_fidelity(TUNE, TUNE)
    assert f["pitch_fid"] == 1.0 and f["note_f1"] == 1.0
    assert f["hist_cos"] == pytest.approx(1.0)  # cosine has a 1e-9 denom guard


def test_reorder_keeps_content_but_loses_order():
    # same notes, shuffled -> content (histogram) high, order (pitch_fid) lower.
    # This is the "generic plausible melody" detector; both must be reported.
    rev = H + "|: C4 E2 G2 | AB c2 BA G2 :|"
    f = music_fidelity(TUNE, rev)
    assert f["hist_cos"] > f["pitch_fid"]
    assert f["hist_cos"] > 0.9


def test_transpose_is_separated_from_garble():
    # a whole-tune transpose is perceptually near-perfect but exact-wrong: transpose_fid
    # must exceed raw pitch_fid, so "wrong key" is never scored the same as "garbled".
    tpose = H + "|: A2 Bc d2 cB | A2 F2 D4 :|"
    f = music_fidelity(TUNE, tpose)
    assert f["transpose_fid"] > f["pitch_fid"]


def test_empty_and_garbage_score_zero_not_raise():
    for bad in ["", "not abc", "\n\n"]:
        f = music_fidelity(TUNE, bad)
        assert f["pitch_fid"] == 0.0 and f["note_f1"] == 0.0
