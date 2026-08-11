import math

import numpy as np

from visual_encoder.channel_metrics import (
    bit_error_rate,
    bsc_equivalent_rate,
    cluster_bootstrap_ci,
    goodput_bits,
    packet_exact_rate,
    paired_information_gain,
    symbol_error_rate,
    variational_rate,
)

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"  # 32 symbols -> 5 bits


def test_identity_channel_metrics():
    p = ["ABCD", "EFGH"]
    assert packet_exact_rate(p, p) == 1.0
    assert symbol_error_rate(p, p) == 0.0
    assert bit_error_rate(p, p, ALPHA) == 0.0


def test_empty_and_malformed_decode_never_raise():
    assert packet_exact_rate(["ABCD"], [""]) == 0.0
    assert symbol_error_rate(["ABCD"], [""]) == 1.0
    assert 0.0 <= bit_error_rate(["ABCD"], ["AB"], ALPHA) <= 1.0


def test_goodput_is_not_capacity():
    # goodput scales linearly with exact rate; distinct from bsc-equivalent
    assert goodput_bits(10.0, 0.5) == 5.0
    assert bsc_equivalent_rate(10.0, 0.0) == 10.0
    assert bsc_equivalent_rate(10.0, 0.5) == 0.0
    # monotonically decreasing in BER, and always below the error-free rate
    assert bsc_equivalent_rate(10.0, 0.2) < bsc_equivalent_rate(10.0, 0.05) < 10.0


def test_variational_rate_perfect_and_uniform_decoders():
    C, A = 8, 32
    H_X = C * math.log2(A)  # 40 bits
    perfect = np.zeros(200)                       # NLL 0 -> I_var = H(X)
    uniform = np.full(200, C * math.log2(A))      # NLL = H(X) -> I_var = 0
    assert abs(variational_rate(perfect, H_X)["i_var_bits_per_message"] - H_X) < 1e-9
    assert abs(variational_rate(uniform, H_X)["i_var_bits_per_message"]) < 1e-9


def test_variational_rate_flags_below_prior():
    r = variational_rate(np.full(50, 100.0), source_entropy_bits=40.0)
    assert r["below_prior"] is True
    assert r["i_var_bits_per_message"] < 0


def test_paired_information_gain_sign():
    # correct latent has lower NLL than null -> positive gain
    correct = np.full(100, 2.0)   # nats
    null = np.full(100, 5.0)
    g = paired_information_gain(correct, null)
    assert g["delta_i_bits_per_message_mean"] > 0
    assert abs(g["delta_i_bits_per_message_mean"] - (5.0 - 2.0) / math.log(2)) < 1e-9


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.7, 0.1, 300)
    ci = cluster_bootstrap_ci(vals, seed=1)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["bootstrap_seed"] == 1 and ci["n"] == 300
