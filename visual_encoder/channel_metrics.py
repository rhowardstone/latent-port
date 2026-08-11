"""Standardized channel metrics — one implementation every experiment calls.

The repo previously computed slightly different metric subsets per experiment and
conflated packet goodput with a BSC-equivalent rate (external reviews). This module
is the single source of truth, with the distinctions kept explicit:

- G(b)   packet goodput = loaded_bits * P[whole packet exact]     (operational)
- R_var  variational/GMI rate = H(X) + E[log2 q(X|Z)]             (LP-1, known H(X))
- dI     paired information gain = (NLL_null - NLL_correct)/ln2   (LP-2, free text)
- BER/SER, and BSC-equivalent rate labeled as such (NOT a Shannon capacity)
- message-cluster bootstrap CIs (resample whole messages; intra-message errors
  are correlated)

The pure math here is CPU-only and synthetic-validated (see synthetic_reference()).
The teacher-forced NLL it consumes for R_var/dI comes from a GPU forward pass in the
experiment code.
"""

from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)


# ---- discrete error rates (failure-safe: empty/malformed -> worst case) -------

def packet_exact_rate(payloads: list[str], decodes: list[str]) -> float:
    if not payloads:
        return 0.0
    return sum(p == d for p, d in zip(payloads, decodes)) / len(payloads)


def symbol_error_rate(payloads: list[str], decodes: list[str]) -> float:
    """Positional symbol error over min-length; length mismatch counts as error."""
    errs, total = 0, 0
    for p, d in zip(payloads, decodes):
        total += len(p)
        for i, pc in enumerate(p):
            if i >= len(d) or d[i] != pc:
                errs += 1
    return errs / total if total else 1.0


def bit_error_rate(payloads: list[str], decodes: list[str], alphabet: str) -> float:
    """Per-bit error using a fixed bit-width code over `alphabet`."""
    width = max(1, math.ceil(math.log2(len(alphabet))))
    index = {c: i for i, c in enumerate(alphabet)}
    errs, total = 0, 0
    for p, d in zip(payloads, decodes):
        for i, pc in enumerate(p):
            pv = index.get(pc, 0)
            dv = index.get(d[i], -1) if i < len(d) else -1
            for b in range(width):
                total += 1
                if ((pv >> b) & 1) != (((dv >> b) & 1) if dv >= 0 else (1 - ((pv >> b) & 1))):
                    errs += 1
    return errs / total if total else 1.0


def goodput_bits(loaded_bits_per_slot: float, exact_rate: float) -> float:
    """G(b): operational packet goodput. NOT a capacity."""
    return loaded_bits_per_slot * exact_rate


def bsc_equivalent_rate(loaded_bits_per_slot: float, ber: float) -> float:
    """b*(1 - H2(BER)). BSC-equivalent ONLY — correlated errors make it optimistic."""
    ber = min(max(ber, 0.0), 0.5)
    if ber in (0.0, 0.5):
        return loaded_bits_per_slot * (1.0 if ber == 0.0 else 0.0)
    h2 = -ber * math.log2(ber) - (1 - ber) * math.log2(1 - ber)
    return loaded_bits_per_slot * (1.0 - h2)


# ---- information rates from teacher-forced NLL --------------------------------

def variational_rate(nll_bits_per_message: np.ndarray, source_entropy_bits: float) -> dict:
    """R_var = H(X) + E[log2 q(X|Z)] = H(X) - E[NLL_bits].

    Valid ONLY when the source entropy H(X) is known (LP-1: uniform Base32,
    H(X)=5C). A negative result means the decoder is worse than the prior.
    """
    nll = np.asarray(nll_bits_per_message, dtype=np.float64)
    i_var = source_entropy_bits - float(nll.mean())
    return {
        "source_entropy_bits": source_entropy_bits,
        "mean_nll_bits": float(nll.mean()),
        "i_var_bits_per_message": i_var,
        "below_prior": i_var < 0,
    }


def paired_information_gain(nll_correct: np.ndarray, nll_null: np.ndarray) -> dict:
    """dI = (NLL_null - NLL_correct)/ln2 in bits (per message).

    For free text where H(X) is unknown: information the *specific* latent delivers
    over a null (deranged / moment-matched / zero) latent. Unifies the variational-
    rate and causal-control lanes. `nll_*` are natural-log NLLs (nats).
    """
    a = np.asarray(nll_correct, dtype=np.float64)
    b = np.asarray(nll_null, dtype=np.float64)
    gain = (b - a) / LN2
    return {
        "delta_i_bits_per_message_mean": float(gain.mean()),
        "delta_i_bits_per_message_median": float(np.median(gain)),
        "fraction_positive": float((gain > 0).mean()),
    }


# ---- message-cluster bootstrap ------------------------------------------------

def cluster_bootstrap_ci(
    per_message_values: np.ndarray, *, seed: int, n_boot: int = 2000, alpha: float = 0.05
) -> dict:
    """CI by resampling whole messages (intra-message errors are correlated)."""
    values = np.asarray(per_message_values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "bootstrap_seed": seed}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(values)),
        "n_boot": n_boot,
        "bootstrap_seed": seed,
    }


# ---- synthetic validation (the CPU reference oracle) --------------------------

def _apply_channel(payloads, alphabet, kind, rate, rng):
    out = []
    A = len(alphabet)
    for p in payloads:
        chars = list(p)
        if kind == "identity":
            pass
        elif kind == "bsc":  # each symbol independently replaced w.p. rate
            for i in range(len(chars)):
                if rng.random() < rate:
                    chars[i] = alphabet[rng.integers(0, A)]
        elif kind == "erasure":
            chars = [c for c in chars if rng.random() >= rate]
        elif kind == "dropout":  # whole-packet drop
            if rng.random() < rate:
                chars = []
        elif kind == "transpose":  # adjacent swaps w.p. rate
            i = 0
            while i < len(chars) - 1:
                if rng.random() < rate:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    i += 2
                else:
                    i += 1
        out.append("".join(chars))
    return out


def synthetic_reference(seed: int = 0) -> dict:
    """Known channels → what each metric does and does not estimate. The committed
    artifact (runs/cpu/channel_metrics_reference.json) documents this."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"  # base32
    rng = np.random.default_rng(seed)
    C = 32
    payloads = ["".join(alphabet[i] for i in rng.integers(0, 32, C)) for _ in range(400)]
    H_X = C * math.log2(len(alphabet))  # 5C bits, known
    report = {"source_entropy_bits": H_X, "chars": C, "channels": {}}
    for kind, rate in [("identity", 0.0), ("bsc", 0.05), ("bsc", 0.2), ("erasure", 0.1),
                       ("dropout", 0.1), ("transpose", 0.3)]:
        d = _apply_channel(payloads, alphabet, kind, rate, np.random.default_rng(seed + 1))
        ber = bit_error_rate(payloads, d, alphabet)
        # idealized decoder NLL: 0 bits on correct symbols, log2(A) on wrong ones
        per_msg_nll = np.array([
            sum(0.0 if (i < len(di) and di[i] == pc) else math.log2(len(alphabet))
                for i, pc in enumerate(p))
            for p, di in zip(payloads, d)
        ])
        report["channels"][f"{kind}@{rate}"] = {
            "packet_exact": packet_exact_rate(payloads, d),
            "symbol_error": symbol_error_rate(payloads, d),
            "bit_error": ber,
            "goodput_bits_per_slot": goodput_bits(5 * C / 16, packet_exact_rate(payloads, d)),
            "bsc_equiv_bits_per_slot": bsc_equivalent_rate(5 * C / 16, ber),
            **variational_rate(per_msg_nll, H_X),
        }
    return report


if __name__ == "__main__":
    import json
    from pathlib import Path

    ref = synthetic_reference()
    out = Path("runs/cpu/channel_metrics_reference.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2) + "\n")
    print(f"wrote {out}")
