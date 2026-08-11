import torch

from visual_encoder.qwen_probe import bsc_equivalent_rate, fit_probe


def test_ridge_probe_generalizes_and_reports_train_gap():
    generator = torch.Generator().manual_seed(99)
    weight = torch.randn(12, 3, generator=generator)
    train_x = torch.randn(20, 8, 12, generator=generator)
    val_x = torch.randn(10, 8, 12, generator=generator)
    train_y = (train_x @ weight > 0).to(torch.uint8)
    val_y = (val_x @ weight > 0).to(torch.uint8)
    result, _ = fit_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        kind="ridge",
        epochs=1,
        device="cpu",
        seed=7,
        ridge_lambdas=[0.001, 0.01, 0.1],
    )
    assert result["train_bit_accuracy"] > 0.95
    assert result["bit_accuracy"] > 0.9
    assert result["selected_ridge"] in (0.001, 0.01, 0.1)
    assert sum(result["image_bit_error_histogram"].values()) == 10


def test_bsc_equivalent_rate_bounds():
    assert bsc_equivalent_rate(8, 1.0) == 8
    assert bsc_equivalent_rate(8, 0.5) == 0
    assert 0 < bsc_equivalent_rate(8, 0.9) < 8
