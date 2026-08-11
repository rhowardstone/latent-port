import numpy as np

from visual_encoder.patterns import (
    clipping_fraction,
    make_basis,
    random_payload_images,
    render_payload_grid,
)


def test_basis_is_deterministic_and_centered():
    first = make_basis(8, 28, seed=7)
    second = make_basis(8, 28, seed=7)
    assert np.array_equal(first, second)
    assert np.max(np.abs(first.mean(axis=(1, 2, 3)))) < 1e-6


def test_render_dimensions_and_variation():
    labels = np.zeros((3, 4, 8), dtype=np.uint8)
    labels[1, 2] = 1
    image = render_payload_grid(labels, patch=28)
    assert image.size == (4 * 28, 3 * 28)
    arr = np.asarray(image)
    assert arr.std() > 0


def test_random_payload_split_shape():
    images, labels = random_payload_images(samples=5, grid=3, bits=4, patch=28, seed=9)
    assert len(images) == 5
    assert images[0].size == (84, 84)
    assert labels.shape == (5, 9, 4)


def test_default_clipping_is_logged_and_limited():
    labels = np.random.default_rng(4).integers(0, 2, (8, 8, 16), dtype=np.uint8)
    fraction = clipping_fraction(labels, patch=32, amplitude=64)
    assert 0 <= fraction < 0.08
