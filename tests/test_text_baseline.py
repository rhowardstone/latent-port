from visual_encoder.text_baseline import levenshtein, random_base32, render_text_grid


def test_random_base32_is_deterministic_and_valid():
    first = random_base32(100, 8)
    assert first == random_base32(100, 8)
    assert len(first) == 100
    assert set(first) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


def test_rendered_text_layout():
    image, layout = render_text_grid("ABCDEFG234567" * 4, side=256, font_size=12)
    assert image.size == (256, 256)
    assert layout["character_capacity"] >= 52


def test_levenshtein():
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("ABC", "ADC") == 1
    assert levenshtein("", "ABC") == 3
