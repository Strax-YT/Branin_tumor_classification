import pytest

from api.service import _decode_image, InvalidImageError


def test_decode_valid_image_returns_rgb(sample_image_bytes):
    image = _decode_image(sample_image_bytes("glioma"))
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0


def test_decode_corrupted_image_raises_invalid_image_error(corrupted_image_bytes):
    with pytest.raises(InvalidImageError):
        _decode_image(corrupted_image_bytes)


def test_decode_empty_bytes_raises_invalid_image_error():
    with pytest.raises(InvalidImageError):
        _decode_image(b"")
