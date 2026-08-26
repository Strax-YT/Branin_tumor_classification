import torch
import torch.nn.functional as F

from direct_model_loader import load_direct_model


def test_model_loads_successfully():
    model = load_direct_model()
    assert model is not None


def test_model_output_has_four_classes():
    model = load_direct_model()
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 4)


def test_softmax_sums_to_one():
    model = load_direct_model()
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = F.softmax(output, dim=1)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5)
