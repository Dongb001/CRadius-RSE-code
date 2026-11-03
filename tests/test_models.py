from pathlib import Path

import pytest
import torch

from cradius_rse.models import ContinuousRadiusExtractionNetwork
from cradius_rse.utils import build_model, load_json, load_model_weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_continuous_network_shape_and_size():
    model = build_model(load_json("configs/cradius_rse/continuous_crc.json")).eval()
    # Six TF blocks with five FiLM modules between them.
    assert len(model.tf_blocks) == 6 and len(model.films) == 5
    assert count_parameters(model) == 685_070
    mixture = torch.randn(2, 1, 16000) * 0.05
    with torch.no_grad():
        out = model({"mixture": mixture, "query_radius": torch.tensor([[0.0], [3.7]])})["output"]
    assert out.shape == mixture.shape and torch.isfinite(out).all()


def test_continuous_radius_range_is_enforced():
    model = ContinuousRadiusExtractionNetwork().eval()
    with pytest.raises(ValueError):
        model({"mixture": torch.zeros(1, 1, 4000), "query_radius": torch.tensor([[4.5]])})


def test_discrete_network_accepts_only_training_radii():
    model = build_model(load_json("configs/ablation/discrete_crc.json")).eval()
    assert count_parameters(model) == 657_114
    mixture = torch.randn(4, 1, 8000) * 0.05
    with torch.no_grad():
        out = model({"mixture": mixture, "query_radius": torch.tensor([[1.0], [2.0], [3.0], [4.0]])})["output"]
    assert out.shape == mixture.shape
    with pytest.raises(ValueError):
        model({"mixture": mixture[:1], "query_radius": torch.tensor([[1.5]])})


def test_output_length_matches_input_for_odd_lengths():
    model = ContinuousRadiusExtractionNetwork().eval()
    for length in (1000, 12345, 16000):
        with torch.no_grad():
            out = model({"mixture": torch.randn(1, length) * 0.05, "query_radius": torch.tensor([2.0])})["output"]
        assert out.shape == (1, 1, length)


RELEASED = {
    "continuous_crc": "configs/cradius_rse/continuous_crc.json",
    "discrete_init": "configs/cradius_rse/discrete_init.json",
    "discrete_crc": "configs/ablation/discrete_crc.json",
    "continuous_wo_crc": "configs/ablation/continuous_wo_crc.json",
}


def test_released_checkpoints_load():
    for name, config in sorted(RELEASED.items()):
        path = Path("checkpoints") / f"{name}.pt"
        model, _ = load_model_weights(build_model(load_json(config)), path)
        with torch.no_grad():
            mixture = torch.randn(1, 1, 8000) * 0.05
            out = model.eval()({"mixture": mixture, "query_radius": torch.tensor([[2.0]])})["output"]
        assert torch.isfinite(out).all(), name
