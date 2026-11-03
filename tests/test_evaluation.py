import torch

from cradius_rse.evaluation import consistency_error, evaluate_output, summarize_rce, summarize_rows


def test_metrics_per_category():
    target = torch.sin(torch.linspace(0, 100, 16000)).view(1, -1) * 0.1
    interferer = torch.cos(torch.linspace(0, 37, 16000)).view(1, -1) * 0.1
    mixture = target + interferer

    full = evaluate_output(target.clone(), target, target, 1)
    assert full["si_sdri_db"] is None and full["si_sdr_db"] > 50

    partial = evaluate_output(target.clone(), mixture, target, 1)
    assert partial["si_sdri_db"] > 0

    empty = evaluate_output(mixture * 0.01, mixture, torch.zeros_like(mixture), 0)
    assert abs(empty["attenuation_db"] - 40.0) < 1e-6


def test_aggregation_and_rce():
    rows = []
    for mixture_index, value in enumerate((1.0, 2.0, 9.0)):
        rows.append({"mixture_index": mixture_index, "profile_id": 0, "query_radius_m": 1.0,
                     "category": "full", "si_sdr_db": value})
    summary = summarize_rows(rows)["summary"]
    assert summary["cell_median"]["full"] == 2.0
    assert summary["profile_mean"]["full"] == 4.0

    mixture = torch.ones(1, 100)
    assert abs(consistency_error(torch.zeros(1, 100), torch.full((1, 100), 0.5), mixture) - 0.5) < 1e-12
    pairs = [
        {"mixture_index": 0, "target_speakers": 0, "rce": 0.0},
        {"mixture_index": 1, "target_speakers": 1, "rce": 0.02},
        {"mixture_index": 1, "target_speakers": 1, "rce": 0.04},
    ]
    rce = summarize_rce(pairs)
    assert rce["adjacent_pairs"]["pairs"] == 3
    assert abs(rce["per_mixture_nonempty"]["median_percent"] - 3.0) < 1e-9
