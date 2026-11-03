import torch

from cradius_rse.losses import CRCObjective, RegionalExtractionLoss

SAMPLES = 32


def grouped_example():
    """Two mixtures x four increasing radii with nested target sets."""
    source_a = torch.linspace(-0.2, 0.2, SAMPLES).view(1, 1, -1)
    source_b = torch.linspace(0.1, -0.1, SAMPLES).view(1, 1, -1)
    targets = torch.cat([source_a, source_a, source_a + source_b, source_a + source_b]).repeat(2, 1, 1)
    targets[4] = 0.0
    targets[5] = source_a
    targets[6] = source_a
    targets[7] = source_a + source_b
    counts = torch.tensor([1, 1, 2, 2, 0, 1, 1, 2])
    radii = torch.tensor([[0.5], [1.5], [2.5], [3.5]]).repeat(2, 1)
    return targets, counts, radii


def objective():
    return CRCObjective(lambda_con=0.1, queries_per_mixture=4, rms_threshold=1e-4,
                        zero_target_weight=100, straight_through_clip=True)


def test_exact_estimate_has_zero_loss():
    targets, counts, radii = grouped_example()
    loss, aux = objective()(targets.clone(), targets, num_target_speakers=counts, query_radius=radii)
    assert loss.abs().item() < 1e-6
    assert aux["crc"].abs().item() < 1e-6


def test_valid_pairs_are_unchanged_nonempty_targets():
    targets, counts, radii = grouped_example()
    est = targets.clone()
    est[1] += 0.01
    est[5] += 0.01
    est.requires_grad_(True)
    loss, aux = objective()(est, targets, num_target_speakers=counts, query_radius=radii)
    loss.backward()
    assert torch.isfinite(est.grad).all()
    # Mixture 0 contributes q0->q1 and q2->q3; mixture 1 contributes q1->q2 only
    # (q0 of mixture 1 is an empty region).
    assert int(aux["crc_valid_pairs"]) == 3
    assert [int(aux[f"crc_q{i}to{i + 1}_count"]) for i in range(3)] == [1, 1, 1]
    assert aux["crc"].item() > 0


def test_pairs_with_new_speakers_or_silence_are_skipped():
    targets, _, radii = grouped_example()
    _, aux = objective()(torch.randn(4, 1, SAMPLES), targets[:4],
                         num_target_speakers=torch.tensor([0, 1, 2, 3]), query_radius=radii[:4])
    assert int(aux["crc_valid_pairs"]) == 0 and aux["crc"].item() == 0.0
    zeros = torch.zeros(4, 1, SAMPLES)
    _, aux = objective()(torch.randn_like(zeros), zeros,
                         num_target_speakers=torch.zeros(4, dtype=torch.long), query_radius=radii[:4])
    assert int(aux["crc_valid_pairs"]) == 0


def test_consistency_value_matches_eq13():
    targets, counts, radii = grouped_example()
    est = targets.clone()
    est[1] += 0.01  # mixture 0, pair q0 -> q1
    _, aux = objective()(est, targets, num_target_speakers=counts, query_radius=radii)
    common_rms = targets[0].pow(2).mean().sqrt()
    pair = 0.01 / common_rms
    # Slots q1->q2 and q2->q3 are valid but exactly consistent: mean over 3 slots.
    assert torch.isclose(aux["crc"], pair / 3, rtol=1e-4)


def test_radii_must_increase_within_a_mixture():
    targets, counts, radii = grouped_example()
    try:
        objective()(targets, targets, num_target_speakers=counts, query_radius=radii.flip(0))
    except ValueError:
        return
    raise AssertionError("decreasing radii were accepted")


def test_empty_target_penalty():
    loss_fn = RegionalExtractionLoss(zero_target_weight=100)
    zero, _ = loss_fn(torch.zeros(1, 1, SAMPLES), torch.zeros(1, 1, SAMPLES))
    assert zero.item() == 0.0
    loss, _ = loss_fn(torch.full((1, 1, SAMPLES), 0.01), torch.zeros(1, 1, SAMPLES))
    assert torch.isclose(loss, torch.tensor([1.0]))
