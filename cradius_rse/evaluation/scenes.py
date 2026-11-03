"""Deterministic evaluation scenes shared by all systems.

The default protocol uses every source profile with 0, 1 or 2 sources over
the five distance intervals (1 + 5 + 15 = 21 profiles), three mixtures per
profile, clean inputs (zero-source mixtures are noise-only), source gains of
0 dB with the distance-symmetric near/far relative gain, and query radii of
1, 2, 3 and 4 m.
"""

from __future__ import annotations

from ..data.regional_mixtures import RegionalMixtureDataset, all_source_profiles


def evaluation_dataset(
    data_args: dict,
    split: str = "validation",
    source_counts=(0, 1, 2),
    mixtures_per_profile: int = 3,
    seed: int = 20260826,
    radii=(1.0, 2.0, 3.0, 4.0),
    noise_snr_db: float | None = None,
) -> RegionalMixtureDataset:
    args = dict(data_args)
    profiles = all_source_profiles(
        {str(int(count)): 1 for count in source_counts}, len(args["distance_bins_m"])
    )
    args.update(
        base_mixtures_per_epoch=len(profiles) * int(mixtures_per_profile),
        source_profiles=[
            {"source_count": p.source_count, "distance_bin_ids": list(p.distance_bin_ids)}
            for p in profiles
        ],
        profile_weight_per_source_count=None,
        query_radii=[float(r) for r in radii],
        query_sampling="fixed",
        query_sampling_bins_m=None,
        noise_probability=0.0 if noise_snr_db is None else 1.0,
        source_gain_db_min=0.0,
        source_gain_db_max=0.0,
        seed=int(seed),
    )
    if noise_snr_db is not None:
        args.update(noise_snr_min=float(noise_snr_db), noise_snr_max=float(noise_snr_db))
    dataset = RegionalMixtureDataset(**args, split=split)
    dataset.set_epoch(0)
    return dataset
