"""Aggregation of per-output metrics into the reported numbers.

``cell_median`` (used for the tables): the metric of every
(profile, query radius) cell is the median over its mixtures; each category
(full / partial / empty) is the mean over its cells.

``profile_mean``: the metric is first averaged over all outputs of a profile
in the category (all radii and mixtures), then over profiles.

``equal_radius_cell_median``: cell medians are averaged within each radius
first and then over radii with equal weight.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from .metrics import CATEGORY_METRIC, finite


def _mean(values):
    values = [float(v) for v in values if finite(v)]
    return statistics.mean(values) if values else None


def summarize_rows(rows: list[dict]) -> dict:
    cells = defaultdict(list)
    profiles = defaultdict(list)
    for row in rows:
        category = row["category"]
        value = row[CATEGORY_METRIC[category]]
        if not finite(value):
            raise ValueError(f"Missing/non-finite {CATEGORY_METRIC[category]} in {row}")
        cells[(category, row["profile_id"], float(row["query_radius_m"]))].append(float(value))
        profiles[(category, row["profile_id"])].append(float(value))

    cell_rows = [
        {
            "category": category,
            "profile_id": profile_id,
            "query_radius_m": radius,
            "mixtures": len(values),
            "median": statistics.median(values),
        }
        for (category, profile_id, radius), values in sorted(cells.items())
    ]
    radii = sorted({cell["query_radius_m"] for cell in cell_rows})
    summary = {"cell_median": {}, "profile_mean": {}, "equal_radius_cell_median": {}, "counts": {}}
    for category in CATEGORY_METRIC:
        selected = [cell for cell in cell_rows if cell["category"] == category]
        summary["cell_median"][category] = _mean(cell["median"] for cell in selected)
        summary["profile_mean"][category] = _mean(
            statistics.mean(values) for (cat, _), values in profiles.items() if cat == category
        )
        per_radius = [
            _mean(cell["median"] for cell in selected if cell["query_radius_m"] == radius)
            for radius in radii
        ]
        summary["equal_radius_cell_median"][category] = (
            _mean(per_radius) if per_radius and all(v is not None for v in per_radius) else None
        )
        summary["counts"][category] = {
            "cells": len(selected),
            "outputs": sum(cell["mixtures"] for cell in selected),
        }
    summary["per_radius_cell_median"] = {
        f"{radius:g}": {
            category: _mean(
                cell["median"]
                for cell in cell_rows
                if cell["category"] == category and cell["query_radius_m"] == radius
            )
            for category in CATEGORY_METRIC
        }
        for radius in radii
    }
    return {"summary": summary, "cells": cell_rows}


def summarize_rce(pair_rows: list[dict]) -> dict:
    """Two statistics of the radius consistency error, in percent.

    ``adjacent_pairs``: median / mean over all adjacent query pairs whose
    target-speaker set is unchanged (including empty sets).
    ``per_mixture_nonempty``: per mixture, the mean over adjacent pairs with an
    unchanged non-empty target set; median / mean over mixtures with at least
    one such pair (Eq. 19 when evaluated on a dense query grid).
    """
    result = {}
    values = [row["rce"] for row in pair_rows]
    result["adjacent_pairs"] = {
        "pairs": len(values),
        "median_percent": 100 * statistics.median(values) if values else None,
        "mean_percent": 100 * statistics.mean(values) if values else None,
    }
    per_mixture = defaultdict(list)
    for row in pair_rows:
        if row["target_speakers"] > 0:
            per_mixture[row["mixture_index"]].append(row["rce"])
    mixture_values = [statistics.mean(v) for v in per_mixture.values()]
    result["per_mixture_nonempty"] = {
        "mixtures": len(mixture_values),
        "pairs": sum(len(v) for v in per_mixture.values()),
        "median_percent": 100 * statistics.median(mixture_values) if mixture_values else None,
        "mean_percent": 100 * statistics.mean(mixture_values) if mixture_values else None,
    }
    return result
