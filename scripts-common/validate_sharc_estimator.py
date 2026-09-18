#!/usr/bin/env python3
"""Numerical contract tests for the Remix SHARC estimator adapter.

These tests intentionally model the small adapter contract rather than SHARC's
hashing or quantization.  ``accumulate_radiance`` is the equivalent of
``SharcUpdateMiss``: it propagates a raw terminal contribution through the
weights of all retained inserted vertices, including the current vertex.
``accumulate_throughput`` is the
equivalent of ``SharcSetThroughput`` and updates those prior weights.  A hit
is inserted after hit emission/MIS and before its local NEE hook, so the NEE
hook accounts for both the new vertex and its predecessors exactly once.

The finite-depth case is covered explicitly: SHARC retains the newest
propagation-depth entries and drops the oldest one.  Hash-table insertion
failure is a separate runtime fallback concern and is not modeled here.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass


def _add(a: float, b: float) -> float:
    return a + b


@dataclass
class ContractCache:
    """Minimal scalar equivalent of SHARC's ordered propagation state."""

    propagation_depth: int = 8

    def __post_init__(self) -> None:
        self.values: list[float] = []
        self.weights: list[float] = []

    def accumulate_radiance(self, raw: float) -> None:
        """Propagate a raw contribution to all earlier inserted vertices."""
        for index, weight in enumerate(self.weights):
            self.values[index] = _add(self.values[index], raw * weight)

    def accumulate_throughput(self, segment: float) -> None:
        """Apply a segment BRDF/PDF weight to earlier cache vertices."""
        for index in range(len(self.weights)):
            self.weights[index] *= segment

    def insert_eligible_hit(self) -> bool:
        if len(self.values) >= self.propagation_depth:
            self.values.pop(0)
            self.weights.pop(0)
        self.values.append(0.0)
        self.weights.append(1.0)
        return True

    def nee_hook(self, raw: float) -> None:
        """Account for local NEE once at current and prior cache vertices."""
        self.accumulate_radiance(raw)


class SharcEstimatorContractTests(unittest.TestCase):
    def assertValues(self, cache: ContractCache, expected: list[float]) -> None:
        self.assertEqual(len(cache.values), len(expected))
        for actual, wanted in zip(cache.values, expected):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_two_bounces_propagate_and_nee_is_once(self) -> None:
        cache = ContractCache()
        self.assertTrue(cache.insert_eligible_hit())  # A, direct/emission = 0
        cache.accumulate_throughput(0.5)
        cache.accumulate_radiance(2.0)  # B hit emission, before inserting B
        self.assertTrue(cache.insert_eligible_hit())  # B
        cache.nee_hook(4.0)
        self.assertValues(cache, [3.0, 4.0])

    def test_three_bounces_with_unequal_throughput(self) -> None:
        cache = ContractCache()
        cache.insert_eligible_hit()  # A
        cache.accumulate_throughput(0.5)
        cache.accumulate_radiance(2.0)  # B emission
        cache.insert_eligible_hit()  # B
        cache.accumulate_throughput(0.25)
        cache.accumulate_radiance(3.0)  # C emission
        cache.insert_eligible_hit()  # C
        cache.nee_hook(4.0)
        # Independent backward recurrence: each event reaches a vertex with
        # the product of segments between that vertex and the event.
        expected = [
            2.0 * 0.5 + 3.0 * 0.5 * 0.25 + 4.0 * 0.5 * 0.25,
            3.0 * 0.25 + 4.0 * 0.25,
            4.0,
        ]
        self.assertValues(cache, expected)

    def test_attenuation_and_skipped_surface_still_reach_prior_hit(self) -> None:
        cache = ContractCache()
        cache.insert_eligible_hit()  # A
        cache.accumulate_throughput(0.8 * 0.5)  # BRDF/PDF and segment attenuation
        cache.accumulate_radiance(5.0)  # emissive skipped surface; no insertion
        cache.accumulate_throughput(0.25)  # next segment after skipped surface
        cache.insert_eligible_hit()  # B
        cache.nee_hook(2.0)
        self.assertValues(cache, [5.0 * 0.8 * 0.5 + 2.0 * 0.8 * 0.5 * 0.25, 2.0])

    def test_emission_before_insert_is_not_counted_as_current_emission(self) -> None:
        cache = ContractCache()
        cache.insert_eligible_hit()  # A
        cache.accumulate_throughput(0.5)
        cache.accumulate_radiance(6.0)  # B's explicit emission/MIS
        cache.insert_eligible_hit()  # B receives zero local emission
        cache.nee_hook(1.0)
        self.assertValues(cache, [3.5, 1.0])

    def test_finite_depth_retains_newest_entries(self) -> None:
        cache = ContractCache(propagation_depth=2)
        cache.insert_eligible_hit()  # A
        cache.accumulate_throughput(0.5)
        cache.insert_eligible_hit()  # B
        cache.accumulate_throughput(0.25)
        cache.insert_eligible_hit()  # C drops A at finite propagation depth
        cache.accumulate_radiance(8.0)  # C terminal contribution
        self.assertValues(cache, [2.0, 8.0])


if __name__ == "__main__":
    unittest.main()
