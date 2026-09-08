from __future__ import annotations

import math

from aqua_assistant.inference.entropy import entropy, normalized_entropy


def test_entropy_of_certain_distribution_is_zero():
    assert entropy({"a": 1.0, "b": 0.0}) == 0.0


def test_entropy_of_uniform_four_way_is_two_bits():
    dist = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert math.isclose(entropy(dist), 2.0, rel_tol=1e-9)


def test_entropy_of_uniform_two_way_is_one_bit():
    dist = {"a": 0.5, "b": 0.5}
    assert math.isclose(entropy(dist), 1.0, rel_tol=1e-9)


def test_normalized_entropy_is_one_for_uniform_distribution():
    dist = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert math.isclose(normalized_entropy(dist), 1.0, rel_tol=1e-9)


def test_normalized_entropy_is_zero_for_single_or_empty_candidate():
    assert normalized_entropy({"a": 1.0}) == 0.0
    assert normalized_entropy({}) == 0.0


def test_normalized_entropy_between_zero_and_one_for_skewed_distribution():
    dist = {"a": 0.7, "b": 0.2, "c": 0.1}
    value = normalized_entropy(dist)
    assert 0.0 < value < 1.0
