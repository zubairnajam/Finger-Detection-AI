"""Rolling-window vote stabiliser."""

from collections import deque

from src.config import MIN_CONFIDENCE, MIN_STABLE_VOTES
from src.prediction import resolve_stable_prediction


def test_empty_window_has_no_label():
    result = resolve_stable_prediction(deque())
    assert result.label is None
    assert result.votes == 0
    assert result.confidence == 0.0


def test_majority_with_enough_votes_and_confidence_is_stable():
    window = deque([(3, 0.9)] * MIN_STABLE_VOTES + [(2, 0.8)])
    result = resolve_stable_prediction(window)
    assert result.label == 3
    assert result.votes == MIN_STABLE_VOTES
    assert abs(result.confidence - 0.9) < 1e-9


def test_too_few_votes_is_not_stable():
    window = deque([(3, 0.99)] * (MIN_STABLE_VOTES - 1))
    result = resolve_stable_prediction(window)
    assert result.label is None
    assert result.votes == MIN_STABLE_VOTES - 1


def test_low_confidence_is_not_stable():
    low = MIN_CONFIDENCE - 0.1
    window = deque([(4, low)] * 8)
    result = resolve_stable_prediction(window)
    assert result.label is None
    assert abs(result.confidence - low) < 1e-9


def test_confidence_is_averaged_over_winning_label_only():
    window = deque([(1, 0.6), (1, 0.8), (1, 0.7), (1, 0.9), (5, 0.1), (5, 0.1)])
    result = resolve_stable_prediction(window)
    assert result.label == 1
    assert abs(result.confidence - 0.75) < 1e-9


def test_thresholds_can_be_overridden():
    window = deque([(2, 0.7)] * 5)
    assert resolve_stable_prediction(window).label == 2
    strict = resolve_stable_prediction(window, min_votes=6, min_confidence=0.65)
    assert strict.label is None
    lenient = resolve_stable_prediction(window, min_votes=5, min_confidence=0.65)
    assert lenient.label == 2
