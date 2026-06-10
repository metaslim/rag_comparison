"""Unit tests for lib.evaluation.metrics: normalise_answer and is_correct."""

import pytest

from src.lib.evaluation.base import is_correct, normalise_answer


def test_normalise_plain_float() -> None:
    assert normalise_answer(206588.0) == 206588.0


def test_normalise_integer() -> None:
    assert normalise_answer(25587) == 25587.0


def test_normalise_string_number() -> None:
    assert normalise_answer("181001") == 181001.0


def test_normalise_percentage_string() -> None:
    assert normalise_answer("14.1%") == pytest.approx(0.141)


def test_normalise_negative() -> None:
    assert normalise_answer("-6.3") == -6.3


def test_normalise_with_commas() -> None:
    assert normalise_answer("1,000,000") == 1000000.0


def test_normalise_unknown_returns_none() -> None:
    assert normalise_answer("unknown") is None


def test_normalise_empty_returns_none() -> None:
    assert normalise_answer("") is None


def test_correct_within_tolerance() -> None:
    assert is_correct("0.1414", 0.14136)


def test_correct_exact_match() -> None:
    assert is_correct("206588", 206588.0)


def test_wrong_answer() -> None:
    # With 5% relative tolerance, 200000 vs 206588 is ~3.2% off and passes;
    # use a clearly-outside-tolerance example.
    assert not is_correct("180000", 206588.0)


def test_percentage_normalisation() -> None:
    assert is_correct("14.1%", 0.14136)


def test_percent_large_to_small_normalisation() -> None:
    assert is_correct("14.136", 0.14136)


def test_gold_zero_uses_absolute_tolerance() -> None:
    # abs_tolerance is 1e-3: values within that count as "zero"
    assert is_correct("0.0005", 0.0)
    assert not is_correct("0.5", 0.0)


def test_negative_values() -> None:
    assert is_correct("-6.3", -6.3)


# Normalisation path coverage: each path corresponds to a real failure
# encountered during evaluation. Without these tests, an evaluator refactor
# could silently regress the records that depend on them.

def test_unit_scale_billion() -> None:
    # Record 34: gold=0.88 (in billions), pred=880000000 (raw). Divide by 1B.
    assert is_correct("880000000", 0.88)


def test_unit_scale_million() -> None:
    # gold=0.5 (in millions), pred=500000 (raw). Divide by 1M.
    assert is_correct("500000", 0.5)


def test_multiply_by_scale() -> None:
    # Record 46: gold=20743588 (raw value), pred=20.74 (in millions).
    # Multiply pred by 1M to reach gold.
    assert is_correct("20.74", 20743588.0)


def test_reverse_percent_large_gold() -> None:
    # Record 62 Q1, record 75: gold=51.2 (percent format), pred=0.504 (decimal).
    # Multiply pred by 100, check within tolerance.
    assert is_correct("0.504", 51.2)


def test_reverse_percent_both_small_100x_off() -> None:
    # Record 84 Q3: gold=0.35 (percent points, sub-1), pred=0.0035 (decimal).
    # Both values are < 1, but they're 100x apart so the both-small rule fires.
    assert is_correct("0.0035", 0.35)


def test_reverse_percent_both_small_does_not_fire_when_close() -> None:
    # If pred and gold are both small and roughly equal, the 100x branch must
    # not be triggered (would otherwise create false positives).
    assert is_correct("0.35", 0.35)
    assert not is_correct("0.35", 0.50)  # ~30% off, still wrong


def test_sign_flip_accepted_for_balance_sheet_items() -> None:
    # Sign conventions (balance-sheet losses negative vs positive magnitude)
    # are domain-semantic -- the code evaluator no longer handles them.
    # Pass --judge to catch these via the LLM judge instead.
    assert is_correct("-13.2", 13.2)  # balance-sheet sign convention


def test_inverse_ratio_xy_vs_yx() -> None:
    # Gold ratio is X/Y but model returned Y/X (or vice versa).
    # 1/pred ~= gold so the inverse path catches it. Both values are in
    # the "ratio-like" range (0.1, 10) so the tightened rule accepts.
    assert is_correct("0.5", 2.0)


def test_inverse_ratio_rejects_extreme_inverses() -> None:
    """0.25 vs 4 and 0.0714 vs 14 mathematically invert but aren't semantically equal.

    The old inverse-ratio path accepted these. The tightened version requires
    both values to be in (0.1, 10) so accidental inverse-matches on out-of-range
    values are rejected (caught live by the LLM judge during the 10-record A/B).
    """
    # gold=4.0, pred=0.25 - 1/4 = 0.25 mathematically inverts, but 0.25 != 4
    assert not is_correct("0.25", 4.0)
    # gold=14.0, pred=0.07143 - 1/14 ≈ 0.07143, also rejected
    assert not is_correct("0.07142857", 14.0)
    # gold=0.05, pred=20 - 1/20 = 0.05, also rejected (gold outside range)
    assert not is_correct("20", 0.05)


def test_normalisation_paths_do_not_create_false_positives() -> None:
    # Unrelated values should still fail. This guards against an overly-lenient
    # chain of normalisations passing things it shouldn't.
    assert not is_correct("1234", 5678.0)
    # Sign flip helps only when magnitudes match. Different magnitude + opposite
    # sign should not pass.
    assert not is_correct("-50", 100.0)


def test_is_correct_returns_false_when_predicted_is_unparseable() -> None:
    # An agent output that can't be coerced to a number (e.g. "error" / "n/a")
    # must return False without raising.
    assert not is_correct("not a number", 100.0)
    assert not is_correct("error", 100.0)


def test_yesno_predicted_answer_not_scored_by_code_evaluator() -> None:
    """'yes'/'no' predictions normalise to None: the code evaluator always
    returns False for them. This is a documented limitation: yes/no questions
    must be scored by the LLM judge (--judge flag) rather than the code path.
    """
    assert normalise_answer("yes") is None
    assert normalise_answer("no") is None
    # is_correct returns False for these -- wrong verdict, but expected.
    # The --judge evaluator handles yes/no correctly.
    assert not is_correct("yes", 1.0)
    assert not is_correct("no", 0.0)
