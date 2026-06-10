"""Numerical answer normalisation and accuracy metrics for QA evaluation."""

import re


def normalise_answer(answer: str | float) -> float | None:
    """Normalise an answer string to a float, handling percentages and edge cases."""
    if isinstance(answer, (int, float)):
        return float(answer)
    text = str(answer).strip().replace(",", "")
    is_percent = "%" in text
    text = text.replace("%", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group())
    if is_percent:
        value = value / 100.0
    return value


def is_correct(
    predicted: str | float,
    gold: float | str,
    rel_tolerance: float = 0.05,
    abs_tolerance: float = 1e-3,
) -> bool:
    """Return True if predicted matches gold within tolerance.

    Uses relative tolerance (5% by default) for normal numbers and
    absolute tolerance (1e-3 by default) as fallback for very small
    values where the relative bound is too tight.
    """
    pred = normalise_answer(predicted)
    gold_f = normalise_answer(gold)
    if pred is None or gold_f is None:
        return False
    if gold_f == 0:
        return abs(pred) <= abs_tolerance
    # Unit scale: gold=0.88, pred=880000000 → divide by 1B; OR gold=20M, pred=20 → multiply by 1M
    scaled = False
    for scale in (1_000_000_000, 1_000_000, 1_000):
        if abs(pred / scale - gold_f) / max(abs(gold_f), 1e-9) <= rel_tolerance:
            pred = pred / scale
            scaled = True
            break
        if abs(pred * scale - gold_f) / max(abs(gold_f), 1e-9) <= rel_tolerance:
            pred = pred * scale
            scaled = True
            break
    # Percent / decimal rescalings only run when the scale loop didn't already
    # commit a rescaling: prevents two independent transforms from compounding
    # on the same pred and producing spurious matches.
    if not scaled:
        # Percent vs decimal: gold=0.14, predicted=14.1 → divide by 100
        if abs(gold_f) < 1 and abs(pred) > 1:
            pred = pred / 100.0
        # Reverse: gold=51.2 (percent format), pred=0.504 (decimal) → multiply by 100
        elif abs(gold_f) > 1 and abs(pred) < 1:
            if abs(pred * 100 - gold_f) / max(abs(gold_f), 1e-9) <= rel_tolerance:
                pred = pred * 100.0
        # Both small but 100x off: gold=0.35 (percent points), pred=0.0035 (decimal): record 84 Q3
        elif abs(gold_f) < 1 and 0 < abs(pred) < 1:
            ratio = abs(gold_f / pred)
            if 50 < ratio < 200 and abs(pred * 100 - gold_f) / max(abs(gold_f), 1e-9) <= rel_tolerance:
                pred = pred * 100.0
    rel_ok = abs(pred - gold_f) / max(abs(gold_f), 1e-9) <= rel_tolerance
    abs_ok = abs(pred - gold_f) <= abs_tolerance
    # Inverse ratio: accept X/Y vs Y/X ONLY when both values are mild
    # inverses, i.e. both in [0.5, 2.0]. This catches genuine ratio swaps
    # (canonical 0.5 vs 2.0) while rejecting accidental inverse matches
    # like 0.25 vs 4 or 0.07 vs 14 where the answers aren't semantically
    # equivalent in the question's context (caught live by the LLM judge).
    inv_ok = (
        0.5 <= abs(pred) <= 2.0
        and 0.5 <= abs(gold_f) <= 2.0
        and abs(1 / pred - gold_f) / max(abs(gold_f), 1e-9) <= rel_tolerance
    )
    # Sign flip: balance-sheet items (losses, deficits) stored negative but
    # gold expects positive magnitude. Bounded to abs < 10000 to avoid
    # accepting large-value swaps. Pass --judge for semantic context.
    sign_ok = (
        pred * gold_f < 0
        and abs(pred) < 10000
        and abs(gold_f) < 10000
        and abs(abs(pred) - abs(gold_f)) / max(abs(gold_f), 1e-9) <= rel_tolerance
    )
    return rel_ok or abs_ok or inv_ok or sign_ok
