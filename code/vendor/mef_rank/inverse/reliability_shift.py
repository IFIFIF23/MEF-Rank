"""Float64 bounded inversion of mean sigmoid reliability shifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class InverseResult:
    delta_beta: npt.NDArray[np.float64]
    target_shift_requested: npt.NDArray[np.float64]
    target_shift_projected: npt.NDArray[np.float64]
    target_shift_achieved: npt.NDArray[np.float64]
    feasible_low: npt.NDArray[np.float64]
    feasible_high: npt.NDArray[np.float64]
    projection_error: npt.NDArray[np.float64]
    solver_residual: npt.NDArray[np.float64]
    mean_sigmoid_derivative_at_solution: npt.NDArray[np.float64]
    per_group_clipped: npt.NDArray[np.bool_]
    iterations_used: npt.NDArray[np.int64]
    converged: npt.NDArray[np.bool_]
    group_reason: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "delta_beta",
            "target_shift_requested",
            "target_shift_projected",
            "target_shift_achieved",
            "feasible_low",
            "feasible_high",
            "projection_error",
            "solver_residual",
            "mean_sigmoid_derivative_at_solution",
            "per_group_clipped",
            "iterations_used",
            "converged",
        ):
            owned = np.array(getattr(self, name), copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


def _sigmoid(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Base logits are bounded by contract and delta is at most 0.5, but this
    # stable branch keeps the primitive well-defined for adversarial fixtures.
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def invert_reliability_shift(
    base_logits: npt.ArrayLike,
    valid_mask: npt.ArrayLike,
    target_shift: npt.ArrayLike,
    *,
    action_cap: float,
    iterations: int = 24,
    tolerance: float = 1.0e-5,
) -> InverseResult:
    """Invert seven groupwise average reliability shifts by fixed bisection.

    ``base_logits`` and ``valid_mask`` have shape ``[N,7]``.  A target outside
    the reachable interval is projected to the nearest endpoint.  A group with
    no current support is disabled exactly, without affecting other groups.
    """

    logits = np.asarray(base_logits, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    requested = np.asarray(target_shift, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 7:
        raise ValueError("base_logits must have shape [N,7]")
    if mask.shape != logits.shape:
        raise ValueError("valid_mask must match base_logits")
    if requested.shape != (7,):
        raise ValueError("target_shift must have shape [7]")
    if not np.isfinite(logits).all() or not np.isfinite(requested).all():
        raise ValueError("logits and target_shift must be finite")
    if action_cap <= 0 or iterations <= 0 or tolerance <= 0:
        raise ValueError("action_cap, iterations and tolerance must be positive")

    delta = np.zeros(7, dtype=np.float64)
    projected = np.zeros(7, dtype=np.float64)
    feasible_low = np.zeros(7, dtype=np.float64)
    feasible_high = np.zeros(7, dtype=np.float64)
    achieved_shift = np.zeros(7, dtype=np.float64)
    solver_residual = np.zeros(7, dtype=np.float64)
    derivative = np.zeros(7, dtype=np.float64)
    clipped = np.zeros(7, dtype=bool)
    iterations_used = np.zeros(7, dtype=np.int64)
    converged = np.zeros(7, dtype=bool)
    reasons: list[str] = []

    for group in range(7):
        supported_logits = logits[mask[:, group], group]
        if supported_logits.size == 0:
            reasons.append("GROUP_NO_SUPPORT")
            continue

        base_reliability = _sigmoid(supported_logits)

        def mean_shift(offset: float) -> float:
            return float(np.mean(_sigmoid(supported_logits + offset) - base_reliability))

        low_value = mean_shift(-action_cap)
        high_value = mean_shift(action_cap)
        feasible_low[group] = low_value
        feasible_high[group] = high_value
        projected[group] = np.clip(requested[group], low_value, high_value)
        clipped[group] = projected[group] != requested[group]

        if abs(projected[group]) <= tolerance:
            achieved_shift[group] = 0.0
            solver_residual[group] = -projected[group]
            reliability = base_reliability
            derivative[group] = float(np.mean(reliability * (1.0 - reliability)))
            converged[group] = abs(solver_residual[group]) <= tolerance
            reasons.append("GROUP_ACTION_ZEROED")
            continue

        lower = -action_cap
        upper = action_cap
        for _ in range(iterations):
            middle = (lower + upper) / 2.0
            if mean_shift(middle) < projected[group]:
                lower = middle
            else:
                upper = middle
        iterations_used[group] = iterations
        delta[group] = (lower + upper) / 2.0
        achieved = mean_shift(delta[group])
        achieved_shift[group] = achieved
        solver_residual[group] = achieved - projected[group]
        reliability = _sigmoid(supported_logits + delta[group])
        derivative[group] = float(np.mean(reliability * (1.0 - reliability)))
        converged[group] = abs(solver_residual[group]) <= tolerance
        if not converged[group]:
            delta[group] = 0.0
            achieved_shift[group] = 0.0
            solver_residual[group] = -projected[group]
            reliability = base_reliability
            derivative[group] = float(np.mean(reliability * (1.0 - reliability)))
            reasons.append("GROUP_INVERSE_ILL_CONDITIONED")
        elif clipped[group]:
            reasons.append("GROUP_TARGET_CLIPPED")
        elif delta[group] == 0.0:
            reasons.append("GROUP_ACTION_ZEROED")
        else:
            reasons.append("APPLY")

    return InverseResult(
        delta_beta=delta,
        target_shift_requested=requested.copy(),
        target_shift_projected=projected,
        target_shift_achieved=achieved_shift,
        feasible_low=feasible_low,
        feasible_high=feasible_high,
        projection_error=projected - requested,
        solver_residual=solver_residual,
        mean_sigmoid_derivative_at_solution=derivative,
        per_group_clipped=clipped,
        iterations_used=iterations_used,
        converged=converged,
        group_reason=tuple(reasons),
    )
