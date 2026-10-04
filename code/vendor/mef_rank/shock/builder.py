"""Frozen five-driver causal shock construction for MEF-Rank v3.3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..fec.builder import FECPanel
from ..signal_builder.full_builder import EPSILON, SignalPanel


ArrayF = npt.NDArray[np.float64]
ArrayB = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ShockScaler:
    fit_date_max: str
    center: ArrayF
    scale: ArrayF
    supported: ArrayB
    scaler_hash: str
    fit_date_min: str | None = None


@dataclass(frozen=True)
class ShockRawPanel:
    """Unscaled causal shock coordinates safe for annual PIT sharding."""

    dates: tuple[str, ...]
    driver: ArrayF
    driver_mask: ArrayB
    innovation: ArrayF
    causal_value: ArrayF
    causal_mask: ArrayB

    def validate(self) -> None:
        shape = (len(self.dates), 5)
        for value, name in (
            (self.driver, "driver"),
            (self.innovation, "innovation"),
            (self.causal_value, "causal_value"),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite [T,5]")
        for value, name in (
            (self.driver_mask, "driver_mask"),
            (self.causal_mask, "causal_mask"),
        ):
            if value.shape != shape or value.dtype != np.bool_:
                raise ValueError(f"{name} must be bool [T,5]")
        if np.any(self.causal_value[~self.causal_mask] != 0.0):
            raise ValueError("invalid causal shock placeholders must be zero")


@dataclass(frozen=True)
class ShockPanel:
    dates: tuple[str, ...]
    driver: ArrayF
    driver_mask: ArrayB
    innovation: ArrayF
    causal_value: ArrayF
    causal_mask: ArrayB
    network_value: ArrayF
    network_mask: ArrayB
    scaler_hash: str

    def validate(self) -> None:
        shape = (len(self.dates), 5)
        for value, name in (
            (self.driver, "driver"),
            (self.innovation, "innovation"),
            (self.causal_value, "causal_value"),
            (self.network_value, "network_value"),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite [T,5]")
        for value, name in (
            (self.driver_mask, "driver_mask"),
            (self.causal_mask, "causal_mask"),
            (self.network_mask, "network_mask"),
        ):
            if value.shape != shape or value.dtype != np.bool_:
                raise ValueError(f"{name} must be bool [T,5]")
        if np.any(self.network_value[~self.network_mask] != 0.0):
            raise ValueError("invalid network shock placeholders must be zero")


def _drivers(panel: SignalPanel, fec: FECPanel) -> tuple[ArrayF, ArrayB]:
    value = np.zeros((len(panel.dates), 5), dtype=np.float64)
    mask = np.zeros_like(value, dtype=bool)
    value[:, 0] = panel.market_return
    mask[:, 0] = panel.market_return_mask
    value[:, 1] = fec.common_raw[:, 2]
    mask[:, 1] = fec.common_mask[:, 2]
    value[:, 2] = fec.common_raw[:, 8]
    mask[:, 2] = fec.common_mask[:, 8]
    value[:, 3] = np.asarray(fec.drivers["log_total_amount"], dtype=np.float64)
    mask[:, 3] = np.asarray(fec.drivers["log_total_amount_mask"], dtype=bool)
    value[:, 4] = fec.common_raw[:, 10]
    mask[:, 4] = fec.common_mask[:, 10]
    value[~mask] = 0.0
    return value, mask


def _causal_transform(driver: ArrayF, driver_mask: ArrayB) -> tuple[ArrayF, ArrayF, ArrayB]:
    innovation = np.zeros_like(driver)
    shock = np.zeros_like(driver)
    shock_mask = np.zeros_like(driver_mask)
    alpha = 2.0 / 21.0
    for coordinate in range(driver.shape[1]):
        ewma: float | None = None
        prior_innovations: list[float] = []
        for time_index in range(driver.shape[0]):
            if not driver_mask[time_index, coordinate]:
                continue
            current = float(driver[time_index, coordinate])
            if ewma is None:
                ewma = current
                continue
            current_innovation = current - ewma
            innovation[time_index, coordinate] = current_innovation
            reference = np.asarray(prior_innovations[-60:], dtype=np.float64)
            if len(reference) >= 20:
                median = np.median(reference)
                scale = 1.4826 * np.median(np.abs(reference - median))
                if scale > EPSILON:
                    shock[time_index, coordinate] = current_innovation / scale
                    shock_mask[time_index, coordinate] = True
            prior_innovations.append(current_innovation)
            ewma = alpha * current + (1.0 - alpha) * ewma
    return innovation, shock, shock_mask


def fit_shock_scaler(
    dates: tuple[str, ...],
    causal_value: ArrayF,
    causal_mask: ArrayB,
    *,
    fit_date_max: str,
    fit_date_min: str | None = None,
) -> ShockScaler:
    date_values = np.asarray(dates)
    rows = date_values <= str(fit_date_max)
    if fit_date_min is not None:
        if str(fit_date_min) > str(fit_date_max):
            raise ValueError("shock scaler fit interval is reversed")
        rows &= date_values >= str(fit_date_min)
    if not rows.any():
        raise ValueError("shock scaler fit interval is empty")
    center = np.zeros(5, dtype=np.float64)
    scale = np.ones(5, dtype=np.float64)
    supported = np.zeros(5, dtype=bool)
    for coordinate in range(5):
        selected = rows & causal_mask[:, coordinate]
        if not selected.any():
            continue
        data = causal_value[selected, coordinate]
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        center[coordinate] = median
        scale[coordinate] = max(1.4826 * mad, EPSILON)
        supported[coordinate] = True
    payload = {
        "fit_date_max": str(fit_date_max),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "supported": supported.tolist(),
    }
    if fit_date_min is not None:
        payload["fit_date_min"] = str(fit_date_min)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return ShockScaler(
        str(fit_date_max),
        center,
        scale,
        supported,
        digest,
        None if fit_date_min is None else str(fit_date_min),
    )


def build_shock_raw_panel(panel: SignalPanel, fec: FECPanel) -> ShockRawPanel:
    """Build causal shock drivers without fitting any date-global scaler."""

    driver, driver_mask = _drivers(panel, fec)
    innovation, causal, causal_mask = _causal_transform(driver, driver_mask)
    result = ShockRawPanel(
        panel.dates,
        driver,
        driver_mask,
        innovation,
        causal,
        causal_mask,
    )
    result.validate()
    return result


def build_shock_panel_from_raw(
    raw: ShockRawPanel, scaler: ShockScaler
) -> ShockPanel:
    """Apply one checkpoint-local scaler to an immutable raw shock panel."""

    raw.validate()
    network_mask = raw.causal_mask & scaler.supported[None, :]
    network = (raw.causal_value - scaler.center) / scaler.scale
    network[~network_mask] = 0.0
    result = ShockPanel(
        dates=raw.dates,
        driver=raw.driver,
        driver_mask=raw.driver_mask,
        innovation=raw.innovation,
        causal_value=raw.causal_value,
        causal_mask=raw.causal_mask,
        network_value=network,
        network_mask=network_mask,
        scaler_hash=scaler.scaler_hash,
    )
    result.validate()
    return result


def build_shock_panel(
    panel: SignalPanel,
    fec: FECPanel,
    *,
    fit_date_max: str,
) -> ShockPanel:
    raw = build_shock_raw_panel(panel, fec)
    scaler = fit_shock_scaler(
        panel.dates,
        raw.causal_value,
        raw.causal_mask,
        fit_date_max=fit_date_max,
    )
    return build_shock_panel_from_raw(raw, scaler)
