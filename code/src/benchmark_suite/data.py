"""Leakage-safe dense panel loader for the GEM 0/1 benchmark contract."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .contracts import (
    FEATURE_DIMENSION,
    FinancialBatch,
    GROUP_DIMENSIONS,
    GROUP_ORDER,
)


LOOKBACK = 60
HORIZON = 5


def schema_columns(path) -> list[str]:
    return path if isinstance(path, list) else pq.ParquetFile(path).schema_arrow.names


def financial_feature_columns(path: Path) -> tuple[list[str], list[str]]:
    columns = schema_columns(path)
    values: list[str] = []
    masks: list[str] = []
    for group, expected in zip(GROUP_ORDER, GROUP_DIMENSIONS, strict=True):
        value_prefix = f"signal_normalized__{group}__"
        mask_prefix = f"signal_normalized_available__{group}__"
        group_values = [name for name in columns if name.startswith(value_prefix)]
        group_masks = [name for name in columns if name.startswith(mask_prefix)]
        if len(group_values) != expected or len(group_masks) != expected:
            raise ValueError(
                f"{path}: {group} expected {expected} values/masks, "
                f"got {len(group_values)}/{len(group_masks)}"
            )
        left = [name.removeprefix(value_prefix) for name in group_values]
        right = [name.removeprefix(mask_prefix) for name in group_masks]
        if left != right:
            raise ValueError(f"{path}: {group} value/mask coordinate order differs")
        values.extend(group_values)
        masks.extend(group_masks)
    if len(values) != FEATURE_DIMENSION:
        raise AssertionError("internal feature dimension mismatch")
    return values, masks


def joint_market_columns(path: Path) -> tuple[list[str], list[str]]:
    columns = schema_columns(path)
    values = [name for name in columns if name.startswith("msjd_H5__")]
    masks = [name for name in columns if name.startswith("msjd_H5_available__")]
    if len(values) != 306 or len(masks) != 306:
        raise ValueError(f"{path}: expected 306 H5 J coordinates and masks")
    left = [name.removeprefix("msjd_H5__") for name in values]
    right = [name.removeprefix("msjd_H5_available__") for name in masks]
    if left != right:
        raise ValueError(f"{path}: J value/mask coordinate order differs")
    return values, masks


class FinancialPanel:
    """Dense arrays indexed by target date and stock; labels never enter x."""

    def __init__(
        self,
        *,
        name: str,
        dates: list[str],
        codes: list[str],
        values: np.ndarray,
        feature_masks: np.ndarray,
        stock_present: np.ndarray,
        signal_universe: np.ndarray,
        label_rank: np.ndarray,
        label_return: np.ndarray,
        label_valid: np.ndarray,
        label_maturity: np.ndarray,
        j_values: np.ndarray,
        j_masks: np.ndarray,
    ) -> None:
        self.name = name
        self.dates = dates
        self.codes = codes
        self.values = values
        self.feature_masks = feature_masks
        self.stock_present = stock_present
        self.signal_universe = signal_universe
        self.label_rank = label_rank
        self.label_return = label_return
        self.label_valid = label_valid
        self.label_maturity = label_maturity
        self.j_values = j_values
        self.j_masks = j_masks
        self.date_index = {value: index for index, value in enumerate(dates)}
        self._validate()

    def _validate(self) -> None:
        shape = (len(self.dates), len(self.codes))
        if self.values.shape != (*shape, FEATURE_DIMENSION):
            raise ValueError("panel feature shape mismatch")
        if self.feature_masks.shape != self.values.shape:
            raise ValueError("panel feature-mask shape mismatch")
        for value in (
            self.stock_present,
            self.signal_universe,
            self.label_rank,
            self.label_return,
            self.label_valid,
            self.label_maturity,
        ):
            if value.shape != shape:
                raise ValueError("panel stock matrix shape mismatch")
        if self.j_values.shape != (len(self.dates), 306) or self.j_masks.shape != self.j_values.shape:
            raise ValueError("panel J matrix shape mismatch")
        if not np.isfinite(self.values).all() or np.any(self.values[~self.feature_masks] != 0):
            raise ValueError("panel features violate finite zero-mask contract")
        if not np.isfinite(self.j_values).all() or np.any(self.j_values[~self.j_masks] != 0):
            raise ValueError("panel J violates finite zero-mask contract")
        if np.any(self.signal_universe & ~self.stock_present):
            raise ValueError("signal universe must be a subset of stock-present")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "date_min": self.dates[0],
            "date_max": self.dates[-1],
            "date_count": len(self.dates),
            "stock_axis_count": len(self.codes),
            "signal_universe_rows": int(self.signal_universe.sum()),
            "valid_h5_rows": int(self.label_valid.sum()),
            "array_bytes": int(
                self.values.nbytes
                + self.feature_masks.nbytes
                + self.stock_present.nbytes
                + self.signal_universe.nbytes
                + self.label_rank.nbytes
                + self.label_return.nbytes
                + self.label_valid.nbytes
                + self.label_maturity.nbytes
                + self.j_values.nbytes
                + self.j_masks.nbytes
            ),
        }

    def eligible_indices(
        self,
        date_index: int,
        *,
        require_label: bool,
        maturity_cutoff: str | None = None,
    ) -> np.ndarray:
        valid = self.signal_universe[date_index].copy()
        if require_label:
            valid &= self.label_valid[date_index]
        if maturity_cutoff is not None:
            maturity = self.label_maturity[date_index]
            valid &= (maturity != "") & (maturity <= maturity_cutoff)
        return np.flatnonzero(valid)

    def make_batch(
        self,
        date_index: int,
        stock_indices: np.ndarray | list[int],
        device: torch.device,
        *,
        maturity_cutoff: str | None = None,
    ) -> FinancialBatch:
        if date_index < LOOKBACK - 1:
            raise ValueError("insufficient lookback")
        stocks = np.asarray(stock_indices, dtype=np.int64)
        if stocks.ndim != 1 or not len(stocks):
            raise ValueError("stock_indices must be a nonempty vector")
        selection = slice(date_index - LOOKBACK + 1, date_index + 1)
        values = np.transpose(self.values[selection][:, stocks], (1, 0, 2)).copy()
        feature_mask = np.transpose(
            self.feature_masks[selection][:, stocks], (1, 0, 2)
        ).copy()
        history = np.transpose(self.stock_present[selection][:, stocks], (1, 0)).copy()
        values[~feature_mask] = 0.0
        valid = self.label_valid[date_index, stocks] & self.signal_universe[date_index, stocks]
        if maturity_cutoff is not None:
            maturity = self.label_maturity[date_index, stocks]
            valid &= (maturity != "") & (maturity <= maturity_cutoff)
        batch = FinancialBatch(
            x=torch.from_numpy(values[None]).to(device),
            feature_mask=torch.from_numpy(feature_mask[None]).to(device),
            history_mask=torch.from_numpy(history[None]).to(device),
            stock_mask=torch.from_numpy(self.signal_universe[date_index, stocks][None]).to(device),
            label_rank=torch.from_numpy(self.label_rank[date_index, stocks][None]).to(device),
            label_return=torch.from_numpy(self.label_return[date_index, stocks][None]).to(device),
            label_valid=torch.from_numpy(valid[None]).to(device),
        )
        batch.validate(lookback=LOOKBACK)
        return batch

    def joint_state(self, date_index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.j_values[date_index].copy(), self.j_masks[date_index].copy()


def _axes(root: Path, years: Iterable[int], read_frame=pd.read_parquet) -> tuple[list[str], list[str]]:
    dates: set[str] = set()
    codes: set[str] = set()
    for year in years:
        frame = read_frame(
            root / f"year={year}" / "stock_daily.parquet",
            columns=["feature_date", "ts_code"],
        )
        dates.update(frame["feature_date"].astype(str))
        codes.update(frame["ts_code"].astype(str))
    return sorted(dates), sorted(codes)


def load_panel(*, name: str, root: Path, years: list[int], reader=None, view=None) -> FinancialPanel:
    if not years:
        raise ValueError("years cannot be empty")
    def read_frame(path, columns=None):
        if reader is None:
            return pd.read_parquet(path, columns=columns)
        year = int(path.parent.name.split('=')[1])
        return reader.read_frame(view, year, path.name.split('_')[0], columns)
    example_stock = root / f"year={years[0]}" / "stock_daily.parquet"
    example_market = root / f"year={years[0]}" / "market_daily.parquet"
    feature_columns, feature_mask_columns = financial_feature_columns(reader.read_table(view, years[0], 'stock').schema.names if reader else example_stock)
    j_columns, j_mask_columns = joint_market_columns(reader.read_table(view, years[0], 'market').schema.names if reader else example_market)
    dates, codes = _axes(root, years, read_frame)
    date_lookup = {value: index for index, value in enumerate(dates)}
    code_lookup = {value: index for index, value in enumerate(codes)}
    shape = (len(dates), len(codes))
    values = np.zeros((*shape, FEATURE_DIMENSION), dtype=np.float32)
    masks = np.zeros((*shape, FEATURE_DIMENSION), dtype=bool)
    present = np.zeros(shape, dtype=bool)
    universe = np.zeros(shape, dtype=bool)
    rank = np.zeros(shape, dtype=np.float32)
    returns = np.zeros(shape, dtype=np.float32)
    label_valid = np.zeros(shape, dtype=bool)
    maturity = np.full(shape, "", dtype="U8")
    j_values = np.zeros((len(dates), 306), dtype=np.float32)
    j_masks = np.zeros((len(dates), 306), dtype=bool)
    stock_columns = [
        "feature_date",
        "ts_code",
        "tensor__stock_present",
        "tensor__signal_universe",
        "label_H5__primary_rank",
        "label_H5__primary_return",
        "label_H5__label_valid",
        "label_H5__label_matured_date",
        *feature_columns,
        *feature_mask_columns,
    ]
    for year in years:
        stock_path = root / f"year={year}" / "stock_daily.parquet"
        frame = read_frame(stock_path, columns=stock_columns)
        di = frame["feature_date"].astype(str).map(date_lookup).to_numpy(np.int64)
        ci = frame["ts_code"].astype(str).map(code_lookup).to_numpy(np.int64)
        flat = di * len(codes) + ci
        if len(np.unique(flat)) != len(flat):
            raise ValueError(f"{stock_path}: duplicate date/code keys")
        present.reshape(-1)[flat] = frame["tensor__stock_present"].fillna(False).astype(bool)
        universe.reshape(-1)[flat] = frame["tensor__signal_universe"].fillna(False).astype(bool)
        rank.reshape(-1)[flat] = np.nan_to_num(
            pd.to_numeric(frame["label_H5__primary_rank"], errors="coerce").to_numpy(np.float32)
        )
        returns.reshape(-1)[flat] = np.nan_to_num(
            pd.to_numeric(frame["label_H5__primary_return"], errors="coerce").to_numpy(np.float32)
        )
        label_valid.reshape(-1)[flat] = (
            frame["label_H5__label_valid"].astype("boolean").to_numpy(dtype=bool, na_value=False)
        )
        maturity.reshape(-1)[flat] = (
            frame["label_H5__label_matured_date"]
            .fillna("")
            .astype(str)
            .str.replace("-", "", regex=False)
            .str[:8]
            .to_numpy(dtype="U8")
        )
        raw_values = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        raw_masks = (
            frame[feature_mask_columns]
            .astype("boolean")
            .to_numpy(dtype=bool, na_value=False)
        )
        raw_values[~raw_masks] = 0.0
        values.reshape(-1, FEATURE_DIMENSION)[flat] = np.nan_to_num(raw_values)
        masks.reshape(-1, FEATURE_DIMENSION)[flat] = raw_masks

        market_path = root / f"year={year}" / "market_daily.parquet"
        market = read_frame(
            market_path, columns=["feature_date", *j_columns, *j_mask_columns]
        )
        if market["feature_date"].astype(str).duplicated().any():
            raise ValueError(f"{market_path}: duplicate dates")
        mi = market["feature_date"].astype(str).map(date_lookup).to_numpy(np.int64)
        raw_j = market[j_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        raw_j_mask = (
            market[j_mask_columns].astype("boolean").to_numpy(dtype=bool, na_value=False)
        )
        raw_j[~raw_j_mask] = 0.0
        j_values[mi] = np.nan_to_num(raw_j)
        j_masks[mi] = raw_j_mask
    return FinancialPanel(
        name=name,
        dates=dates,
        codes=codes,
        values=values,
        feature_masks=masks,
        stock_present=present,
        signal_universe=universe,
        label_rank=rank,
        label_return=returns,
        label_valid=label_valid,
        label_maturity=maturity,
        j_values=j_values,
        j_masks=j_masks,
    )
