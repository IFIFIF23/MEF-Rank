"""Pinned official-source loading and invocation without modifying upstream code."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
LOCK_PATH = WORKSPACE / "third_party" / "model_sources.lock.yaml"
PROFILE_PATH = WORKSPACE / "config" / "model_profiles.yaml"
SOURCE_ROOT = WORKSPACE / "third_party" / "sources"


class OfficialSourceError(RuntimeError):
    """Raised when a locked upstream implementation cannot be used exactly."""


class _BroadcastAttentionMaskCache:
    """Cache ChannelTokenFormer's deterministic structural attention mask.

    The locked upstream implementation rebuilds and materializes the same mask
    for every layer and every forward pass.  Build its single-batch,
    single-head form once per structural identity, then expose a broadcast view
    with the exact shape expected by the upstream attention implementation.
    """

    def __init__(self, builder: Any) -> None:
        self._builder = builder
        self._cache: dict[tuple[Any, ...], tuple[Tensor, tuple[int, ...]]] = {}
        self.cache_misses = 0

    def __call__(
        self,
        batch_size: int,
        n_heads: int,
        n_vars: int,
        length: int,
        num_patches_list: list[int],
        num_global_tokens: int = 1,
        device: str | torch.device = "cuda",
    ) -> tuple[Tensor, list[int]]:
        key = (
            int(n_vars),
            int(length),
            tuple(int(value) for value in num_patches_list),
            int(num_global_tokens),
            str(torch.device(device)),
        )
        cached = self._cache.get(key)
        if cached is None:
            base_mask, num_tokens_list = self._builder(
                1,
                1,
                n_vars,
                length,
                num_patches_list,
                num_global_tokens,
                device=device,
            )
            cached = (base_mask, tuple(int(value) for value in num_tokens_list))
            self._cache[key] = cached
            self.cache_misses += 1
        base_mask, num_tokens = cached
        return (
            base_mask.expand(int(batch_size), int(n_heads), -1, -1),
            list(num_tokens),
        )


class _RegularDailyChannelTokenForecast:
    """Vectorized execution of the locked model's homogeneous daily path.

    The benchmark profile assigns all 125 variables sampling period 1 and
    patch length 10.  Upstream expresses that homogeneous case as hundreds of
    Python loops and tiny CUDA launches.  This adapter uses the same learned
    modules, token order, normalization, encoder, token selection, projection,
    and de-normalization in batched tensor form.
    """

    def __init__(self, model: nn.Module) -> None:
        patcher = model.channel_wise_patch_embedding
        patch_lens = tuple(int(value) for value in model.patch_lens)
        sampling_periods = tuple(int(value) for value in model.sampling_periods)
        if (
            not patch_lens
            or len(set(patch_lens)) != 1
            or len(set(sampling_periods)) != 1
            or sampling_periods[0] != 1
            or float(model.keep_prob) != 1.0
        ):
            raise ValueError("regular-daily ChannelTokenFormer fast path invariants failed")
        if len(set(patcher.var_to_key)) != 1 or len(model.head.unique_period) != 1:
            raise ValueError("regular-daily ChannelTokenFormer modules are not homogeneous")
        self._model = model
        self._forecast = model.forecast_multi
        self.patch_len = patch_lens[0]
        self.patch_key = patcher.var_to_key[0]
        self.period_key = model.head.var_to_period[0]

    def __call__(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor | None,
        x_dec: Tensor | None,
        x_mark_dec: Tensor | None,
    ) -> Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        model = self._model
        batch_size, length, n_vars = x_enc.shape
        if n_vars != model.n_vars or length % self.patch_len:
            raise ValueError("regular-daily ChannelTokenFormer input identity changed")
        if model.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            stdevs = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = (x_enc - means) / stdevs

        patcher = model.channel_wise_patch_embedding
        num_patches = length // self.patch_len
        if patcher.training and num_patches > 1:
            # Upstream consumes one CPU RNG value for each non-final patch even
            # when keep_prob=1. Preserve that RNG advance while removing the
            # Python loop; no patch can be dropped under the frozen profile.
            torch.rand(n_vars * (num_patches - 1))
        patches = x_enc.transpose(1, 2).reshape(
            batch_size, n_vars, num_patches, self.patch_len
        )
        embedded = patcher.embedding_dict[self.patch_key](patches)
        position = patcher.position_embedding(embedded[:, 0]).unsqueeze(1)
        embedded = embedded + position
        global_tokens = patcher.glb_tokens.reshape(
            1, 1, model.num_global_tokens, patcher.glb_tokens.shape[-1]
        ).expand(batch_size, n_vars, -1, -1)
        embedded = torch.cat((embedded, global_tokens), dim=2)
        channel_ids = torch.arange(n_vars, device=embedded.device)
        channel = patcher.channel_embedding(channel_ids).reshape(1, n_vars, 1, -1)
        embedded = embedded + channel
        tokens_per_variable = num_patches + model.num_global_tokens
        encoded, num_tokens_list = model.encoder(
            embedded.reshape(batch_size, n_vars * tokens_per_variable, -1),
            [num_patches] * n_vars,
            n_vars,
        )
        model.latest_num_tokens_list = num_tokens_list
        blocks = encoded.reshape(batch_size, n_vars, tokens_per_variable, -1)
        selected = blocks[:, :, : model.num_global_tokens]
        if model.num_global_tokens == 1:
            selected = selected[:, :, 0]
        else:
            selected = selected.reshape(batch_size, n_vars, -1)
        projected = model.head.period_heads[self.period_key](selected)
        projected = model.head.dropout(projected).transpose(1, 2)
        if model.use_norm:
            projected = projected * stdevs + means
        return projected


MODULES = {
    "m01_itransformer": ("model.iTransformer", "Model"),
    "m02_timemixer": ("models.TimeMixer", "Model"),
    "m03_moderntcn": ("models.ModernTCN", "Model"),
    "m04_softs": ("models.SOFTS", "Model"),
    "m05_timexer": ("models.TimeXer", "Model"),
    "m06_timebridge": ("model.TimeBridge", "Model"),
    "m07_timefilter": ("models.TimeFilter", "Model"),
    "m09_srsnet": (
        "ts_benchmark.baselines.srsnet.models.srsnet_model",
        "SRSNetModel",
    ),
    "m10_channeltokenformer": ("models.ChannelTokenFormer", "Model"),
}


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def source_record(source_id: str) -> dict[str, Any]:
    sources = _yaml(LOCK_PATH)["sources"]
    if source_id not in sources:
        raise KeyError(f"unknown official source: {source_id}")
    return dict(sources[source_id])


def model_profile(source_id: str) -> dict[str, Any]:
    payload = _yaml(PROFILE_PATH)
    profile = {**payload["common"], **payload["profiles"][source_id]}
    for name in ("patch_lens", "sampling_periods"):
        value = profile.get(name)
        if isinstance(value, str) and value.startswith("repeat:"):
            repeated = float(value.split(":", 1)[1])
            if repeated.is_integer():
                repeated = int(repeated)
            profile[name] = [repeated] * int(profile["enc_in"])
    return profile


def _source_import_root(source_id: str, record: dict[str, Any]) -> Path:
    checkout = SOURCE_ROOT / source_id
    inventory = json.loads((WORKSPACE / 'third_party/source_hashes.json').read_text())[source_id]
    for relative, expected in inventory.items():
        path = checkout / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise OfficialSourceError(f'upstream missing or changed: {source_id}/{relative}; run python setup_sources.py')
    root = checkout / str(record.get("source_subdir", "."))
    if not (checkout / str(record["model_file"])).is_file():
        raise OfficialSourceError(f"locked model file missing for {source_id}")
    return root.resolve()


def _purge_colliding_upstream_modules() -> None:
    prefixes = ("layers", "model", "models", "utils", "ts_benchmark")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(f"{prefix}." for prefix in prefixes)):
            del sys.modules[name]


def load_official_class(source_id: str) -> type[nn.Module]:
    if source_id not in MODULES:
        raise KeyError(f"unsupported source ID: {source_id}")
    record = source_record(source_id)
    import_root = _source_import_root(source_id, record)
    _purge_colliding_upstream_modules()
    if source_id == 'm09_srsnet':
        # Load only the SRSNet core. Its benchmark package __init__ imports
        # an unrelated full model zoo, which is outside this ten-model handoff.
        import types
        for name in ('ts_benchmark', 'ts_benchmark.baselines', 'ts_benchmark.baselines.srsnet'):
            module = types.ModuleType(name)
            module.__path__ = [str(import_root.joinpath(*name.split('.')))]
            sys.modules[name] = module
    sys.path.insert(0, str(import_root))
    try:
        module_name, class_name = MODULES[source_id]
        module = importlib.import_module(module_name)
        value = getattr(module, class_name)
    except Exception as exc:
        raise OfficialSourceError(f"failed to import {source_id}: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(import_root))
        except ValueError:
            pass
    return value


class OfficialForecastBackbone(nn.Module):
    """Thin invocation shim; the architecture itself remains upstream code."""

    def __init__(self, source_id: str) -> None:
        super().__init__()
        self.source_id = source_id
        self.record = source_record(source_id)
        self.profile = model_profile(source_id)
        official_class = load_official_class(source_id)
        self.model = official_class(SimpleNamespace(**self.profile))
        if source_id == "m10_channeltokenformer":
            # This is an execution-only optimization around the locked model:
            # the mask values and public tensor shape remain byte-identical.
            for layer in self.model.encoder.layers:
                layer.build_attention_mask = _BroadcastAttentionMaskCache(
                    layer.build_attention_mask
                )
            self.model.forecast_multi = _RegularDailyChannelTokenForecast(self.model)
        self.endogenous_feature_index = self.profile.get("endogenous_feature_index")
        if source_id == "m07_timefilter":
            tokens_per_variable = int(self.profile["seq_len"]) // int(self.profile["patch_len"])
            token_count = int(self.profile["enc_in"]) * tokens_per_variable
            query = torch.arange(token_count)[:, None]
            key = torch.arange(token_count)[None, :]
            spatial = (key % tokens_per_variable == query % tokens_per_variable) & (key != query)
            temporal = (key // tokens_per_variable) == (query // tokens_per_variable)
            other = ~(spatial | temporal)
            structural = torch.stack((spatial, temporal, other), dim=1).to(torch.float32)
            self.register_buffer("timefilter_structural_mask", structural, persistent=False)

    @property
    def output_channels(self) -> int:
        return 1 if self.source_id == "m05_timexer" else int(self.profile["enc_in"])

    @property
    def identity(self) -> dict[str, Any]:
        core = {
            "source_id": self.source_id,
            "official_commit": self.record["commit"],
            "official_model_file": self.record["model_file"],
            "profile": self.profile,
        }
        encoded = json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return {**core, "identity_sha256": hashlib.sha256(encoded).hexdigest()}

    def _model_input(self, x: Tensor) -> Tensor:
        if self.source_id != "m05_timexer":
            return x
        endogenous = int(self.endogenous_feature_index)
        order = [index for index in range(x.shape[-1]) if index != endogenous] + [endogenous]
        return x[..., order]

    def forward(self, x: Tensor, feature_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if x.ndim != 3:
            raise ValueError("official backbone input must be [batch, lookback, features]")
        if feature_mask is not None:
            if feature_mask.shape != x.shape or feature_mask.dtype is not torch.bool:
                raise ValueError("feature_mask must be bool and match x")
            x = x.masked_fill(~feature_mask, 0.0)
        x = self._model_input(x)
        if self.source_id == "m09_srsnet":
            output = self.model(x)
        elif self.source_id == "m07_timefilter":
            output = self.model(
                x,
                masks=self.timefilter_structural_mask,
                is_training=self.training,
            )
        elif self.source_id == "m03_moderntcn":
            output = self.model(x, te=None)
        else:
            output = self.model(x, None, None, None)
        auxiliary = None
        if isinstance(output, tuple):
            output, auxiliary = output
            if not isinstance(auxiliary, Tensor):
                auxiliary = torch.as_tensor(float(auxiliary), dtype=x.dtype, device=x.device)
        if output.ndim != 3 or output.shape[0] != x.shape[0]:
            raise OfficialSourceError(
                f"{self.source_id} returned incompatible shape {tuple(output.shape)}"
            )
        if output.shape[-1] != self.output_channels:
            raise OfficialSourceError(
                f"{self.source_id} expected {self.output_channels} output channels, got {output.shape[-1]}"
            )
        return output, auxiliary
