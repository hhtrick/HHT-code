"""Train-only target scaling and inverse transforms (no torch dependency)."""
from typing import Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler


def target_transform_for(cfg: Mapping, property_name: str) -> str:
    """Select by property, so E_article follows E rather than its folder name."""
    names = cfg.get("LOG_TARGET_DATASETS", [])
    if not isinstance(names, (list, tuple)) or not all(isinstance(x, str) for x in names):
        raise ValueError("LOG_TARGET_DATASETS must be a list of property names, e.g. ['E', 'UTS'].")
    property_name = "E" if property_name == "E_article" else property_name
    return "log" if property_name in names else "identity"


def _finite(values, label):
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values; no clipping or filtering is applied.")
    return array


def validate_targets(values, transform="identity"):
    values = _finite(values, "Target")
    if transform not in ("identity", "log"):
        raise ValueError(f"Unknown target transform: {transform!r}")
    if transform == "log" and np.any(values <= 0):
        invalid = np.flatnonzero(values.reshape(-1) <= 0)
        raise ValueError(
            f"Log target transform requires strictly positive targets; "
            f"found {len(invalid)} non-positive value(s), first flat indices {invalid[:10].tolist()}. "
            "Correct the data or remove this property from LOG_TARGET_DATASETS; "
            "targets are never silently clipped or removed."
        )
    return values


def inverse_from_parameters(values, mean, scale, transform="identity"):
    """Return physical-unit predictions, raising rather than clipping overflow."""
    values = _finite(values, "Scaled prediction")
    with np.errstate(over="raise", invalid="raise"):
        try:
            restored = values * scale + mean
            if transform == "log":
                restored = np.exp(restored)
            elif transform != "identity":
                raise ValueError(f"Unknown target transform: {transform!r}")
        except FloatingPointError as exc:
            raise ValueError("Target inverse transform overflowed; inspect predictions/training stability.") from exc
    return _finite(restored, "Inverse-transformed prediction")


class TargetScaler:
    """Optional natural log followed by sklearn StandardScaler; pickle safe."""

    def __init__(self, transform="identity"):
        if transform not in ("identity", "log"):
            raise ValueError(f"Unknown target transform: {transform!r}")
        self.target_transform = transform
        self.scaler = StandardScaler()

    def fit(self, values):
        values = validate_targets(values, self.target_transform)
        transformed = np.log(values) if self.target_transform == "log" else values
        self.scaler.fit(transformed)
        return self

    def transform(self, values):
        values = validate_targets(values, self.target_transform)
        transformed = np.log(values) if self.target_transform == "log" else values
        return self.scaler.transform(transformed)

    def inverse_transform(self, values):
        return inverse_from_parameters(values, self.mean_, self.scale_, self.target_transform)

    @property
    def mean_(self):
        return self.scaler.mean_

    @property
    def scale_(self):
        return self.scaler.scale_


def scaler_transform(scaler):
    """Legacy pickled StandardScaler objects always mean raw-target scaling."""
    return getattr(scaler, "target_transform", "identity")


def fit_target_scaler(targets, train_indices, *, cfg=None, property_name=""):
    transform = target_transform_for(cfg or {}, property_name)
    values = validate_targets(targets, transform)
    return TargetScaler(transform).fit(values[np.asarray(train_indices, dtype=int)].reshape(-1, 1))


def validate_scaler_config(scaler, cfg, property_name, *, context="checkpoint"):
    expected = target_transform_for(cfg, property_name)
    actual = scaler_transform(scaler)
    if expected != actual:
        raise ValueError(
            f"{context}: configured target transform {expected!r} for {property_name} "
            f"does not match saved scaler transform {actual!r}. A trained regression head "
            "must keep its saved target scale. Restore the matching config/scaler, or "
            "use RANDOM_INIT_WEIGHTS=True to fit a new head and scaler."
        )
    return actual


def inject_scaler_config(cfg, scaler, property_name):
    """Serializable parameters used by Lightning's physical-unit metrics."""
    cfg["_SCALER_MEAN"] = float(scaler.mean_[0])
    cfg["_SCALER_STD"] = float(scaler.scale_[0])
    cfg["_TARGET_TRANSFORM"] = scaler_transform(scaler)
    cfg["_TARGET_PROPERTY"] = property_name
    cfg["_NORMALIZED_METRICS"] = False
