"""Commented CSV/TXT result export and JSON config persistence.

``export_results_table`` writes a ``# key: value`` metadata header
followed by the axial table (uniform ``dz`` grid) of every derived
quantity of a :class:`~src.models.SimulationResult`.
``save_config_json`` / ``load_config_json`` round-trip the full
:class:`~src.models.SimulationInputs` (and optionally
:class:`~src.models.OptimizationInputs`) without any ``.mat``
dependency.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.models import (
    InputError,
    OptimizationInputs,
    RingSpec,
    SimulationInputs,
    SimulationResult,
)

__all__ = [
    "default_user_config_path",
    "export_results_table",
    "load_config_json",
    "read_results_table",
    "read_ui_state",
    "save_config_json",
]

_CONFIG_VERSION = 1


def default_user_config_path() -> Path:
    """Default on-disk location of the user configuration file."""
    return Path.home() / ".harf_drawing" / "config.json"


# --------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------


def _metadata(result: SimulationResult) -> dict[str, Any]:
    inp = result.inputs
    meta: dict[str, Any] = {
        "status": result.status.value,
        "tau0_N": float(result.tau0),
        "z_end_m": float(result.z_end),
        "dz_m": (
            float(result.z[1] - result.z[0]) if len(result.z) > 1 else 0.0
        ),
        "elapsed_s": float(result.elapsed_s),
        "nb_nested": inp.nb_nested,
        "n_capillaries": inp.n_capillaries,
        "Tmax_C": inp.Tmax_C,
        "Pcore_Pa": inp.Pcore,
        "Pcap_Pa": [float(p) for p in inp.Pcap],
        "tension_target_N": inp.tension,
        "w0_m_per_s": inp.w0,
        "L_m": inp.L,
        "Lfour_m": inp.Lfour,
        "R0_m": inp.R0,
        "r0_m": inp.r0,
        "Text_C": inp.Text_C,
        "dT0_K": inp.dT0,
    }
    for i, ring in enumerate(inp.rings, start=1):
        meta[f"Rc0_{i}_m"] = ring.Rc0
        meta[f"rc0_{i}_m"] = ring.rc0
    if result.diagnostics is not None:
        diag = result.diagnostics
        meta["failure_status"] = diag.status.value
        meta["z_stop_m"] = float(diag.z_stop)
        if diag.ring is not None:
            meta["failure_ring"] = diag.ring
        meta["failure_message"] = diag.message
        meta["failure_advice"] = diag.advice
    return meta


def _columns(result: SimulationResult) -> dict[str, np.ndarray]:
    cols: dict[str, np.ndarray] = {
        "z_m": np.asarray(result.z, dtype=float),
        "V_m_min": np.asarray(result.w, dtype=float) * 60.0,
        "D_um": np.asarray(result.R, dtype=float) * 2e6,
        "d_um": np.asarray(result.r, dtype=float) * 2e6,
        "T_C": np.asarray(result.T, dtype=float) - 273.15,
        "T_furnace_C": np.asarray(result.Ta, dtype=float) - 273.15,
        "tension_g": np.asarray(result.tau, dtype=float) / 9.81 * 1e3,
    }
    if result.Rc:
        cols["Dcore_um"] = np.asarray(result.D_core, dtype=float) * 1e6
        contact = np.asarray(result.R_contact, dtype=float)
        cols["Rc1_over_Rcontact"] = np.asarray(
            result.Rc[0], dtype=float
        ) / np.where(contact > 0.0, contact, 1.0)
        for i, (Rc, rc) in enumerate(
            zip(result.Rc, result.rc, strict=True), start=1
        ):
            Dc = 2.0 * np.asarray(Rc, dtype=float)
            dc = 2.0 * np.asarray(rc, dtype=float)
            cols[f"Dc{i}_um"] = Dc * 1e6
            cols[f"dc{i}_um"] = dc * 1e6
            cols[f"t{i}_um"] = (Dc - dc) * 1e6
            cols[f"dc{i}_over_Dc{i}"] = dc / np.where(
                Dc > 0.0, Dc, 1.0
            )
            if i >= 2:
                prev_dc = 2.0 * np.asarray(result.rc[i - 2], dtype=float)
                cols[f"Dc{i}_over_dc{i - 1}"] = Dc / np.where(
                    prev_dc > 0.0, prev_dc, 1.0
                )
    return cols


def _fmt_meta(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_meta(v) for v in value) + "]"
    return str(value)


def export_results_table(
    result: SimulationResult,
    path: str | Path,
    fmt: str | None = None,
) -> Path:
    """Export the axial results table to ``.csv`` or ``.txt``.

    The file starts with ``# key: value`` metadata lines followed by a
    header row and one row per axial grid point (uniform ``dz``).
    """
    path = Path(path)
    suffix = (fmt if fmt is not None else path.suffix).lower().lstrip(".")
    if suffix not in ("csv", "txt"):
        raise ValueError(
            f"unsupported table format {suffix!r} (use 'csv' or 'txt')"
        )
    if path.suffix == "":
        path = path.with_suffix(f".{suffix}")
    meta = _metadata(result)
    cols = _columns(result)
    names = list(cols.keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.writelines(f"# {key}: {_fmt_meta(value)}\n" for key, value in meta.items())
        if suffix == "csv":
            writer = csv.writer(fh)
            writer.writerow(names)
            for row in zip(*(cols[name] for name in names), strict=True):
                writer.writerow([f"{v:.12g}" for v in row])
        else:
            fh.write("\t".join(names) + "\n")
            for row in zip(*(cols[name] for name in names), strict=True):
                fh.write("\t".join(f"{v:.12g}" for v in row) + "\n")
    return path


def read_results_table(path: str | Path) -> tuple[dict, list[str], dict]:
    """Read back a table written by :func:`export_results_table`.

    Returns ``(metadata, column_names, columns)`` where ``columns``
    maps each column name to a float ndarray.
    """
    path = Path(path)
    meta: dict[str, Any] = {}
    names: list[str] = []
    rows: list[list[float]] = []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for line in lines:
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            meta[key] = _parse_meta(value)
        elif line.strip():
            fields_ = line.split("\t") if "\t" in line else _split_csv(line)
            if not names:
                names = fields_
            else:
                rows.append([float(v) for v in fields_])
    columns = {
        name: np.array([row[i] for row in rows], dtype=float)
        for i, name in enumerate(names)
    }
    return meta, names, columns


def _split_csv(line: str) -> list[str]:
    return next(csv.reader([line]))


def _parse_meta(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_meta(v.strip()) for v in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# --------------------------------------------------------------------------
# JSON config persistence
# --------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_jsonable(getattr(obj, f.name))
            for f in dc_fields(obj)
        }
    if isinstance(obj, (tuple, list)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float) and not math.isfinite(obj):
        raise InputError(f"non-finite value {obj!r} cannot be saved")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_config_json(
    inputs: SimulationInputs,
    opt: OptimizationInputs | None = None,
    path: str | Path | None = None,
    ui_state: dict[str, Any] | None = None,
) -> Path:
    """Save the simulation (and optimization) inputs as JSON.

    ``ui_state`` optionally carries GUI-only values that the dataclasses
    cannot represent (e.g. inactive Ring 2/3 edits while
    ``Nnested = 1``) so reopening never loses typed values.
    """
    path = Path(path) if path is not None else default_user_config_path()
    payload: dict[str, Any] = {
        "version": _CONFIG_VERSION,
        "simulation": _to_jsonable(inputs),
    }
    if opt is not None:
        payload["optimization"] = _to_jsonable(opt)
    if ui_state is not None:
        payload["ui_state"] = ui_state
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def read_ui_state(path: str | Path) -> dict[str, Any]:
    """Read the optional ``ui_state`` block from a config file.

    Returns an empty dict for missing/unreadable files or configs saved
    without GUI state, so callers can merge unconditionally.
    """
    try:
        with open(Path(path), encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    state = payload.get("ui_state")
    return state if isinstance(state, dict) else {}


def load_config_json(
    path: str | Path,
) -> tuple[SimulationInputs, OptimizationInputs | None]:
    """Load a JSON config written by :func:`save_config_json`.

    Raises :class:`~src.models.InputError` on a missing/invalid file or
    schema mismatch.
    """
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise InputError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or "simulation" not in payload:
        raise InputError(f"{path} is not a HARF-drawing config file")
    if payload.get("version") != _CONFIG_VERSION:
        raise InputError(
            f"unsupported config version {payload.get('version')!r}"
        )
    sim = dict(payload["simulation"])
    try:
        rings = tuple(RingSpec(**r) for r in sim.pop("rings"))
        pcap = tuple(float(p) for p in sim.pop("Pcap"))
        inputs = SimulationInputs(rings=rings, Pcap=pcap, **sim)
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"invalid simulation schema: {exc}") from exc
    inputs.validate()
    opt = None
    if payload.get("optimization") is not None:
        raw = dict(payload["optimization"])
        try:
            raw["Pcap_min"] = tuple(float(p) for p in raw["Pcap_min"])
            raw["Pcap_max"] = tuple(float(p) for p in raw["Pcap_max"])
            opt = OptimizationInputs(**raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise InputError(f"invalid optimization schema: {exc}") from exc
        opt.validate(len(rings))
    return inputs, opt
