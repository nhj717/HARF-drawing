"""CSV/TXT export and JSON config round-trips (A5)."""

import json

import numpy as np
import pytest

from src.io_utils import (
    export_results_table,
    load_config_json,
    read_results_table,
    save_config_json,
)
from src.models import (
    InputError,
    default_optimization_inputs,
    default_simulation_inputs,
)
from src.solvers import run_direct_simulation


@pytest.fixture(scope="module")
def sim():
    """One default direct simulation reused by the whole module."""
    result = run_direct_simulation(default_simulation_inputs(nb_nested=1))
    assert result.ok
    return result


@pytest.fixture(scope="module")
def sim2():
    """A two-ring direct simulation with valid nested geometry."""
    from dataclasses import replace

    from src.models import RingSpec

    inp = default_simulation_inputs(nb_nested=2)
    rings = list(inp.rings)
    rings[1] = RingSpec(Rc0=0.24e-3 / 2, rc0=0.20e-3 / 2)
    inp = replace(inp, rings=tuple(rings))
    result = run_direct_simulation(inp)
    assert result.ok
    return result


# --------------------------------------------------------------------------
# Results table export / import
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["csv", "txt"])
def test_export_read_round_trip(sim, tmp_path, fmt):
    path = export_results_table(sim, tmp_path / f"table.{fmt}")
    assert path.exists()
    meta, names, cols = read_results_table(path)

    # metadata round-trips
    assert meta["status"] == "success"
    assert meta["nb_nested"] == 1
    assert meta["Tmax_C"] == pytest.approx(sim.inputs.Tmax_C)
    assert meta["Pcore_Pa"] == pytest.approx(sim.inputs.Pcore)
    assert meta["Rc0_1_m"] == pytest.approx(sim.inputs.rings[0].Rc0)
    assert meta["tau0_N"] == pytest.approx(sim.tau0, rel=1e-9)

    # uniform dz grid and column values round-trip to full precision
    z = cols["z_m"]
    assert len(z) == len(sim.z)
    assert np.all(np.diff(z) <= 1e-5 + 1e-12)
    np.testing.assert_allclose(cols["z_m"], sim.z, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(
        cols["V_m_min"], sim.w * 60.0, rtol=1e-11, atol=1e-15
    )
    np.testing.assert_allclose(
        cols["D_um"], sim.R * 2e6, rtol=1e-11, atol=1e-9
    )
    np.testing.assert_allclose(
        cols["T_C"], sim.T - 273.15, rtol=1e-11, atol=1e-9
    )
    np.testing.assert_allclose(
        cols["tension_g"], sim.tau / 9.81 * 1e3, rtol=1e-11, atol=1e-9
    )
    assert "Dcore_um" in names and "Rc1_over_Rcontact" in names
    assert "Dc1_um" in names and "dc1_over_Dc1" in names


def test_export_two_ring_columns(sim2, tmp_path):
    path = export_results_table(sim2, tmp_path / "two.csv")
    _, names, cols = read_results_table(path)
    for i in (1, 2):
        assert f"Dc{i}_um" in names
        assert f"dc{i}_um" in names
        assert f"t{i}_um" in names
        assert f"dc{i}_over_Dc{i}" in names
    assert "Dc2_over_dc1" in names
    assert cols["Dc2_over_dc1"].shape == cols["z_m"].shape
    # ring 2 nests strictly inside ring 1's bore
    assert np.all(cols["Dc2_um"] < cols["dc1_um"])


def test_export_txt_is_tab_separated(sim, tmp_path):
    path = export_results_table(sim, tmp_path / "table.txt")
    lines = path.read_text(encoding="utf-8").splitlines()
    meta_lines = [ln for ln in lines if ln.startswith("# ")]
    assert len(meta_lines) >= 10
    header = next(ln for ln in lines if not ln.startswith("#"))
    assert "\t" in header
    assert header.split("\t")[0] == "z_m"


def test_export_failure_run_includes_diagnostics(tmp_path):
    from src.models import default_simulation_inputs

    sim = run_direct_simulation(
        default_simulation_inputs(nb_nested=1).with_process(Pcore=1e6)
    )
    assert not sim.ok
    path = export_results_table(sim, tmp_path / "fail.csv")
    meta, _, cols = read_results_table(path)
    assert meta["status"] != "success"
    assert meta["failure_status"] == sim.status.value
    assert 0.0 < meta["z_stop_m"] < sim.inputs.L
    assert len(cols["z_m"]) == len(sim.z)


def test_export_rejects_unknown_format(sim, tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        export_results_table(sim, tmp_path / "table.parquet")


# --------------------------------------------------------------------------
# JSON config persistence
# --------------------------------------------------------------------------


def test_config_json_round_trip(tmp_path):
    inp = default_simulation_inputs(nb_nested=2)
    opt = default_optimization_inputs(nb_nested=2)
    path = save_config_json(inp, opt, tmp_path / "config.json")
    back, back_opt = load_config_json(path)
    assert back.r0 == inp.r0 and back.R0 == inp.R0
    assert back.rings == inp.rings
    assert back.Pcap == inp.Pcap
    assert back.Tmax_C == inp.Tmax_C and back.Pcore == inp.Pcore
    assert back.viscosity_expr == inp.viscosity_expr
    assert back == inp  # full dataclass equality
    assert back_opt == opt


def test_config_json_minimal(sim, tmp_path):
    path = save_config_json(sim.inputs, None, tmp_path / "min.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert "optimization" not in payload
    back, back_opt = load_config_json(path)
    assert back == sim.inputs
    assert back_opt is None


def test_load_missing_file(tmp_path):
    with pytest.raises(InputError, match="not found"):
        load_config_json(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError, match="invalid JSON"):
        load_config_json(p)


def test_load_wrong_schema(tmp_path):
    p = tmp_path / "wrong.json"
    p.write_text(json.dumps({"version": 1, "simulation": {"r0": 1.0}}),
                 encoding="utf-8")
    with pytest.raises(InputError, match="schema"):
        load_config_json(p)


def test_load_wrong_version(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"version": 99, "simulation": {}}),
                 encoding="utf-8")
    with pytest.raises(InputError, match="version"):
        load_config_json(p)


def test_load_not_a_config(tmp_path):
    p = tmp_path / "notcfg.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(InputError, match="not a HARF-drawing config"):
        load_config_json(p)


def test_non_finite_value_rejected_on_save(tmp_path):
    inp = default_simulation_inputs(nb_nested=1).with_process(Pcore=float("nan"))
    with pytest.raises(InputError, match="non-finite"):
        save_config_json(inp, None, tmp_path / "nan.json")
