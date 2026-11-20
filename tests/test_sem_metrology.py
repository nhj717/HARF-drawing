"""Unit tests for ``src.sem_metrology``.

Covers the TIFF metadata calibration cascade (Zeiss / FEI / ImageJ /
DPI rejection), the pure-struct image decoder round-trip, the
algebraic circle fits, the sub-pixel wall caliper on synthetic
erf-membrane bands and the full polar cross-section estimator on
synthetic cane / NANF micrographs.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest
from scipy.special import erf

from src.sem_metrology import (
    Polarity,
    ScaleCalibration,
    calibrate_from_scale_bar,
    detect_sem_banner_row,
    estimate_cross_section,
    extract_tiff_pixel_size,
    fit_circle_3pt,
    fit_circle_robust,
    fit_circle_taubin,
    load_microscope_image,
    measure_wall_profile_subpixel,
    normalize_sem_image,
)

# ---------------------------------------------------------------------------
# Minimal baseline TIFF builder (pure struct)
# ---------------------------------------------------------------------------

_TYPE_SIZES = {2: 1, 3: 2, 4: 4, 5: 8}


def _build_tiff(
    endian="<",
    w=16,
    h=16,
    bps=8,
    pixels=None,
    extra_tags=(),
    xres=None,
    resunit=None,
):
    """A baseline uncompressed greyscale TIFF with optional metadata."""
    if pixels is None:
        pixels = np.full((h, w), 128, dtype=np.uint8)
    arr = np.asarray(pixels)
    if bps == 8:
        strip = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    else:
        bo = ">" if endian == ">" else "<"
        strip = (
            np.ascontiguousarray(arr)
            .astype(np.uint16)
            .astype(np.dtype(bo + "u2"))
            .tobytes()
        )
    entries: list[tuple[int, int, int, bytes]] = [
        (256, 4, 1, struct.pack(endian + "I", w)),
        (257, 4, 1, struct.pack(endian + "I", h)),
        (258, 3, 1, struct.pack(endian + "H", bps)),
        (259, 3, 1, struct.pack(endian + "H", 1)),
        (262, 3, 1, struct.pack(endian + "H", 1)),
        (273, 4, 1, struct.pack(endian + "I", 8)),
        (278, 3, 1, struct.pack(endian + "H", h)),
        (279, 4, 1, struct.pack(endian + "I", len(strip))),
    ]
    entries.extend(extra_tags)
    if xres is not None:
        entries.append(
            (282, 5, 1, struct.pack(endian + "II", xres[0], xres[1]))
        )
    if resunit is not None:
        entries.append((296, 3, 1, struct.pack(endian + "H", resunit)))
    entries.sort(key=lambda e: e[0])

    ifd_off = 8 + len(strip)
    ext_off = ifd_off + 2 + 12 * len(entries) + 4
    ifd = bytearray(struct.pack(endian + "H", len(entries)))
    externals: list[bytes] = []
    for tag, typ, cnt, raw in entries:
        if _TYPE_SIZES[typ] * cnt <= 4:
            field = raw[:4] + b"\x00" * max(0, 4 - len(raw))
            ifd += struct.pack(endian + "HHI", tag, typ, cnt) + field
        else:
            ifd += struct.pack(endian + "HHII", tag, typ, cnt, ext_off)
            externals.append(raw)
            ext_off += len(raw)
    ifd += struct.pack(endian + "I", 0)

    out = bytearray()
    out += b"II" if endian == "<" else b"MM"
    out += struct.pack(endian + "HI", 42, ifd_off)
    out += strip
    out += ifd
    for blob in externals:
        out += blob
    return bytes(out)


# ---------------------------------------------------------------------------
# TIFF metadata calibration cascade
# ---------------------------------------------------------------------------


def test_extract_tiff_pixel_size_zeiss() -> None:
    text = b"ap_image_pixel_size = 2.5 [nm]\x00"
    data = _build_tiff(extra_tags=[(34118, 2, len(text), text)])
    cal = extract_tiff_pixel_size(data)
    assert cal.pixel_size_m == pytest.approx(2.5e-9)
    assert cal.method == "zeiss"


def test_extract_tiff_pixel_size_fei() -> None:
    text = b"PixelWidth = 1.2e-08\x00"
    data = _build_tiff(extra_tags=[(34682, 2, len(text), text)])
    cal = extract_tiff_pixel_size(data)
    assert cal.pixel_size_m == pytest.approx(1.2e-8)
    assert cal.method == "fei"


def test_extract_tiff_pixel_size_imagej_cm() -> None:
    desc = b"unit=cm\x00"
    data = _build_tiff(
        extra_tags=[(270, 2, len(desc), desc)], xres=(10000, 1)
    )
    cal = extract_tiff_pixel_size(data)
    assert cal.pixel_size_m == pytest.approx(1e-6)
    assert cal.method == "imagej"


@pytest.mark.parametrize("dpi", (72, 96, 300))
def test_extract_tiff_pixel_size_dpi_rejected(dpi: int) -> None:
    """Plain DPI (inch) resolution tags must never calibrate."""
    data = _build_tiff(xres=(dpi, 1), resunit=2)
    assert extract_tiff_pixel_size(data).pixel_size_m is None


def test_extract_tiff_pixel_size_garbage() -> None:
    assert extract_tiff_pixel_size(b"not a tiff at all").pixel_size_m is None
    assert ScaleCalibration(None).pixel_size_m is None


def test_calibrate_from_scale_bar() -> None:
    px = calibrate_from_scale_bar((0.0, 0.0), (10.0, 0.0), 5e-6)
    assert px == pytest.approx(0.5e-6)
    # Diagonal 3-4-5 scale bar of 25 um.
    px = calibrate_from_scale_bar((0.0, 0.0), (3.0, 4.0), 25e-6)
    assert px == pytest.approx(5e-6)
    with pytest.raises(ValueError):
        calibrate_from_scale_bar((1.0, 1.0), (1.0, 1.0), 1e-6)


# ---------------------------------------------------------------------------
# Image loading round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endian", ("<", ">"))
@pytest.mark.parametrize("bps", (8, 16))
def test_load_microscope_image_roundtrip(endian: str, bps: int,
                                          tmp_path) -> None:
    rng = np.random.default_rng(3)
    h = w = 9
    if bps == 8:
        vals = rng.integers(0, 256, (h, w), dtype=np.uint8)
        peak = 255.0
    else:
        vals = rng.integers(0, 65536, (h, w), dtype=np.uint16)
        peak = 65535.0
    data = _build_tiff(endian=endian, w=w, h=h, bps=bps, pixels=vals)
    path = tmp_path / f"img_{endian.strip('<>')}_{bps}.tif"
    path.write_bytes(data)
    img = load_microscope_image(path)
    assert img.shape == (h, w)
    assert img.dtype == np.float64
    assert img == pytest.approx(vals / peak)


# ---------------------------------------------------------------------------
# Circle fits
# ---------------------------------------------------------------------------


def _arc_points(xc, yc, r, n, a0=0.3, span=2.4):
    th = a0 + np.linspace(0.0, span, n)
    return np.column_stack((xc + r * np.cos(th), yc + r * np.sin(th)))


def test_fit_circle_taubin_exact_arc() -> None:
    pts = _arc_points(11.3, -7.6, 137.42, 60)
    xc, yc, r = fit_circle_taubin(pts)
    assert xc == pytest.approx(11.3, rel=1e-6)
    assert yc == pytest.approx(-7.6, rel=1e-6)
    assert r == pytest.approx(137.42, rel=1e-6)


def test_fit_circle_3pt_exact() -> None:
    pts = _arc_points(-3.0, 4.0, 21.5, 3, a0=1.0)
    xc, yc, r = fit_circle_3pt(
        (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]),
        (pts[2, 0], pts[2, 1]),
    )
    assert xc == pytest.approx(-3.0, rel=1e-9)
    assert yc == pytest.approx(4.0, rel=1e-9)
    assert r == pytest.approx(21.5, rel=1e-9)
    with pytest.raises(ValueError):
        fit_circle_3pt((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))


def test_fit_circle_robust_rejects_outliers() -> None:
    pts = _arc_points(-4.0, 9.0, 55.5, 40)
    bad = pts[:5] + np.array([30.0, -25.0])
    xc, yc, r = fit_circle_robust(np.vstack((pts, bad)))
    assert xc == pytest.approx(-4.0, rel=1e-3)
    assert yc == pytest.approx(9.0, rel=1e-3)
    assert r == pytest.approx(55.5, rel=1e-3)


# ---------------------------------------------------------------------------
# Banner detection & normalization
# ---------------------------------------------------------------------------


def test_detect_sem_banner_row_noise_only() -> None:
    rng = np.random.default_rng(0)
    img = rng.uniform(0.0, 1.0, (256, 256))
    assert detect_sem_banner_row(img) is None


def test_detect_sem_banner_row_rejects_small() -> None:
    assert detect_sem_banner_row(np.zeros((16, 16))) is None


def test_normalize_sem_image() -> None:
    img = np.array([[0.0, 0.5], [1.0, 0.25]])
    bright = normalize_sem_image(img, Polarity.BRIGHT_SILICA, 0.0, 100.0)
    assert bright[0, 0] == pytest.approx(0.0)
    assert bright[0, 1] == pytest.approx(0.5)
    assert bright[1, 0] == pytest.approx(1.0)
    dark = normalize_sem_image(img, Polarity.DARK_BOUNDARIES, 0.0, 100.0)
    assert dark[0, 0] == pytest.approx(1.0)
    assert dark[1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sub-pixel wall caliper on erf membranes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t_px", (3.25, 5.60, 9.85))
@pytest.mark.parametrize("theta_deg", (0.0, 27.0, 63.0))
def test_wall_caliper_erf_band(t_px: float, theta_deg: float) -> None:
    """A vertical band of width t; the caliper crosses at angle theta."""
    n = 160
    a, b = 70.0, 70.0 + t_px
    s2 = 0.6 * math.sqrt(2.0)
    x = np.arange(n, dtype=float)
    band = 0.5 * (1.0 + erf((x - a) / s2)) * 0.5 * (
        1.0 - erf((x - b) / s2)
    )
    img = np.repeat(band[None, :], n, axis=0)
    th = math.radians(theta_deg)
    direction = (math.cos(th), math.sin(th))
    start = (a - 12.0, 80.0 - 12.0 * math.tan(th))
    length = (t_px + 24.0) / math.cos(th)
    prof = measure_wall_profile_subpixel(img, start, direction, length)
    assert prof.thickness_px is not None
    assert prof.thickness_px == pytest.approx(t_px / math.cos(th), abs=0.2)


# ---------------------------------------------------------------------------
# Synthetic cross-sections (cane / NANF)
# ---------------------------------------------------------------------------

_EDGE_SIGMA = 0.6
_NOISE_SIGMA = 0.04
_BANNER_START = 905
_IMAGE_N = 1024
_CENTER = 512.0


def _inside(r: np.ndarray, r0: float) -> np.ndarray:
    """Smooth (erf) indicator of ``r < r0``."""
    s = _EDGE_SIGMA * math.sqrt(2.0)
    return 0.5 * (1.0 - erf((r - r0) / s))


def _synthetic_image(rings, r_zeta, od, n_cap, theta0, dark):
    """Bright-silica union model of a cane / fiber cross-section.

    ``rings`` is a tuple of ``(rho, rc, Rc)`` tube specs per nested
    ring; the core disk ends at ``rho_1 - Rc_1``.  Gaussian noise and
    a quasi-uniform bottom metadata banner are added afterwards.
    """
    xx, yy = np.meshgrid(
        np.arange(_IMAGE_N, dtype=float), np.arange(_IMAGE_N, dtype=float)
    )
    r = np.hypot(xx - _CENTER, yy - _CENTER)
    comps = [_inside(r, od) - _inside(r, r_zeta)]
    rho1, _rc1, rcap1 = rings[0]
    comps.append(_inside(r, rho1 - rcap1))
    for k in range(n_cap):
        th = theta0 + k * 2.0 * math.pi / n_cap
        for rho, rc, rcap in rings:
            d = np.hypot(
                xx - (_CENTER + rho * math.cos(th)),
                yy - (_CENTER + rho * math.sin(th)),
            )
            comps.append(_inside(d, rcap) - _inside(d, rc))
    silica = np.ones((_IMAGE_N, _IMAGE_N))
    for comp in comps:
        silica *= 1.0 - comp
    silica = 1.0 - silica
    base = 1.0 - silica if dark else silica
    rng = np.random.default_rng(42)
    img = np.clip(
        base + rng.normal(0.0, _NOISE_SIGMA, (_IMAGE_N, _IMAGE_N)),
        0.0,
        1.0,
    )
    img[_BANNER_START:, :] = np.clip(
        0.90
        + rng.normal(
            0.0, _NOISE_SIGMA, (_IMAGE_N - _BANNER_START, _IMAGE_N)
        ),
        0.0,
        1.0,
    )
    return img


@pytest.fixture(scope="module")
def cane_bright():
    """1-ring cane: R_zeta 320, tube (rho 215, rc 85, Rc 105), OD 365."""
    return _synthetic_image(
        ((215.0, 85.0, 105.0),), 320.0, 365.0, 6, 0.0, dark=False
    )


@pytest.fixture(scope="module")
def cane_dark(cane_bright):
    """Same geometry, dark-boundary polarity."""
    return _synthetic_image(
        ((215.0, 85.0, 105.0),), 320.0, 365.0, 6, 0.0, dark=True
    )


@pytest.fixture(scope="module")
def nanf_bright():
    """2-ring NANF: ring 1 (rho 230, rc 82, Rc 100), ring 2 tangent."""
    return _synthetic_image(
        ((230.0, 82.0, 100.0), (257.0, 45.0, 55.0)),
        330.0,
        370.0,
        6,
        0.0,
        dark=False,
    )


def _assert_cane(est) -> None:
    assert est.banner_crop_rows == (_BANNER_START, _IMAGE_N)
    assert est.od_px == pytest.approx(730.0, rel=0.015)
    assert est.zeta_px == pytest.approx(640.0, rel=0.015)
    assert est.d_core_px == pytest.approx(220.0, rel=0.015)
    assert est.n_cap == 6
    assert len(est.ring_bore_px) == 1
    assert est.ring_bore_px[0] == pytest.approx(170.0, rel=0.015)
    assert est.ring_od_px[0] == pytest.approx(210.0, rel=0.015)
    assert est.ring_wall_px[0] == pytest.approx(20.0, abs=0.6)


def test_estimate_cane_cross_section_bright(cane_bright) -> None:
    est = estimate_cross_section(cane_bright, Polarity.BRIGHT_SILICA)
    _assert_cane(est)
    assert est.polarity is Polarity.BRIGHT_SILICA


def test_estimate_cane_cross_section_dark(cane_dark) -> None:
    est = estimate_cross_section(cane_dark, Polarity.DARK_BOUNDARIES)
    _assert_cane(est)
    assert est.polarity is Polarity.DARK_BOUNDARIES


def test_estimate_cane_carries_pixel_size(cane_bright) -> None:
    est = estimate_cross_section(
        cane_bright, Polarity.BRIGHT_SILICA, pixel_size_m=2e-6
    )
    assert est.pixel_size_m == pytest.approx(2e-6)


def test_estimate_nanf_cross_section(nanf_bright) -> None:
    est = estimate_cross_section(nanf_bright, Polarity.BRIGHT_SILICA)
    assert est.banner_crop_rows == (_BANNER_START, _IMAGE_N)
    assert est.od_px == pytest.approx(740.0, rel=0.015)
    assert est.zeta_px == pytest.approx(660.0, rel=0.015)
    assert est.d_core_px == pytest.approx(260.0, rel=0.015)
    assert est.n_cap == 6
    assert len(est.ring_bore_px) == 2
    assert est.ring_bore_px[0] == pytest.approx(164.0, rel=0.015)
    assert est.ring_bore_px[1] == pytest.approx(90.0, rel=0.015)
    assert est.ring_od_px[0] == pytest.approx(200.0, rel=0.015)
    assert est.ring_od_px[1] == pytest.approx(110.0, rel=0.015)
    assert est.ring_wall_px[0] == pytest.approx(18.0, abs=0.6)
    assert est.ring_wall_px[1] == pytest.approx(10.0, abs=0.6)


def test_estimate_no_banner_adds_warning(cane_bright) -> None:
    est = estimate_cross_section(
        cane_bright[:_BANNER_START], Polarity.BRIGHT_SILICA
    )
    assert est.banner_crop_rows is None
    assert any("banner" in w for w in est.warnings)
