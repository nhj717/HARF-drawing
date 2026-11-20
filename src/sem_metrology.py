"""SEM / microscope cross-section metrology (pure numpy + scipy).

Units: pixel coordinates are image pixels; physical lengths are SI
metres (``pixel_size_m``). ``estimate_cross_section`` works on the
canonical *bright-silica* map (silica ~ 1.0, air ~ 0.0), produced by
:func:`normalize_sem_image` for either acquisition polarity.

Components
----------
* TIFF metadata calibration (Zeiss / FEI / ImageJ / ResolutionUnit)
  parsed with pure-Python :mod:`struct` -- no external TIFF library.
* Banner (metadata bar) masking via row-mean jump detection.
* Algebraic (Taubin), 3-point and robust MAD circle fits.
* Polar multi-ring cross-section estimator (OD, zeta, D_core, n_cap,
  ring bores d_i, ring ODs Delta_i, walls t_i) with sub-pixel edges.
* 1-D sub-pixel wall caliper (strip-averaged, Gaussian-derivative
  edge localization with parabolic refinement).

Image loading supports uncompressed 8/16-bit baseline TIFF via a
pure-struct decoder, falls back to Pillow, then to ``QImage``.
"""

from __future__ import annotations

import enum
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

__all__ = [
    "CrossSectionEstimate",
    "MeasuredCircle",
    "Polarity",
    "ScaleCalibration",
    "WallProfile",
    "calibrate_from_scale_bar",
    "detect_sem_banner_row",
    "estimate_cross_section",
    "extract_tiff_pixel_size",
    "fit_circle_3pt",
    "fit_circle_robust",
    "fit_circle_taubin",
    "load_microscope_image",
    "measure_wall_profile_subpixel",
    "normalize_sem_image",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class Polarity(enum.Enum):
    """Acquisition polarity of the greyscale image."""

    BRIGHT_SILICA = "bright_silica"
    DARK_BOUNDARIES = "dark_boundaries"


@dataclass(frozen=True)
class ScaleCalibration:
    """Physical pixel size decoded from image metadata.

    ``pixel_size_m`` is None when the image carries no usable
    calibration (the inspector then falls back to the 2-click scale
    bar tool).
    """

    pixel_size_m: float | None
    method: str = "none"
    detail: str = ""


@dataclass(frozen=True)
class MeasuredCircle:
    """One fitted circle, in full-image pixel coordinates."""

    kind: str  # "od" | "zeta" | "ring_outer" | "ring_bore"
    ring: int  # 0 for od/zeta, 1-based ring index otherwise
    azimuth: int  # tube index k for ring circles, 0 otherwise
    xc: float
    yc: float
    r: float


@dataclass(frozen=True)
class WallProfile:
    """Strip-averaged 1-D intensity profile across a wall."""

    s: np.ndarray  # arc positions (px)
    intensity: np.ndarray
    derivative: np.ndarray
    edge_positions: tuple[float, float]  # rising / falling edge (px)
    thickness_px: float | None  # None when no wall was resolved


@dataclass(frozen=True)
class CrossSectionEstimate:
    """Full cross-section estimate in full-image pixel coordinates."""

    image_size_px: tuple[int, int]
    pixel_size_m: float | None
    polarity: Polarity
    banner_crop_rows: tuple[int, int] | None
    center_px: tuple[float, float]
    od_px: float
    zeta_px: float
    d_core_px: float
    n_cap: int
    theta0_rad: float
    ring_od_px: tuple[float, ...] = ()
    ring_bore_px: tuple[float, ...] = ()
    ring_wall_px: tuple[float, ...] = ()
    circles: tuple[MeasuredCircle, ...] = ()
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Circle fits
# ---------------------------------------------------------------------------


def fit_circle_taubin(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Taubin) circle fit; exact on noiseless arcs.

    Solves the 3x3 symmetric eigenproblem on the feature vector
    ``(z - zbar, u, v)`` with ``N = diag(4*zbar, 1, 1)`` scaling.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        raise ValueError("fit_circle_taubin needs >= 3 (x, y) points")
    x, y = pts[:, 0], pts[:, 1]
    mx, my = float(x.mean()), float(y.mean())
    u, v = x - mx, y - my
    z = u * u + v * v
    zm = float(z.mean())
    if zm <= 0.0:
        return mx, my, 0.0
    phi = np.column_stack((z - zm, u, v))
    m = phi.T @ phi / pts.shape[0]
    n_half = np.diag((1.0 / (2.0 * math.sqrt(zm)), 1.0, 1.0))
    _w, vecs = np.linalg.eigh(n_half @ m @ n_half)
    b = n_half @ vecs[:, 0]
    if abs(b[0]) < 1e-15:
        raise ValueError("degenerate Taubin fit")
    a = -b[1] / b[0]
    bb = -b[2] / b[0]
    xc = mx + 0.5 * a
    yc = my + 0.5 * bb
    r = math.sqrt(zm + 0.25 * (a * a + bb * bb))
    return float(xc), float(yc), float(r)


def fit_circle_3pt(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> tuple[float, float, float]:
    """Circumcircle of three points (exact closed form)."""
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        raise ValueError("collinear points have no circumcircle")
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    return float(ux), float(uy), float(r)


def _minimal_triplets(n: int) -> list[tuple[int, int, int]]:
    """~96 deterministic index triples (strided + seed-0 RNG subsets)."""
    triples: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    def add(i: int, j: int, k: int) -> None:
        key = tuple(sorted((i % n, j % n, k % n)))
        if len(set(key)) == 3 and key not in seen:
            seen.add(key)
            triples.append(key)

    stride = max(1, n // 3)
    for a in range(min(48, n)):
        add(a, a + stride, a + 2 * stride)
    rng = np.random.default_rng(0)
    attempts = 0
    while len(triples) < 96 and attempts < 512:
        attempts += 1
        i, j, k = (int(v) for v in rng.integers(0, n, size=3))
        add(i, j, k)
    return triples


def fit_circle_robust(
    points: np.ndarray,
    iterations: int = 3,
    mad_factor: float = 2.5,
    min_keep: int = 6,
) -> tuple[float, float, float]:
    """LMedS-seeded robust circle fit with trimmed Taubin refinement.

    Deterministic 3-point minimal subsets (strided + ``seed=0`` RNG)
    are scored by the median absolute radial residual, so gross
    leverage outliers cannot drag the seed (algebraic Taubin alone
    has a 0% breakdown point).  The best seed's inliers — residuals
    within ``max(mad_factor * 1.4826 * median|res|, 0.5)`` px — then
    go through ``iterations`` trimmed :func:`fit_circle_taubin` steps.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        raise ValueError("fit_circle_robust needs >= 3 (x, y) points")
    if pts.shape[0] < 6:
        return fit_circle_taubin(pts)
    best: tuple[float, float, float] | None = None
    best_med = math.inf
    for i, j, k in _minimal_triplets(pts.shape[0]):
        try:
            xc, yc, r = fit_circle_3pt(
                (pts[i, 0], pts[i, 1]),
                (pts[j, 0], pts[j, 1]),
                (pts[k, 0], pts[k, 1]),
            )
        except ValueError:
            continue
        if not math.isfinite(r) or r <= 0.0:
            continue
        resid = np.abs(np.hypot(pts[:, 0] - xc, pts[:, 1] - yc) - r)
        med = float(np.median(resid))
        if med < best_med:
            best_med = med
            best = (xc, yc, r)
    if best is None:
        xc, yc, r = fit_circle_taubin(pts)
    else:
        xc, yc, r = best
    keep = pts
    for _ in range(max(1, iterations)):
        resid = np.abs(np.hypot(keep[:, 0] - xc, keep[:, 1] - yc) - r)
        med = float(np.median(resid))
        thresh = max(mad_factor * 1.4826 * med, 0.5)
        mask = resid <= thresh
        if mask.all() or int(mask.sum()) < max(min_keep, 3):
            break
        keep = keep[mask]
        xc, yc, r = fit_circle_taubin(keep)
    return xc, yc, r


# ---------------------------------------------------------------------------
# TIFF metadata calibration (pure struct)
# ---------------------------------------------------------------------------

_UNIT_TO_M = {
    "m": 1.0,
    "nm": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
}

_ZEISS_VALUE = re.compile(
    r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*\[?\s*"
    r"(m|nm|um|µm|μm|mm|cm)?\s*\]?",
    re.IGNORECASE,
)


def _read_ifd(data: bytes) -> dict[int, object]:
    """Decode IFD0 of a baseline TIFF into ``{tag: value}``."""
    if len(data) < 8:
        return {}
    bo = data[:2]
    if bo == b"II":
        end = "<"
    elif bo == b"MM":
        end = ">"
    else:
        return {}
    if struct.unpack_from(end + "H", data, 2)[0] != 42:
        return {}
    ifd_off = struct.unpack_from(end + "I", data, 4)[0]
    if ifd_off + 2 > len(data):
        return {}
    count = struct.unpack_from(end + "H", data, ifd_off)[0]
    sizes = {2: 1, 3: 2, 4: 4, 5: 8}
    tags: dict[int, object] = {}
    for i in range(count):
        entry = ifd_off + 2 + 12 * i
        if entry + 12 > len(data):
            break
        tag, typ, cnt = struct.unpack_from(end + "HHI", data, entry)
        if typ not in sizes or cnt > 1 << 20:
            continue
        size = sizes[typ] * cnt
        if size <= 4:
            raw = data[entry + 8 : entry + 8 + size]
        else:
            off = struct.unpack_from(end + "I", data, entry + 8)[0]
            raw = data[off : off + size]
        if len(raw) < size:
            continue
        if typ == 2:
            tags[tag] = raw.split(b"\x00")[0].decode("latin-1", "replace")
        elif typ == 3:
            vals = struct.unpack(end + f"{cnt}H", raw[: 2 * cnt])
            tags[tag] = vals[0] if cnt == 1 else vals
        elif typ == 4:
            vals = struct.unpack(end + f"{cnt}I", raw[: 4 * cnt])
            tags[tag] = vals[0] if cnt == 1 else vals
        else:
            nums = struct.unpack(end + f"{2 * cnt}I", raw[: 8 * cnt])
            fracs = [
                a / b if b else 0.0
                for a, b in zip(nums[0::2], nums[1::2], strict=False)
            ]
            tags[tag] = fracs[0] if cnt == 1 else tuple(fracs)
    return tags


def _parse_keyed_text(text: str, key: str) -> str | None:
    """Value of ``key = value`` lines in keyed metadata text."""
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip().lower() == key.lower():
            return v.strip()
    return None


def extract_tiff_pixel_size(source: bytes | str | Path) -> ScaleCalibration:
    """Pixel size (metres) from TIFF metadata, priority-cascaded.

    (1) Zeiss ``34118`` keyed ``ap_image_pixel_size`` / ``Image Pixel
    Size`` with a bracketed unit, (2) FEI/TFS ``34682``/``34680``
    ``PixelWidth`` (m/px), (3) ImageJ ``270`` ``unit=`` +
    ``XResolution``, (4) ``ResolutionUnit == 3`` (cm) with
    ``XResolution > 1000``. Plain DPI (inch) tags are never accepted.
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        data = Path(source).read_bytes()
    tags = _read_ifd(data)
    if not tags:
        return ScaleCalibration(None, "none", "no TIFF IFD found")
    zeiss = tags.get(34118)
    if isinstance(zeiss, str):
        for key in ("ap_image_pixel_size", "image pixel size"):
            val = _parse_keyed_text(zeiss, key)
            if not val:
                continue
            m = _ZEISS_VALUE.search(val)
            if m is None:
                continue
            unit = (m.group(2) or "m").strip().lower()
            unit = {"µm": "µm"}.get(unit, unit)
            px = float(m.group(1)) * _UNIT_TO_M.get(unit, 1.0)
            return ScaleCalibration(px, "zeiss", f"{key} = {val}")
    for tag in (34682, 34680):
        fei = tags.get(tag)
        if isinstance(fei, str):
            m = re.search(r"PixelWidth\s*=\s*([-+0-9.eE]+)", fei)
            if m is not None:
                px = float(m.group(1))
                return ScaleCalibration(px, "fei", f"tag {tag} PixelWidth")
    desc = tags.get(270)
    xres = tags.get(282)
    if isinstance(desc, str) and isinstance(xres, float) and xres > 0.0:
        m = re.search(r"unit=(\S+)", desc)
        if m is not None:
            unit = m.group(1).strip()
            if unit in _UNIT_TO_M:
                px = _UNIT_TO_M[unit] / xres
                return ScaleCalibration(
                    px, "imagej", f"unit={unit}, xres={xres:g}"
                )
    res_unit = tags.get(296)
    if res_unit == 3 and isinstance(xres, float) and xres > 1000.0:
        return ScaleCalibration(
            0.01 / xres, "resolution_unit", f"{xres:g} px/cm"
        )
    return ScaleCalibration(None, "none", "no calibrated tag found")


def calibrate_from_scale_bar(
    p0: tuple[float, float],
    p1: tuple[float, float],
    length_m: float,
) -> float:
    """Pixel size from a 2-click scale-bar measurement (m per px)."""
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    n = math.hypot(dx, dy)
    if n < 1e-9:
        raise ValueError("scale-bar endpoints coincide")
    return float(length_m) / n


# ---------------------------------------------------------------------------
# Image loading (struct TIFF -> Pillow -> QImage)
# ---------------------------------------------------------------------------


def _decode_tiff_pixels(data: bytes) -> np.ndarray:
    """Greyscale [0, 1] float pixels of an uncompressed TIFF."""
    bo = data[:2]
    if bo == b"II":
        end = "<"
    elif bo == b"MM":
        end = ">"
    else:
        raise ValueError("not a TIFF")
    if struct.unpack_from(end + "H", data, 2)[0] != 42:
        raise ValueError("not a baseline TIFF")
    ifd_off = struct.unpack_from(end + "I", data, 4)[0]
    count = struct.unpack_from(end + "H", data, ifd_off)[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
    raw_tags: dict[int, object] = {}
    for i in range(count):
        entry = ifd_off + 2 + 12 * i
        if entry + 12 > len(data):
            break
        tag, typ, cnt = struct.unpack_from(end + "HHI", data, entry)
        if typ not in sizes or cnt > 1 << 24:
            continue
        size = sizes[typ] * cnt
        if size <= 4:
            raw = data[entry + 8 : entry + 8 + size]
        else:
            off = struct.unpack_from(end + "I", data, entry + 8)[0]
            raw = data[off : off + size]
        if len(raw) < size:
            continue
        if typ in (1, 2):
            raw_tags[tag] = raw
        elif typ == 3:
            vals = struct.unpack(end + f"{cnt}H", raw[: 2 * cnt])
            raw_tags[tag] = vals[0] if cnt == 1 else vals
        elif typ == 4:
            vals = struct.unpack(end + f"{cnt}I", raw[: 4 * cnt])
            raw_tags[tag] = vals[0] if cnt == 1 else vals
        else:
            nums = struct.unpack(end + f"{2 * cnt}I", raw[: 8 * cnt])
            fracs = [
                a / b if b else 0.0
                for a, b in zip(nums[0::2], nums[1::2], strict=False)
            ]
            raw_tags[tag] = fracs[0] if cnt == 1 else tuple(fracs)
    width = raw_tags.get(256)
    height = raw_tags.get(257)
    bps = raw_tags.get(258, 8)
    compression = raw_tags.get(259, 1)
    photo = raw_tags.get(262, 1)
    offsets = raw_tags.get(273)
    counts = raw_tags.get(279)
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or compression != 1
        or photo not in (0, 1)
        or bps not in (8, 16)
        or not isinstance(offsets, (tuple, int))
        or not isinstance(counts, (tuple, int))
    ):
        raise ValueError("unsupported TIFF variant")
    offs = offsets if isinstance(offsets, tuple) else (offsets,)
    cnts = counts if isinstance(counts, tuple) else (counts,)
    rows = raw_tags.get(278, height)
    if not isinstance(rows, int) or rows <= 0:
        rows = height
    buf = bytearray()
    for off, cnt in zip(offs, cnts, strict=False):
        buf.extend(data[off : off + cnt])
    if bps == 16:
        fmt = end + f"{len(buf) // 2}H"
        arr = np.asarray(
            struct.unpack(fmt, bytes(buf[: 2 * (len(buf) // 2)])),
            dtype=np.float64,
        )
        img = arr.reshape(-1, width)[:height].astype(np.float64)
        img /= 65535.0
    else:
        arr = np.frombuffer(bytes(buf), dtype=np.uint8)
        img = arr.reshape(-1, width)[:height].astype(np.float64)
        img /= 255.0
    if photo == 0:  # WhiteIsZero
        img = 1.0 - img
    return np.ascontiguousarray(img)


def load_microscope_image(path: str | Path) -> np.ndarray:
    """Greyscale image as float array in [0, 1] (rows, cols)."""
    p = Path(path)
    data = p.read_bytes()
    if data[:2] in (b"II", b"MM") and len(data) >= 8:
        try:
            return _decode_tiff_pixels(data)
        except Exception:  # noqa: BLE001 - fall through to Pillow
            pass
    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(p) as im:
            if im.mode in ("I;16", "I;16L", "I;16B", "I;16;N", "I"):
                arr = np.asarray(im, dtype=np.float64)
                peak = float(arr.max()) if arr.size else 0.0
                return arr / (65535.0 if peak > 255.5 else 255.0)
            arr = np.asarray(im.convert("L"), dtype=np.float64)
            return arr / 255.0
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - fall through to QImage
        pass
    from PySide6.QtGui import QImage

    img = QImage(str(p))
    if img.isNull():
        raise ValueError(f"cannot decode image: {p}")
    bpl = img.bytesPerLine()
    if img.format() == QImage.Format.Format_Grayscale16:
        raw = bytes(img.constBits())[: img.height() * bpl]
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(
            img.height(), bpl
        )
        out = (
            arr.view(np.uint16)[:, : img.width()].astype(np.float64)
            / 65535.0
        )
        return np.ascontiguousarray(out)
    img8 = img.convertToFormat(QImage.Format.Format_Grayscale8)
    bpl = img8.bytesPerLine()
    raw = bytes(img8.constBits())[: img8.height() * bpl]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(img8.height(), bpl)
    out = arr[:, : img8.width()].astype(np.float64) / 255.0
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Banner masking & normalization
# ---------------------------------------------------------------------------


def detect_sem_banner_row(image: np.ndarray) -> tuple[int, int] | None:
    """Rows ``(start, stop)`` of a metadata banner to exclude.

    Looks for a quasi-uniform row-mean band at the bottom
    (``[0.78 H, H)``) or top (``[0.02 H, 0.12 H]``) whose mean jumps
    by > 0.06 from the adjacent content rows, is uniform right at
    its start (a dark jacket edge yields a ramp band, not a banner)
    and sits far from the content median / 5th percentile. Returns
    ``None`` when no banner is found.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or img.shape[0] < 64 or img.shape[1] < 16:
        return None
    h, _w = img.shape
    mu = img.mean(axis=1)
    best: tuple[float, int, int, bool] | None = None  # score, b, top

    def scan(candidates: range, top: bool) -> None:
        nonlocal best
        for b in candidates:
            if top:
                band, content = mu[:b], mu[b:]
                above = mu[b : b + 5]
            else:
                band, content = mu[b:], mu[:b]
                above = mu[max(0, b - 5) : b]
            if band.size < 0.02 * h or band.size > 0.25 * h:
                continue
            if content.size < 16 or above.size == 0:
                continue
            if float(band.std()) >= 0.05:
                continue
            # A banner is uniform right at its start; the jacket
            # edge produces a ramp band whose head rows differ from
            # the band mean.
            if top:
                head = mu[max(0, b - 5) : b]
            else:
                head = mu[b : b + 5]
            if head.size == 0 or abs(
                float(head.mean()) - float(band.mean())
            ) >= 0.025:
                continue
            jump = abs(float(above.mean()) - float(band.mean()))
            if jump <= 0.06:
                continue
            med = float(np.median(content))
            if abs(float(band.mean()) - med) <= 0.08:
                continue
            q05 = float(np.quantile(content, 0.05))
            if abs(float(band.mean()) - q05) <= 0.08:
                continue
            if best is None or jump > best[0]:
                best = (jump, b, int(bool(top)), top)

    top_lo = max(3, int(0.02 * h))
    top_hi = max(top_lo + 1, int(0.12 * h))
    scan(range(top_lo, top_hi), True)
    scan(range(int(0.78 * h), h), False)
    if best is None:
        return None
    _score, b, _flag, top = best
    # Snap b onto the sharpest single-row mean jump nearby.
    lo = max(1, b - 5)
    hi = min(h - 1, b + 4)
    if hi > lo:
        diffs = np.abs(np.diff(mu[lo - 1 : hi + 1]))
        b = lo + int(np.argmax(diffs))
    if top:
        return (0, max(2, min(b, h - 2)))
    return (max(2, min(b, h - 2)), h)


def normalize_sem_image(
    image: np.ndarray,
    polarity: Polarity,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> np.ndarray:
    """Percentile-normalized *bright-silica* map in [0, 1]."""
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("normalize_sem_image needs a 2-D image")
    lo, hi = np.percentile(img, (p_low, p_high))
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    if polarity is Polarity.DARK_BOUNDARIES:
        norm = 1.0 - norm
    return norm


# ---------------------------------------------------------------------------
# Sub-pixel edge helpers
# ---------------------------------------------------------------------------


def _parabolic_shift(ym: float, y0: float, yp: float) -> float:
    """3-point parabolic vertex offset, clipped to +/- 1 sample."""
    denom = ym - 2.0 * y0 + yp
    if abs(denom) < 1e-15:
        return 0.0
    return float(np.clip((ym - yp) / (2.0 * denom), -1.0, 1.0))


def _parabolic_shift_arr(ym: np.ndarray, y0: np.ndarray, yp: np.ndarray):
    denom = ym - 2.0 * y0 + yp
    safe = np.where(np.abs(denom) < 1e-15, 1.0, denom)
    shift = np.where(
        np.abs(denom) < 1e-15, 0.0, (ym - yp) / (2.0 * safe)
    )
    return np.clip(shift, -1.0, 1.0)


def _peak_centroid(m: np.ndarray, s: np.ndarray, idx: int) -> float:
    """Weighted centroid of the contiguous 0.35-height run at ``m[idx]``.

    Sub-pixel erf edges become piecewise-linear ramps under bilinear
    sampling, so a derivative extremum is a flat plateau on which a
    plain argmax (or parabolic vertex) lands on quantization noise.
    The run of contiguous samples at or above 0.35 of the peak,
    weighted by their height above that level, centers the plateau
    without bias; the 0.35 level spans ~3 quantization ripple
    periods while keeping the two edges' runs of even a 3 px wall
    separated.
    """
    m = np.asarray(m, dtype=float)
    peak = float(m[idx])
    if peak <= 0.0:
        return float(s[idx])
    thr = 0.35 * peak
    lo = idx
    while lo > 0 and m[lo - 1] >= thr:
        lo -= 1
    hi = idx
    while hi < m.size - 1 and m[hi + 1] >= thr:
        hi += 1
    w = m[lo : hi + 1] - thr
    tot = float(w.sum())
    if tot <= 0.0:
        return float(s[idx])
    return float(np.dot(w, s[lo : hi + 1]) / tot)


def _wall_tangent(
    img: np.ndarray, x_peak: float, y_peak: float, d: np.ndarray
) -> np.ndarray:
    """Local wall tangent unit vector at a ridge peak point.

    A 5x5 structure-tensor window of the smoothed-image gradients
    gives the dominant gradient direction ``(gx, gy)`` (perpendicular
    to the wall); the tangent is its 90 degree rotation ``(-gy, gx)``.
    Falls back to the caliper-normal direction for flat windows.
    """
    gx_img = gaussian_filter(img, 1.5, order=(0, 1))
    gy_img = gaussian_filter(img, 1.5, order=(1, 0))
    h, w = img.shape
    ix, iy = int(round(x_peak)), int(round(y_peak))
    x0, x1 = max(ix - 2, 0), min(ix + 3, w)
    y0, y1 = max(iy - 2, 0), min(iy + 3, h)
    gx = gx_img[y0:y1, x0:x1]
    gy = gy_img[y0:y1, x0:x1]
    sxx = float(np.mean(gx * gx))
    syy = float(np.mean(gy * gy))
    sxy = float(np.mean(gx * gy))
    if sxx + syy < 1e-12:
        return np.array((-d[1], d[0]))
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    t = np.array((-math.sin(ang), math.cos(ang)))
    return t / max(math.hypot(t[0], t[1]), 1e-12)


def measure_wall_profile_subpixel(
    image: np.ndarray,
    start_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    length: float,
    half_width: float = 3.0,
    ds: float = 0.25,
) -> WallProfile:
    """Strip-averaged wall profile with sub-pixel edge locations.

    A 1-D centerline is sampled first (no lateral smearing) to
    locate the wall peak; the local wall tangent from the structure
    tensor of the smoothed image then defines the lateral strip
    direction.  Parallel lines offset along the wall tangent cross a
    straight wall band at the same arc position, so lateral
    averaging cannot broaden the edges at any oblique crossing
    angle.  The profile is differentiated with a 0.45 px
    Gaussian-derivative kernel and the rising / falling edges are
    located as the weighted centroid of the contiguous 0.35-height
    run of each derivative peak: sub-pixel erf edges become
    piecewise-linear ramps under bilinear sampling, so the derivative
    extrema are flat plateaus on which a plain argmax (or parabolic
    vertex) is biased by quantization noise.  ``thickness_px`` is
    None when either edge is not resolved.
    """
    img = np.asarray(image, dtype=float)
    d = np.asarray(direction_xy, dtype=float)
    norm = float(np.hypot(d[0], d[1]))
    if norm < 1e-12:
        raise ValueError("zero direction vector")
    d = d / norm
    n_s = max(4, int(round(float(length) / ds)) + 1)
    s = np.arange(n_s) * ds
    # 1) Centerline: locate the wall peak with zero lateral smearing.
    cx = start_xy[0] + s * d[0]
    cy = start_xy[1] + s * d[1]
    prof0 = map_coordinates(img, [cy, cx], order=1, mode="nearest")
    dev = prof0 - float(np.median(prof0))
    if float(dev.max()) >= -float(dev.min()):
        j = int(np.argmax(dev))
    else:
        j = int(np.argmin(dev))
    t_hat = _wall_tangent(img, float(cx[j]), float(cy[j]), d)
    # 2) Lateral strips along the wall tangent (parallel to the wall,
    #    NOT perpendicular to the oblique caliper line).
    lat = np.arange(-half_width, half_width + 1e-9, 0.5)
    sx = start_xy[0] + s[:, None] * d[0] + lat[None, :] * t_hat[0]
    sy = start_xy[1] + s[:, None] * d[1] + lat[None, :] * t_hat[1]
    vals = map_coordinates(
        img, [sy.ravel(), sx.ravel()], order=1, mode="nearest"
    )
    prof = vals.reshape(n_s, lat.size).mean(axis=1)
    deriv = gaussian_filter1d(prof, sigma=0.45 / ds, order=1) / ds
    i1 = int(np.argmax(deriv))
    i2 = int(np.argmin(deriv))
    ok = (
        0 < i1 < n_s - 1
        and 0 < i2 < n_s - 1
        and i1 != i2
        and abs(deriv[i1]) >= 1e-4
        and abs(deriv[i2]) >= 1e-4
    )
    if not ok:
        return WallProfile(s, prof, deriv, (math.nan, math.nan), None)
    e1 = _peak_centroid(deriv, s, i1)
    e2 = _peak_centroid(-deriv, s, i2)
    return WallProfile(
        s, prof, deriv, (float(e1), float(e2)), float(abs(e2 - e1))
    )


def _edge_points_along(
    norm: np.ndarray,
    origin: tuple[float, float],
    dirs: np.ndarray,
    s_lo: float,
    s_hi: float,
    sign: int,
    half_width: float = 1.5,
    ds: float = 0.25,
    min_slope: float = 0.02,
) -> np.ndarray:
    """Sub-pixel edge points along per-ray search windows.

    ``dirs`` is ``(n, 2)`` unit vectors; for each ray the profile is
    laterally averaged over 5 offsets, differentiated, and the
    argmax (``sign > 0``) or argmin (``sign < 0``) located with a
    parabolic shift. Rays whose extremum sits on the window boundary
    or whose slope is weaker than ``min_slope`` are dropped.
    """
    dirs = np.asarray(dirs, dtype=float)
    n_s = max(4, int(round((s_hi - s_lo) / ds)) + 1)
    s = s_lo + np.arange(n_s) * ds
    offs = np.linspace(-half_width, half_width, 5)
    perp = np.stack((-dirs[:, 1], dirs[:, 0]), axis=1)
    dx = dirs[:, 0][:, None, None]
    dy = dirs[:, 1][:, None, None]
    px = perp[:, 0][:, None, None]
    py = perp[:, 1][:, None, None]
    sx = origin[0] + s[None, :, None] * dx + offs[None, None, :] * px
    sy = origin[1] + s[None, :, None] * dy + offs[None, None, :] * py
    vals = map_coordinates(
        norm,
        [sy.ravel(), sx.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    )
    prof = vals.reshape(dirs.shape[0], n_s, 5).mean(axis=2)
    deriv = gaussian_filter1d(prof, sigma=0.75 / ds, order=1, axis=1) / ds
    idx = np.argmax(deriv, axis=1) if sign > 0 else np.argmin(deriv, axis=1)
    rows = np.arange(dirs.shape[0])
    y0 = deriv[rows, idx]
    ym = deriv[rows, np.maximum(idx - 1, 0)]
    yp = deriv[rows, np.minimum(idx + 1, n_s - 1)]
    shift = _parabolic_shift_arr(ym, y0, yp)
    pts: list[tuple[float, float]] = []
    for k in range(dirs.shape[0]):
        i = int(idx[k])
        if i == 0 or i == n_s - 1:
            continue
        if abs(float(deriv[k, i])) < min_slope:
            continue
        sd = float(s[i]) + float(shift[k]) * ds
        pts.append((origin[0] + sd * dirs[k, 0], origin[1] + sd * dirs[k, 1]))
    return np.asarray(pts, dtype=float).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Polar cross-section estimator
# ---------------------------------------------------------------------------


def _fit_edge_circle(
    norm: np.ndarray,
    center: tuple[float, float],
    r_seed: float,
    sign: int,
    n_rays: int = 120,
) -> tuple[float, float, float]:
    """Robust circle fit of one circular edge around ``center``.

    Search window is ``+/- (0.10 r + 6)`` px around the seed radius;
    ``sign`` selects a rising (+1, argmax) or falling (-1, argmin)
    derivative extremum.
    """
    half = 0.10 * r_seed + 6.0
    s_lo = max(r_seed - half, 2.0)
    s_hi = r_seed + half
    th = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    dirs = np.stack((np.cos(th), np.sin(th)), axis=1)
    pts = _edge_points_along(
        norm, center, dirs, s_lo, s_hi, sign, half_width=1.5
    )
    if pts.shape[0] < 8:
        return float(center[0]), float(center[1]), float(r_seed)
    xc, yc, r = fit_circle_robust(pts)
    return xc, yc, r


def _count_and_phase(
    profile: np.ndarray,
    theta: np.ndarray,
    n_cap_hint: int | None = None,
) -> tuple[int, float]:
    """Capillary count and fold-argmax phase from an angular profile.

    ``n_cap`` is the dominant FFT harmonic in ``[2, 25)`` (or the
    hint); ``theta0`` maximizes the n-fold phase-folded profile and
    lives in ``[0, 2 pi / n)``.  Because flat arc tops saturate the
    folded argmax anywhere on a plateau, the phase is then refined
    with the intensity-weighted circular mean of each fold's bright
    arc (dark-gap baseline subtracted).
    """
    n = int(n_cap_hint) if n_cap_hint and n_cap_hint >= 2 else 0
    if n < 2:
        spec = np.abs(np.fft.rfft(profile - profile.mean()))
        n = int(np.argmax(spec[2:25]) + 2)
        n = max(2, min(n, 24))
    th_ext = np.concatenate(
        (theta - 2.0 * math.pi, theta, theta + 2.0 * math.pi)
    )
    a_ext = np.concatenate((profile, profile, profile))
    grid = np.linspace(0.0, 2.0 * math.pi / n, 720)
    folded = np.empty_like(grid)
    for i, phi in enumerate(grid):
        total = 0.0
        for k in range(n):
            total += float(
                np.interp(phi + k * 2.0 * math.pi / n, th_ext, a_ext)
            )
        folded[i] = total
    j = int(np.argmax(folded))
    theta0 = float(grid[j])
    if 0 < j < grid.size - 1:
        theta0 += _parabolic_shift(
            folded[j - 1], folded[j], folded[j + 1]
        ) * float(grid[1] - grid[0])
    # Refine: centroid of each fold's bright arc.  The window is
    # +/- 0.35 step so a neighbouring arc's tail cannot leak in.
    step = 2.0 * math.pi / n
    offs = np.linspace(-0.35 * step, 0.35 * step, 97)
    deltas: list[float] = []
    for k in range(n):
        aw = np.interp(theta0 + k * step + offs, th_ext, a_ext)
        wt = aw - float(aw.min())
        if float(wt.sum()) <= 1e-12:
            continue
        re = float(np.dot(wt, np.cos(offs)))
        im = float(np.dot(wt, np.sin(offs)))
        deltas.append(math.atan2(im, re))
    if deltas:
        zm = np.mean(np.exp(1j * np.asarray(deltas)))
        theta0 += math.atan2(float(zm.imag), float(zm.real))
    return n, float(theta0 % step)


def _radial_edge_scan(
    norm: np.ndarray,
    origin: tuple[float, float],
    azimuth: float,
    r_zeta: float,
    ds: float = 0.25,
    half_width: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Sub-pixel rising / falling edge radii along one tube ray.

    The ray is sampled over ``[0.08, 0.99] R_zeta`` (the 0.99 cap
    keeps the outer jacket wall outside the scan) with lateral
    offsets perpendicular to the ray — tangential to the concentric
    tube walls, so lateral averaging cannot broaden the edges.  A
    0.45 px Gaussian-derivative kernel locates dark->bright
    ``rises`` (core-facing wall starts) and bright->dark ``falls``
    (bore starts); extrema weaker than ``0.35 max|deriv|`` — e.g.
    the tangent-merge dip where the central core touches ring 1's
    core-facing wall — are rejected.
    """
    s_lo = 0.08 * r_zeta
    s_hi = 0.99 * r_zeta
    n_s = max(4, int(round((s_hi - s_lo) / ds)) + 1)
    s = s_lo + np.arange(n_s) * ds
    dirv = (math.cos(azimuth), math.sin(azimuth))
    perp = (-dirv[1], dirv[0])
    offs = np.asarray((-half_width, 0.0, half_width))
    sx = origin[0] + s[:, None] * dirv[0] + offs[None, :] * perp[0]
    sy = origin[1] + s[:, None] * dirv[1] + offs[None, :] * perp[1]
    vals = map_coordinates(
        norm, [sy.ravel(), sx.ravel()], order=1, mode="constant", cval=0.0
    )
    prof = vals.reshape(n_s, offs.size).mean(axis=1)
    deriv = gaussian_filter1d(prof, sigma=0.45 / ds, order=1) / ds
    thr = 0.35 * float(np.max(np.abs(deriv)))
    rises: list[float] = []
    falls: list[float] = []
    for i in range(1, n_s - 1):
        d0 = float(deriv[i])
        dm = float(deriv[i - 1])
        dp = float(deriv[i + 1])
        if d0 >= dm and d0 > dp and d0 > thr:
            rises.append(float(s[i]) + _parabolic_shift(dm, d0, dp) * ds)
        elif d0 <= dm and d0 < dp and -d0 > thr:
            falls.append(float(s[i]) + _parabolic_shift(dm, d0, dp) * ds)
    return rises, falls


def _ring_count(rises: list[float], falls: list[float]) -> int:
    """Nested ring count along one ray: bore starts below the stack."""
    if not rises or not falls:
        return 0
    g_env = max(rises)
    return sum(1 for f in falls if f < g_env)


def _tube_seeds_from_edges(
    rises: list[float],
    falls: list[float],
    r_zeta: float,
    n_nested: int,
) -> list[tuple[float, float, float, float]] | None:
    """Per-ring tube seeds ``(rho, bore, outer, t_bar)`` from ray edges.

    Outer-tangent ray kinematics: along a tube azimuth the
    core-facing wall of ring ``i`` appears as the ridge
    ``[r_out_i, r_in_i]`` whose end ``r_in_i`` is a falling edge
    (bore start ``f_i``); for ``i >= 2`` the ridge start ``r_out_i``
    is a rising edge, while ring 1's rising edge is invisible
    because the central core is tangent-merged with its wall.  The
    last rise ``g_env`` opens the envelope wall stack, so
    ``t_sum = R_zeta - g_env`` is the total ring thickness; ring 1
    is recovered as the residual ``t_1 = t_sum - sum(t_i, i >= 2)``
    with ``r_out_1 = f_1 - t_1``.  Anchors follow the outer-tangent
    stacking rule ``anchor_i = R_zeta - sum(t_m, m < i)``:
    ``delta_i = anchor_i - r_out_i = 2 R_outer_i``,
    ``rho_i = anchor_i - delta_i / 2`` and ``bore_i = outer_i - t_i``.
    """
    if not rises or not falls or n_nested < 1:
        return None
    g_env = float(max(rises))
    fs = sorted(f for f in falls if f < g_env)
    rs = sorted(r for r in rises if r < g_env)
    if len(fs) < n_nested or len(rs) < n_nested - 1:
        return None
    fs = fs[:n_nested]
    t = [0.0] * n_nested
    r_out = [0.0] * n_nested
    for i in range(1, n_nested):
        cand = [r for r in rs if fs[i - 1] < r < fs[i]]
        if not cand:
            return None
        r_out[i] = cand[-1]
        t[i] = fs[i] - cand[-1]
    t_sum = float(r_zeta) - g_env
    t[0] = t_sum - sum(t[1:])
    r_out[0] = fs[0] - t[0]
    if min(t) <= 1.0:
        return None
    seeds: list[tuple[float, float, float, float]] = []
    anchor = float(r_zeta)
    for i in range(n_nested):
        if i > 0:
            anchor -= t[i - 1]
        delta = anchor - r_out[i]
        outer = 0.5 * delta
        bore = outer - t[i]
        rho = anchor - outer
        if bore <= 1.0 or outer <= bore + 0.5 or rho <= outer:
            return None
        seeds.append((rho, bore, outer, t[i]))
    return seeds


def _refine_tube(
    norm: np.ndarray,
    origin: tuple[float, float],
    tube_center: tuple[float, float],
    bore_seed: float,
    outer_seed: float,
    t_bar: float,
    ring: int,
    azimuth_index: int,
) -> tuple[MeasuredCircle, MeasuredCircle] | None:
    """Refine one tube's bore and outer circles in its local frame.

    Local angle ``alpha = 0`` points away from the cross-section
    center — the outer contact stem where the tube is tangent to
    the jacket / parent ring; the free arc ``[80, 280]`` deg (the
    core-facing +/- 100 deg) stays clear of that stem.  Windows
    are ``+/- max(5, 0.4 t_bar)`` around the seeds.
    """
    cx, cy = tube_center
    base = math.atan2(cy - origin[1], cx - origin[0])
    alphas = math.radians(80.0) + np.linspace(
        0.0, math.radians(200.0), 72
    )
    th = base + alphas
    dirs = np.stack((np.cos(th), np.sin(th)), axis=1)
    win = max(5.0, 0.4 * max(t_bar, 1.0))
    bore_pts = _edge_points_along(
        norm, (cx, cy), dirs, bore_seed - win, bore_seed + win, +1
    )
    outer_pts = _edge_points_along(
        norm, (cx, cy), dirs, outer_seed - win, outer_seed + win, -1
    )
    if bore_pts.shape[0] < 8 or outer_pts.shape[0] < 8:
        return None
    bxc, byc, brc = fit_circle_robust(bore_pts)
    oxc, oyc, orc = fit_circle_robust(outer_pts)
    if not (1.0 < brc < orc and orc < 4.0 * brc):
        return None
    xc = 0.5 * (bxc + oxc)
    yc = 0.5 * (byc + oyc)
    return (
        MeasuredCircle("ring_bore", ring, azimuth_index, xc, yc, brc),
        MeasuredCircle("ring_outer", ring, azimuth_index, xc, yc, orc),
    )


def estimate_cross_section(
    image: np.ndarray,
    polarity: Polarity | None = None,
    pixel_size_m: float | None = None,
    n_cap_hint: int | None = None,
    nested_hint: int | None = None,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> CrossSectionEstimate:
    """Estimate a cane / fiber cross-section from a greyscale image.

    Pipeline: banner masking -> percentile normalization ->
    threshold centroid -> azimuthal-median polar profile (OD and
    zeta seeds) -> robust OD / zeta circle fits -> probe-ring
    angular FFT for ``n_cap`` and ``theta0`` -> per-azimuth radial
    edge scans seeding each nested tube via outer-tangent ray
    kinematics -> sector-refined bore / outer circle fits -> ring
    statistics.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or min(img.shape) < 64:
        raise ValueError("estimate_cross_section needs a >= 64x64 image")
    if polarity is None:
        polarity = Polarity.BRIGHT_SILICA
    warnings: list[str] = []
    banner = detect_sem_banner_row(img)
    if banner is not None:
        b0, b1 = banner
        if b0 == 0:
            content = img[b1:]
            y_off = int(b1)
        else:
            content = img[:b0]
            y_off = 0
    else:
        content = img
        y_off = 0
        warnings.append("no metadata banner detected")
    norm = normalize_sem_image(content, polarity, p_low, p_high)
    h, w = norm.shape

    mask = norm > 0.5
    if int(mask.sum()) < 200:
        raise ValueError("no silica structure found above 0.5 level")
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())

    r_max = max(8.0, min(cx, cy, w - cx, h - cy) - 2.0)
    rr = np.arange(0.0, r_max, 0.5)
    th = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    grid_r, grid_t = np.meshgrid(rr, th)
    xs_p = cx + grid_r * np.cos(grid_t)
    ys_p = cy + grid_r * np.sin(grid_t)
    polar = map_coordinates(
        norm,
        [ys_p.ravel(), xs_p.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(th.size, rr.size)
    prof = np.median(polar, axis=0)
    deriv = np.gradient(prof, rr)

    bright = rr[prof > 0.5]
    r_sil = float(bright[-1]) if bright.size else float(r_max - 2.0)
    lo_od = 0.55 * r_sil
    hi_od = min(r_max - 1.0, 1.06 * r_sil)
    sel = (rr >= lo_od) & (rr <= hi_od)
    if not sel.any():
        raise ValueError("cannot locate the outer diameter edge")
    r_od = float(rr[sel][int(np.argmin(deriv[sel]))])
    sel = (rr >= 0.45 * r_od) & (rr <= r_od - 5.0)
    if not sel.any():
        raise ValueError("cannot locate the zeta envelope edge")
    r_zeta = float(rr[sel][int(np.argmax(deriv[sel]))])

    od_xc, od_yc, od_r = _fit_edge_circle(norm, (cx, cy), r_od, -1)
    ze_xc, ze_yc, ze_r = _fit_edge_circle(
        norm, (od_xc, od_yc), r_zeta, +1
    )
    circles = [
        MeasuredCircle("od", 0, 0, od_xc, od_yc + y_off, od_r),
        MeasuredCircle("zeta", 0, 0, ze_xc, ze_yc + y_off, ze_r),
    ]

    r_probe = ze_r - max(6.0, 0.02 * ze_r)
    radii = np.arange(
        max(r_probe - 2.5, 1.0), r_probe + 2.5 + 1e-9, 0.5
    )
    th2 = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    grid_r2, grid_t2 = np.meshgrid(radii, th2)
    xs2 = ze_xc + grid_r2 * np.cos(grid_t2)
    ys2 = ze_yc + grid_r2 * np.sin(grid_t2)
    annulus = map_coordinates(
        norm, [ys2.ravel(), xs2.ravel()], order=1, mode="constant", cval=0.0
    ).reshape(th2.size, radii.size)
    angular = annulus.mean(axis=1)
    n_cap, theta0 = _count_and_phase(angular, th2, n_cap_hint)

    azimuths = [
        theta0 + k * 2.0 * math.pi / n_cap for k in range(n_cap)
    ]
    counts: list[int] = []
    scans: list[tuple[list[float], list[float]]] = []
    for azimuth in azimuths:
        rises, falls = _radial_edge_scan(
            norm, (ze_xc, ze_yc), azimuth, ze_r
        )
        scans.append((rises, falls))
        counts.append(_ring_count(rises, falls))
    if nested_hint:
        n_nested = int(nested_hint)
    else:
        n_nested = max(1, int(np.median(counts))) if counts else 1

    bores: list[list[float]] = [[] for _ in range(n_nested)]
    outers: list[list[float]] = [[] for _ in range(n_nested)]
    for k, (azimuth, (rises, falls)) in enumerate(
        zip(azimuths, scans, strict=True)
    ):
        seeds = _tube_seeds_from_edges(rises, falls, ze_r, n_nested)
        if seeds is None:
            continue
        for i, (rho, bore, outer, t_bar) in enumerate(seeds):
            tcx = ze_xc + rho * math.cos(azimuth)
            tcy = ze_yc + rho * math.sin(azimuth)
            fitted = _refine_tube(
                norm,
                (ze_xc, ze_yc),
                (tcx, tcy),
                bore,
                outer,
                t_bar,
                i + 1,
                k,
            )
            if fitted is None:
                continue
            bore_c, outer_c = fitted
            bores[i].append(bore_c.r)
            outers[i].append(outer_c.r)
            circles.append(
                MeasuredCircle(
                    bore_c.kind, bore_c.ring, k,
                    bore_c.xc, bore_c.yc + y_off, bore_c.r,
                )
            )
            circles.append(
                MeasuredCircle(
                    outer_c.kind, outer_c.ring, k,
                    outer_c.xc, outer_c.yc + y_off, outer_c.r,
                )
            )

    ring_bore: list[float] = []
    ring_od: list[float] = []
    ring_wall: list[float] = []
    for i in range(n_nested):
        if len(bores[i]) < 2:
            warnings.append(f"ring {i + 1}: only {len(bores[i])} tubes")
        if bores[i] and outers[i]:
            ring_bore.append(2.0 * float(np.mean(bores[i])))
            ring_od.append(2.0 * float(np.mean(outers[i])))
            walls = [
                o - b for o, b in zip(outers[i], bores[i], strict=False)
            ]
            ring_wall.append(float(np.mean(walls)))
        else:
            warnings.append(f"ring {i + 1}: no tube fitted")

    d_core = 2.0 * ze_r - 2.0 * ring_od[0] if ring_od else math.nan
    return CrossSectionEstimate(
        image_size_px=(int(img.shape[0]), int(img.shape[1])),
        pixel_size_m=pixel_size_m,
        polarity=polarity,
        banner_crop_rows=banner,
        center_px=(float(ze_xc), float(ze_yc + y_off)),
        od_px=float(2.0 * od_r),
        zeta_px=float(2.0 * ze_r),
        d_core_px=float(d_core),
        n_cap=int(n_cap),
        theta0_rad=float(theta0),
        ring_od_px=tuple(ring_od),
        ring_bore_px=tuple(ring_bore),
        ring_wall_px=tuple(ring_wall),
        circles=tuple(circles),
        warnings=tuple(warnings),
    )
