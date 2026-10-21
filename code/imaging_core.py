"""Auditable float features for the v3 imaging experiments.

No legacy ``tool`` modules are imported. All scalar feature maps have shape H x W,
float32 dtype, and range [0, 1]. Input images must already be finite scalar
float images in [0, 1], normally obtained with :func:`read_image`.

``mfpc`` preserves the original *algebraic* definition: replace the wavelet
approximation by raw phase-congruency maximum moment, retain signed detail,
reconstruct, clip to [0, 1], and apply final min-max normalization. Integer
quantization and an unused HH computation from the old implementation are
removed deliberately. This is not byte-for-byte reproduction of the old code.

``mfpc_scale_matched`` is a separately named experimental control. Its details
are multiplied by RMS(PC(approximation))/RMS(approximation), preserving the
original detail/approximation RMS ratio before reconstruction. It must never be
reported as the original MFPC. ``detail_scale`` is a further explicit multiplier.

All wavelet variants use signed floating coefficients, symmetric extension, and
one common clip/min-max output rule. No uint8 rendering, saturation or wraparound
is performed before PC. The directional baseline is PC orientation channels with
local pooling and channel L2 normalization, inspired by HOPC but NOT a reproduction
of the published HOPC descriptor or registration pipeline.
"""
from __future__ import annotations

import hashlib
import json
import operator
from pathlib import Path
import warnings

import cv2
import numpy as np
import pywt
from scipy.ndimage import uniform_filter
import tifffile

PC_DEFAULTS = {
    "nscale": 5, "norient": 6, "minWaveLength": 3, "mult": 2.1,
    "sigmaOnf": 0.55, "k": 2.0, "cutOff": 0.5, "g": 10.0,
    "noiseMethod": -1,
}
PC_PARAMETER_ALIASES = {
    "scale": "nscale", "orient": "norient", "min_wavelength": "minWaveLength",
    "sigma_onf": "sigmaOnf", "cut_off": "cutOff", "noise_method": "noiseMethod",
}
RGB_WEIGHTS = (0.299, 0.587, 0.114)  # Explicit RGB order, matching old grayscale weights.
WAVELET_MODE = "symmetric"
STRATEGY_DESCRIPTIONS = {
    "raw": "Normalized input intensity; no feature extraction.",
    "full_pc": "Full-resolution raw PC maximum moment, then clip/min-max.",
    "down_pc": "Half-resolution area resize, raw PC, linear upsample, clip/min-max.",
    "mfpc": "Original MFPC algebra on unit-scale floats; original signed detail retained.",
    "mfpc_scale_matched": "Control: detail gain RMS(PC(A))/RMS(A); not original MFPC.",
    "no_detail": "MFPC with all detail coefficients set to zero.",
    "gated": "MFPC with each signed detail multiplied by its raw PC maximum moment.",
    "lowfreq": "Raw PC of approximation, resized directly to original size; no inverse DWT.",
    "all_subbands": "Raw PC of approximation and every signed detail subband, then inverse DWT.",
    "denoise": "Soft-threshold signed detail, reconstruct intensity, then full-resolution PC.",
    "noise": "Independent zero-mean Gaussian detail, each realization exactly RMS-matched.",
    "surrogate": "Independent random-phase detail preserving each subband amplitude spectrum.",
    "pc_orientation_channels": "HOPC-inspired pooled PC orientation channels; not original HOPC.",
}
STRATEGIES = tuple(STRATEGY_DESCRIPTIONS)
SCALAR_STRATEGIES = tuple(s for s in STRATEGIES if s != "pc_orientation_channels")
STOCHASTIC_STRATEGIES = ("noise", "surrogate")
# Compatibility alias only; metadata and figures should use the descriptive name.
STRATEGY_ALIASES = {"hopc_directional": "pc_orientation_channels"}


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize a finite array to float32 [0,1]; constant arrays become zero."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Image must contain finite, nonempty data")
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _to_hwc(array: np.ndarray, axes: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected one 2D image with optional channels, got {arr.shape} ({axes})")
    if len(axes) == 3 and axes[:2] == "YX" and axes[2] in "SC":
        return arr
    if len(axes) == 3 and axes[0] in "SC" and axes[1:] == "YX":
        return np.moveaxis(arr, 0, -1)
    raise ValueError(f"Ambiguous channel axes {axes!r}; cannot infer spectral order")


def read_image(path, modality="gray", return_info=False):
    """Read raw samples and select spectral channels explicitly before normalization.

    TIFF is always read with tifffile, including MINISBLACK four-sample RGBIR.
    ``nir`` requires exactly four bands and selects zero-based band 3. The dataset
    manifest must establish that the source order is RGB,NIR; TIFF shape alone
    cannot establish wavelength identity. ``rgb`` explicitly asserts RGB order,
    including Potsdam files with incomplete photometric metadata. ``gray`` and
    ``label`` convert multichannel data only when metadata says RGB. RGB label
    colors are treated as a documented intensity encoding, NOT semantic ordinals.
    PNG and other ordinary raster formats use Pillow, never cv2 image decoding.
    """
    path = Path(path)
    if modality not in {"gray", "rgb", "nir", "label"}:
        raise ValueError("modality must be gray, rgb, nir, or label")
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tf:
            if len(tf.series) != 1:
                raise ValueError("Multiple TIFF series require an explicit series selection")
            series = tf.series[0]
            axes = series.axes
            raw = series.asarray()
            photo = tf.pages[0].photometric.name
        arr = _to_hwc(raw, axes)
        decoder = "tifffile"
        metadata_rgb = photo == "RGB"
    else:
        from PIL import Image
        with Image.open(path) as im:
            if im.mode == "P":
                im = im.convert("RGB")
            if im.mode not in {"L", "I", "F", "I;16", "RGB", "RGBA", "1"}:
                raise ValueError(f"Unsupported raster mode {im.mode!r}")
            photo = im.mode
            arr = np.asarray(im)
            raw = arr
        axes = "YX" if arr.ndim == 2 else "YXS"
        decoder = "Pillow"
        metadata_rgb = photo in {"RGB", "RGBA"}
    if not np.all(np.isfinite(arr)):
        raise ValueError("Source image contains NaN or infinity")
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    if modality == "nir":
        if arr.ndim != 3 or channels != 4:
            raise ValueError(f"NIR requires an explicit four-band RGBIR image; got {arr.shape}")
        selected = arr[..., 3].astype(np.float64)
        chosen = [3]
        semantic = "NIR (manifest assertion: source bands are R,G,B,NIR)"
    elif arr.ndim == 2:
        if modality == "rgb":
            raise ValueError("RGB modality requires at least three channels")
        selected = arr.astype(np.float64)
        chosen = [0]
        semantic = "single scalar channel"
    else:
        if channels not in (3, 4):
            raise ValueError(f"RGB conversion requires 3 or 4 channels, got {channels}")
        if modality != "rgb" and not metadata_rgb:
            raise ValueError("Multichannel file lacks RGB metadata; use explicit rgb or nir modality")
        selected = np.einsum("...c,c->...", arr[..., :3].astype(np.float64), RGB_WEIGHTS)
        chosen = [0, 1, 2]
        semantic = "RGB luminance encoding" if modality != "label" else "RGB label-color luminance encoding"
    lo, hi = float(selected.min()), float(selected.max())
    image = normalize_image(selected)
    info = {
        "source_path": str(path.resolve()), "decoder": decoder,
        "source_shape": list(raw.shape), "source_dtype": str(raw.dtype),
        "source_axes": axes, "photometric": photo, "modality": modality,
        "selected_channels_zero_based": chosen, "channel_semantics": semantic,
        "rgb_weights": list(RGB_WEIGHTS) if len(chosen) == 3 else None,
        "normalization": "per-image min-max after channel selection; constant image -> zero",
        "selected_min": lo, "selected_max": hi, "constant": hi == lo,
        "normalized_sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
    }
    return (image, info) if return_info else image


def _image(image):
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or min(arr.shape, default=0) < 2:
        raise ValueError("Feature extraction expects an H x W scalar image, each dimension >= 2")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or infinity")
    if arr.min() < -1e-7 or arr.max() > 1 + 1e-7:
        raise ValueError("Input must be normalized to [0,1]; call normalize_image/read_image first")
    return np.clip(arr, 0.0, 1.0)


def _pc_parameters(overrides):
    result = dict(PC_DEFAULTS)
    supplied = {} if overrides is None else dict(overrides)
    for name, value in supplied.items():
        key = PC_PARAMETER_ALIASES.get(name, name)
        if key not in result:
            raise ValueError(f"Unknown PC parameter {name!r}")
        if key in supplied and key != name:
            raise ValueError(f"Duplicate PC parameter aliases {name!r}, {key!r}")
        result[key] = value
    return result


def _phase_result(array, parameters):
    # Lazy import keeps the core importable without executing any old experiment.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"(?s).*pyfftw.*", category=UserWarning)
        from phasepack.phasecong import phasecong
    return phasecong(np.ascontiguousarray(array, dtype=np.float64), **parameters)


def _pc(array, parameters):
    arr = np.asarray(array, dtype=np.float64)
    if np.ptp(arr) <= 1e-14:
        return np.zeros(arr.shape, dtype=np.float64)
    result = np.asarray(_phase_result(arr, parameters)[0], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("phasepack returned non-finite maximum moments")
    return np.maximum(result, 0.0)


def _finish(array, shape):
    cropped = np.asarray(array, dtype=np.float64)[:shape[0], :shape[1]]
    if cropped.shape != shape or not np.all(np.isfinite(cropped)):
        raise FloatingPointError("Reconstruction has invalid shape or non-finite values")
    clipped = np.clip(cropped, 0.0, 1.0)
    # Do not turn wavelet round-off on a constant image into a full-contrast map.
    if float(np.ptp(clipped)) <= 64 * np.finfo(np.float64).eps:
        return np.zeros(shape, dtype=np.float32)
    return normalize_image(clipped)


def rms(array):
    arr = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def independent_rng(seed, image_key, strategy, level_index, band_index):
    """Stable random stream keyed by image identity, independent of call order."""
    if image_key is None or str(image_key) == "":
        raise ValueError("Stochastic strategies require a distinct nonempty image_key for each image")
    try:
        seed = operator.index(seed)
    except TypeError as error:
        raise ValueError("seed must be an integer") from error
    payload = json.dumps([seed, str(image_key), strategy, int(level_index), int(band_index)],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    entropy = np.frombuffer(hashlib.sha256(payload).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(entropy.tolist()))


def matched_noise(detail, rng):
    """A zero-mean Gaussian draw rescaled to the exact RMS of this detail array."""
    detail = np.asarray(detail, dtype=np.float64)
    target = rms(detail)
    if target == 0:
        return np.zeros_like(detail)
    noise = rng.normal(size=detail.shape)
    noise -= noise.mean()
    magnitude = rms(noise)
    if magnitude == 0:
        raise ValueError("Detail array is too small for zero-mean matched noise")
    return noise * (target / magnitude)


def phase_surrogate(detail, rng):
    """Random phases from a real white field guarantee conjugate symmetry.

    The amplitude of *every* Fourier coefficient and the signed DC coefficient
    are preserved. The inverse FFT therefore preserves energy to float precision.
    This operation is independent between image_key streams. It does not claim
    that all possible spatial structures are removed from every special input.
    """
    detail = np.asarray(detail, dtype=np.float64)
    spectrum = np.fft.rfft2(detail)
    random_spectrum = np.fft.rfft2(rng.normal(size=detail.shape))
    magnitude = np.abs(random_spectrum)
    phase = np.divide(random_spectrum, magnitude, out=np.ones_like(random_spectrum),
                      where=magnitude > 0)
    randomized = np.abs(spectrum) * phase
    randomized[0, 0] = spectrum[0, 0]
    return np.fft.irfft2(randomized, s=detail.shape)


def extract_feature(image, strategy, *, wavelet="db2", level=1, seed=0,
                    image_key=None, detail_scale=1.0, pc_kwargs=None,
                    orientation_pool=8):
    """Extract a named strategy, with one scalar or orientation-channel output.

    The primary study uses level=1. Greater levels are explicitly exploratory.
    At every level, all signed details are retained/substituted consistently.
    PC is the phasepack maximum covariance moment, with PC_DEFAULTS unless
    explicitly overridden. Stochastic strategies require image_key, e.g.
    ``pair_001:optical`` and ``pair_001:sar``; seed indexes realizations.
    """
    image = _image(image)
    strategy = STRATEGY_ALIASES.get(strategy, strategy)
    if strategy not in STRATEGY_DESCRIPTIONS:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if not np.isfinite(detail_scale) or detail_scale < 0:
        raise ValueError("detail_scale must be finite and nonnegative")
    try:
        level = operator.index(level)
    except TypeError as error:
        raise ValueError("level must be a positive integer") from error
    if level < 1:
        raise ValueError("level must be a positive integer")
    if strategy in STOCHASTIC_STRATEGIES and (image_key is None or str(image_key) == ""):
        raise ValueError("Stochastic strategies require a distinct image_key per image")
    parameters = _pc_parameters(pc_kwargs)
    if strategy == "raw":
        return image.astype(np.float32)
    if strategy == "full_pc":
        return _finish(_pc(image, parameters), image.shape)
    if strategy == "down_pc":
        target = (max(2, image.shape[1] // 2), max(2, image.shape[0] // 2))
        low = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        low_pc = _pc(low, parameters)
        result = cv2.resize(low_pc, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return _finish(result, image.shape)
    if strategy == "pc_orientation_channels":
        try:
            pool = operator.index(orientation_pool)
        except TypeError as error:
            raise ValueError("orientation_pool must be a positive integer") from error
        if pool < 1:
            raise ValueError("orientation_pool must be a positive integer")
        if np.ptp(image) <= 1e-14:
            return np.zeros((*image.shape, parameters["norient"]), dtype=np.float32)
        directional = np.stack(_phase_result(image, parameters)[4], axis=-1)
        if not np.all(np.isfinite(directional)):
            raise FloatingPointError("phasepack returned non-finite orientation channels")
        directional = np.maximum(directional, 0.0)
        pooled = np.maximum(uniform_filter(directional, size=(pool, pool, 1), mode="reflect"), 0.0)
        norm = np.sqrt(np.sum(pooled * pooled, axis=-1, keepdims=True))
        return np.divide(pooled, norm, out=np.zeros_like(pooled), where=norm > 1e-12).astype(np.float32)
    wl = pywt.Wavelet(wavelet)
    max_level = pywt.dwtn_max_level(image.shape, wl)
    if level > max_level:
        raise ValueError(f"level={level} exceeds boundary-safe maximum {max_level} for {image.shape}, {wavelet}")
    coeffs = pywt.wavedec2(image, wl, mode=WAVELET_MODE, level=level)
    approximation = coeffs[0]
    if strategy == "denoise":
        # Finest horizontal detail defines the same universal threshold rule as
        # the old experiment, without artificial sigma=1 for a zero-noise image.
        finest_h = coeffs[-1][0]
        sigma = float(np.median(np.abs(finest_h))) / 0.6745
        threshold = sigma * np.sqrt(2.0 * np.log(max(2, finest_h.size)))
        details = [tuple(np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0) * detail_scale
                         for d in group) for group in coeffs[1:]]
        denoised = pywt.waverec2([approximation, *details], wl, mode=WAVELET_MODE)
        denoised = np.clip(denoised[:image.shape[0], :image.shape[1]], 0.0, 1.0)
        return _finish(_pc(denoised, parameters), image.shape)
    pc_a = _pc(approximation, parameters)
    if strategy == "lowfreq":
        result = cv2.resize(pc_a, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return _finish(result, image.shape)
    gain = detail_scale
    if strategy == "mfpc_scale_matched":
        approx_rms = rms(approximation)
        gain *= rms(pc_a) / approx_rms if approx_rms > 0 else 0.0
    details = []
    for level_index, group in enumerate(coeffs[1:], start=1):
        transformed = []
        for band_index, detail in enumerate(group):
            if strategy == "no_detail":
                out = np.zeros_like(detail)
            elif strategy == "gated":
                out = detail * _pc(detail, parameters)
            elif strategy == "all_subbands":
                out = _pc(detail, parameters)
            elif strategy in STOCHASTIC_STRATEGIES:
                rng = independent_rng(seed, image_key, strategy, level_index, band_index)
                out = matched_noise(detail, rng) if strategy == "noise" else phase_surrogate(detail, rng)
            else:
                out = detail
            transformed.append(out * gain)
        details.append(tuple(transformed))
    result = pywt.waverec2([pc_a, *details], wl, mode=WAVELET_MODE)
    return _finish(result, image.shape)
