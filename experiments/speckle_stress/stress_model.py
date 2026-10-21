"""Synthetic multiplier on normalized observed images, not calibrated SAR speckle."""
import hashlib
import json
import numpy as np


def gamma_multiplier(shape, k, pair_id, seed, master_seed=2026101506):
    if not np.isfinite(k) or k <= 0 or not pair_id or int(seed) != seed or seed < 0:
        raise ValueError('positive finite k, nonempty pair_id and nonnegative integer seed required')
    payload = json.dumps([master_seed, str(pair_id), float(k), int(seed)], separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    stream_seed = int.from_bytes(digest[:16], 'little')
    rng = np.random.Generator(np.random.PCG64(stream_seed))
    return rng.gamma(shape=float(k), scale=1.0 / k, size=shape)


def apply_multiplier(image, multiplier):
    b = np.asarray(image, dtype=np.float64)
    g = np.asarray(multiplier, dtype=np.float64)
    if b.ndim != 2 or b.shape != g.shape or not np.isfinite(b).all() or not np.isfinite(g).all():
        raise ValueError('finite equally shaped 2D image and multiplier required')
    if b.min() < 0 or b.max() > 1 or g.min() < 0:
        raise ValueError('image must be in [0,1] and multiplier nonnegative')
    before = b * g
    after = np.clip(before, 0.0, 1.0).astype(np.float32)
    info = dict(multiplier_mean=float(g.mean()), multiplier_variance=float(g.var()),
                multiplier_min=float(g.min()), multiplier_max=float(g.max()),
                input_mean=float(b.mean()), pre_clip_mean=float(before.mean()),
                post_clip_mean=float(after.astype(np.float64).mean()),
                upper_clipped_fraction=float(np.mean(before > 1)),
                clipped_mass_mean=float(np.mean(np.maximum(before - 1, 0))),
                perturbed_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
                multiplier_sha256=hashlib.sha256(g.tobytes()).hexdigest())
    return after, info


def perturb(image, k, pair_id, seed, master_seed=2026101506):
    if k is None:
        g = np.ones(np.asarray(image).shape, dtype=np.float64)
    else:
        g = gamma_multiplier(np.asarray(image).shape, k, pair_id, seed, master_seed)
    return apply_multiplier(image, g)
