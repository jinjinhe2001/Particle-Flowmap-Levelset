"""Renderer tests (CPU device; media writers are skipped when their backends are missing)."""

import os
import sys

import numpy as np
import pytest

from pfmls.render import MediaWriter, WaterRenderer, WaterStyle


def _drop_field(n=100):
    h = 1.0 / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.minimum(Y - 0.3, np.hypot(X - 0.5, Y - 0.7) - 0.1)


def test_render_shape_and_colours():
    r = WaterRenderer(1.0, 1.0, height_px=160, style=WaterStyle(speed_tint=0.5))
    phi = _drop_field()
    img = r.render(phi, t=0.25, speed=np.ones((100, 100)), label="test")
    assert img.shape == (160, 160, 3) and img.dtype == np.uint8
    bg = img[30, 80].astype(int)       # air above the drop
    pool = img[150, 80].astype(int)    # deep liquid
    drop = img[48, 80].astype(int)     # inside the drop (y = 0.7)
    assert bg.sum() > pool.sum() + 150
    assert drop[2] > drop[0] + 40       # bluish


def test_render_wide_domain_and_nan():
    r = WaterRenderer(2.0, 1.0, height_px=96)
    assert r.W == 192 and r.H == 96
    phi = np.full((40, 20), np.nan)
    img = r.render(phi)
    assert np.isfinite(img).all()


def test_gif_writer(tmp_path):
    pytest.importorskip("PIL")
    r = WaterRenderer(1.0, 1.0, height_px=64)
    w = MediaWriter(str(tmp_path / "anim"), fps=10, mp4=False, gif=True, gif_every=1, gif_scale=1.0)
    for k in range(3):
        w.add(r.render(_drop_field() + 0.01 * k, t=0.1 * k))
    out = w.close()
    assert out and os.path.getsize(out[0]) > 0


def test_dense_particles_are_thinned():
    """A lattice of particles denser than the pixel grid is drawn as a sparse random subset, not a solid lattice."""
    r = WaterRenderer(1.0, 1.0, height_px=160)
    g = (np.arange(400) + 0.5) / 400
    X, Y = np.meshgrid(g, g[g < 0.3], indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    phi = _drop_field()
    base = r.render(phi).astype(int)
    img = r.render(phi, particles=pts).astype(int)
    pool = (slice(120, 155), slice(10, 150))
    changed = np.any(img[pool] != base[pool], axis=-1).mean()
    assert 0.01 < changed < 0.3  # unthinned: every pixel


def test_device_render_matches_host():
    """A device level set (rendered in place) gives the same image as the numpy array (rendered on the CPU)."""
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("CUDA device required")
    r = WaterRenderer(1.0, 1.0, height_px=128, style=WaterStyle(speed_tint=0.4))
    phi = _drop_field()
    speed = np.linspace(0.0, 1.5, 100 * 100).reshape(100, 100)
    host = r.render(phi, speed=speed).astype(int)
    dev = r.render(wp.array(phi.astype(np.float32), device="cuda:0"),
                   speed=wp.array(speed.astype(np.float32), device="cuda:0")).astype(int)
    assert np.abs(host - dev).max() <= 1


def test_mp4_falls_back_to_png_without_ffmpeg(tmp_path, monkeypatch):
    """Without imageio-ffmpeg the writer warns, leaves no partial MP4 and writes PNG frames instead."""
    pytest.importorskip("PIL")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # import fails
    r = WaterRenderer(1.0, 1.0, height_px=32)
    with pytest.warns(RuntimeWarning, match="MP4 output disabled"):
        w = MediaWriter(str(tmp_path / "clip"), fps=10)
    for k in range(2):
        w.add(r.render(_drop_field() + 0.01 * k))
    out = w.close()
    assert not os.path.exists(tmp_path / "clip.mp4")
    assert out == [str(tmp_path / "clip_frames")] and len(os.listdir(tmp_path / "clip_frames")) == 2
