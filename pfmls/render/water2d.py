"""Renderer for 2D liquid level sets (Warp kernels); the MP4/GIF/PNG writers are in :mod:`pfmls.render.media`.

The liquid is drawn directly from a (preferably fine) cell-centred level set: anti-aliased fill from the
signed distance in pixel units, depth shading (``-phi`` measures the distance below the surface, box-blurred to hide
the staircases of the redistanced far field), a thin inner highlight, a darker outline, an optional
velocity-magnitude tint, optional particles and a time stamp.  The per-pixel work runs as three Warp kernels on the
device that holds the level set (device arrays from the solver are used in place; numpy inputs run on the CPU
device), and only the finished 8-bit image is copied to the host, so a frame costs a few milliseconds on a GPU.
Particles and text are drawn on the host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import warp as wp


def _hash01(i: np.ndarray) -> np.ndarray:
    """Pseudo-random numbers in [0, 1) from integer indices (murmur3 finalizer; uncorrelated for neighbours)."""
    x = np.asarray(i, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(13)
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    return x.astype(np.float64) / 2.0 ** 32


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


@dataclass
class WaterStyle:
    """Colours and sizes (hex colours; sizes in output pixels unless noted)."""

    bg_top: str = "#f7f9fc"
    bg_bottom: str = "#dfe6ef"
    liquid_surface: str = "#5aa0e0"
    liquid_deep: str = "#1b4c8c"
    outline: str = "#12345e"
    foam: str = "#e8f4ff"
    wall: str = "#4a5563"
    text: str = "#1f2933"
    particle: str = "#f2a541"
    depth_scale: float = 0.18      # domain units over which the colour goes from surface to deep
    depth_blur_px: int = 6         # smoothing of the depth used for the colour (hides eikonal staircases)
    outline_px: float = 1.6
    highlight: float = 0.18        # strength of the bright band just inside the surface
    highlight_px: tuple[float, float] = (1.5, 5.0)
    speed_tint: float = 0.0        # 0..1 blend towards ``foam`` at |u| >= speed_scale
    speed_scale: float = 1.0
    wall_px: int = 3
    particle_px: int = 2           # dot size at a 720 px frame height (scaled with the frame)
    particle_spacing_px: float = 5.0  # denser particle sets are thinned to this mean spacing (0 = draw all)
    font_px: int = 22              # at a 720 px frame height (scaled with the frame)

    @classmethod
    def dark(cls, **kw) -> "WaterStyle":
        base = dict(bg_top="#0a1626", bg_bottom="#152c48", liquid_surface="#57c1ff", liquid_deep="#0e5aa7",
                    outline="#bfe8ff", foam="#ffffff", wall="#8a9bb0", text="#e6eef7")
        base.update(kw)
        return cls(**base)


# --------------------------------------------------------------------------------------------- kernels
@wp.func
def _sample_cells(f: wp.array2d(dtype=float), s: float, t: float) -> float:
    """Bilinear sample of a cell-centred field at continuous cell coordinates (s, t) (clamped)."""
    m = f.shape[0]
    n = f.shape[1]
    s = wp.clamp(s, 0.0, float(m - 1))
    t = wp.clamp(t, 0.0, float(n - 1))
    i0 = wp.min(int(s), wp.max(m - 2, 0))
    j0 = wp.min(int(t), wp.max(n - 2, 0))
    i1 = wp.min(i0 + 1, m - 1)
    j1 = wp.min(j0 + 1, n - 1)
    fx = s - float(i0)
    fy = t - float(j0)
    a = f[i0, j0] * (1.0 - fx) + f[i1, j0] * fx
    b = f[i0, j1] * (1.0 - fx) + f[i1, j1] * fx
    return a * (1.0 - fy) + b * fy


@wp.func
def _finite(x: float, nan_value: float) -> float:
    if wp.isnan(x):
        return nan_value
    if not wp.isfinite(x):
        return wp.sign(x)
    return x


@wp.kernel
def sample_phi_kernel(phi: wp.array2d(dtype=float), lx: float, ly: float, px: float, pxx: float,
                      inv_depth: float, ph_out: wp.array2d(dtype=float), depth: wp.array2d(dtype=float)):
    """Level set at the pixel centres (row 0 at the top) and the unblurred depth fraction."""
    r, c = wp.tid()
    s = (float(c) + 0.5) * pxx / (lx / float(phi.shape[0])) - 0.5
    t = (ly - (float(r) + 0.5) * px) / (ly / float(phi.shape[1])) - 0.5
    p = _finite(_sample_cells(phi, s, t), 1.0)
    ph_out[r, c] = p
    depth[r, c] = wp.clamp(-p * inv_depth, 0.0, 1.0)


@wp.kernel
def box_blur_kernel(src: wp.array2d(dtype=float), radius: int, axis: int, dst: wp.array2d(dtype=float)):
    """Box blur of radius ``radius`` along ``axis`` (edge-clamped)."""
    r, c = wp.tid()
    h = src.shape[0]
    w = src.shape[1]
    acc = float(0.0)
    for k in range(-radius, radius + 1):
        if axis == 0:
            acc += src[wp.clamp(r + k, 0, h - 1), c]
        else:
            acc += src[r, wp.clamp(c + k, 0, w - 1)]
    dst[r, c] = acc / float(2 * radius + 1)


@wp.func
def _mix(a: wp.vec3, b: wp.vec3, w: float) -> wp.vec3:
    return a + w * (b - a)


@wp.kernel
def shade_kernel(ph: wp.array2d(dtype=float), depth: wp.array2d(dtype=float), speed: wp.array2d(dtype=float),
                 use_speed: int, lx: float, ly: float, px: float, pxx: float, colors: wp.array(dtype=wp.vec3),
                 prm: wp.array(dtype=float), wall_px: int, img: wp.array3d(dtype=wp.uint8)):
    """Liquid colour (depth LUT, highlight, speed tint), coverage over the background gradient, outline, walls.

    colors: bg_top, bg_bottom, surface, deep, foam, outline, wall.  prm: px per domain unit (1/px), highlight,
    a0, a1, speed_tint, 1/speed_scale, outline_px.
    """
    r, c = wp.tid()
    hgt = img.shape[0]
    wid = img.shape[1]
    tb = float(r) / float(wp.max(hgt - 1, 1))
    bg = _mix(colors[0], colors[1], tb)
    p = ph[r, c]
    d_px = p * prm[0]
    k = float(int(wp.clamp(depth[r, c], 0.0, 1.0) * 255.0)) / 255.0
    col = _mix(colors[2], colors[3], wp.pow(k, 0.75))
    w = float(0.0)
    hl = prm[1]
    if hl > 0.0:
        w += hl * wp.clamp(1.0 - prm[2] - d_px, 0.0, 1.0) * wp.clamp(prm[3] + 1.0 + d_px, 0.0, 1.0)
    if use_speed == 1 and prm[4] > 0.0:
        s = (float(c) + 0.5) * pxx / (lx / float(speed.shape[0])) - 0.5
        t = (ly - (float(r) + 0.5) * px) / (ly / float(speed.shape[1])) - 0.5
        sp = wp.clamp(_finite(_sample_cells(speed, s, t), 0.0) * prm[5], 0.0, 1.0)
        w += prm[4] * sp * wp.sqrt(sp)
    col = _mix(col, colors[4], wp.clamp(w, 0.0, 1.0))
    out = _mix(bg, col, wp.clamp(0.5 - d_px, 0.0, 1.0))
    out = _mix(out, colors[5], wp.clamp(0.5 * prm[6] + 0.5 - wp.abs(d_px), 0.0, 1.0))
    if wall_px > 0 and (c < wall_px or c >= wid - wall_px or r >= hgt - wall_px):
        out = colors[6]
    for q in range(3):
        img[r, c, q] = wp.uint8(int(wp.clamp(out[q], 0.0, 1.0) * 255.0 + 0.5))


class WaterRenderer:
    """Render cell-centred level sets of the box ``[0, lx] x [0, ly]`` to RGB images.

    Args:
        lx, ly: domain size.
        height_px: output height; the width follows the aspect ratio.  Both are rounded to multiples of 16
            (friendly to H.264).
        style: a :class:`WaterStyle`.
    """

    def __init__(self, lx: float, ly: float, height_px: int = 720, style: WaterStyle | None = None):
        self.lx, self.ly = float(lx), float(ly)
        self.style = style or WaterStyle()
        self.H = max(16, int(round(height_px / 16)) * 16)
        self.W = max(16, int(round(self.H * self.lx / self.ly / 16)) * 16)
        self.px = self.ly / self.H                       # domain units per pixel (vertical)
        self.pxx = self.lx / self.W
        self._buffers: dict = {}
        self._font = None

    def _bufs(self, device):
        """Per-device work arrays (allocated on first use)."""
        key = str(device)
        b = self._buffers.get(key)
        if b is None:
            s = self.style
            shp = (self.H, self.W)
            cols = np.stack([_rgb(x) for x in (s.bg_top, s.bg_bottom, s.liquid_surface, s.liquid_deep, s.foam,
                                               s.outline, s.wall)])
            prm = np.array([1.0 / self.px, s.highlight, s.highlight_px[0], s.highlight_px[1], s.speed_tint,
                            1.0 / max(s.speed_scale, 1e-12), s.outline_px], dtype=np.float32)
            b = self._buffers[key] = {
                "ph": wp.zeros(shp, dtype=float, device=device), "d0": wp.zeros(shp, dtype=float, device=device),
                "d1": wp.zeros(shp, dtype=float, device=device),
                "img": wp.zeros((self.H, self.W, 3), dtype=wp.uint8, device=device),
                "colors": wp.array(cols, dtype=wp.vec3, device=device),
                "prm": wp.array(prm, dtype=float, device=device),
                "nospeed": wp.zeros((1, 1), dtype=float, device=device)}
        return b

    # ------------------------------------------------------------------------------------------- render
    def render(self, phi, t: float | None = None, speed=None, particles: np.ndarray | None = None,
               label: str | None = None) -> np.ndarray:
        """Return an ``(H, W, 3)`` uint8 image.

        Args:
            phi: cell-centred level set on any grid covering the domain (negative = liquid); a ``wp.array2d``
                (rendered on its device) or a numpy array (rendered on the CPU device).
            t: time printed in the corner (omitted if None).
            speed: optional cell-centred speed field for the foam tint (``wp.array2d`` on the same device, or
                numpy).
            particles: optional ``(N, 2)`` positions drawn as dots.
            label: optional text printed in the top-right corner.
        """
        s = self.style
        if not isinstance(phi, wp.array):
            phi = wp.array(np.ascontiguousarray(phi, dtype=np.float32), dtype=float, device="cpu")
        dev = phi.device
        b = self._bufs(dev)
        use_speed = int(speed is not None and s.speed_tint > 0)
        if use_speed and not isinstance(speed, wp.array):
            speed = wp.array(np.ascontiguousarray(speed, dtype=np.float32), dtype=float, device=dev)
        shp = (self.H, self.W)
        wp.launch(sample_phi_kernel, dim=shp, inputs=[phi, self.lx, self.ly, self.px, self.pxx,
                                                      1.0 / s.depth_scale], outputs=[b["ph"], b["d0"]], device=dev)
        depth = b["d0"]
        if s.depth_blur_px > 0:
            wp.launch(box_blur_kernel, dim=shp, inputs=[b["d0"], s.depth_blur_px, 1], outputs=[b["d1"]], device=dev)
            wp.launch(box_blur_kernel, dim=shp, inputs=[b["d1"], s.depth_blur_px, 0], outputs=[b["d0"]], device=dev)
        wp.launch(shade_kernel, dim=shp,
                  inputs=[b["ph"], depth, speed if use_speed else b["nospeed"], use_speed, self.lx, self.ly, self.px,
                          self.pxx, b["colors"], b["prm"], s.wall_px], outputs=[b["img"]], device=dev)
        out = b["img"].numpy().copy()
        if particles is not None and len(particles):
            self._draw_particles(out, particles)
        if t is not None or label:
            out = self._draw_text(out, None if t is None else f"t = {t:.3f}", label)
        return out

    def _draw_particles(self, img: np.ndarray, pts: np.ndarray) -> None:
        """Dots at the particle positions.  Particle sets denser than ``particle_spacing_px`` (e.g. several
        particles per pixel row on a regular lattice, which would alias into a moire pattern) are thinned to that
        mean spacing by a fixed pseudo-random subset of the particle indices."""
        s = self.style
        c = (pts[:, 0] / self.pxx).astype(np.int64)
        r = ((self.ly - pts[:, 1]) / self.px).astype(np.int64)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        idx = np.nonzero(ok)[0]
        c, r = c[ok], r[ok]
        if len(c) == 0:
            return
        if s.particle_spacing_px > 0:
            bs = max(1, int(round(s.particle_spacing_px)))
            occupied = len(np.unique((r // bs) * (self.W // bs + 1) + c // bs)) * bs * bs  # covered pixels
            frac = occupied / (len(c) * s.particle_spacing_px ** 2)
            if frac < 1.0:
                keep = _hash01(idx) < frac
                c, r = c[keep], r[keep]
        size = max(1, int(round(s.particle_px * self.H / 720)))
        col = _rgb(s.particle) * 255.0
        for dr in range(size):
            for dc in range(size):
                rr = np.clip(r + dr, 0, self.H - 1)
                cc = np.clip(c + dc, 0, self.W - 1)
                img[rr, cc] = (0.35 * img[rr, cc] + 0.65 * col + 0.5).astype(np.uint8)

    def _get_font(self):
        if self._font is None:
            from PIL import ImageFont

            path = None
            try:
                import matplotlib

                path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSansMono.ttf")
            except Exception:
                path = None
            size = max(10, int(round(self.style.font_px * self.H / 720)))  # font_px is for a 720 px frame
            try:
                self._font = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            except Exception:
                self._font = ImageFont.load_default()
        return self._font

    def _draw_text(self, img: np.ndarray, left: str | None, right: str | None) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return img
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        font = self._get_font()
        col = tuple(int(255 * c) for c in _rgb(self.style.text))
        pad = max(8, self.H // 60)
        x_left_end = pad + self.style.wall_px
        if left:
            dr.text((x_left_end, pad), left, fill=col, font=font)
            x_left_end += dr.textlength(left, font=font)
        if right:
            w = dr.textlength(right, font=font)
            x = self.W - pad - w - self.style.wall_px
            if x > x_left_end + pad:  # drop the label rather than overlap the time stamp
                dr.text((x, pad), right, fill=col, font=font)
        return np.asarray(im)
