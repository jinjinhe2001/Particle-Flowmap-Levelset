"""Video, GIF and PNG output of the water examples.

:class:`MediaWriter` streams RGB frames to an H.264 MP4 (imageio's FFMPEG plugin, ``imageio-ffmpeg``) and
optionally writes a looping GIF and PNG frames.  If the MP4 cannot be written (``imageio-ffmpeg`` missing, or a
frame fails), the writer warns, removes the partial file and writes PNG frames into ``<path_stem>_frames``
instead.  The GIF keeps every ``gif_every``-th frame, scaled by ``gif_scale`` and quantised by Pillow (median cut,
128 colours), which suits the flat shading of the 2D renderer.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

__all__ = ["MediaWriter", "save_png"]


def save_png(path: str, img: np.ndarray) -> None:
    """Write an RGB uint8 image as PNG (Pillow)."""
    from PIL import Image

    Image.fromarray(img).save(path)


class MediaWriter:
    """Stream RGB uint8 frames to ``<path_stem>.mp4``, optionally a looping ``<path_stem>.gif`` and PNG frames.

    Args:
        path_stem: output path without extension.
        fps: frames per second of the MP4 (and of the GIF, divided by ``gif_every``).
        mp4: write the MP4 (H.264, yuv420p, imageio quality ``quality``, 0-10).
        gif: write a GIF: every ``gif_every``-th frame scaled by ``gif_scale``.
        png_dir: if given, every frame is also written there as ``frame_%05d.png``.
    """

    def __init__(self, path_stem: str, fps: int = 30, mp4: bool = True, gif: bool = False, gif_scale: float = 0.5,
                 gif_every: int = 2, png_dir: str | None = None, quality: float = 8.0):
        self.stem, self.fps = path_stem, fps
        self.gif, self.gif_scale, self.gif_every = gif, gif_scale, gif_every
        self.png_dir = png_dir
        self.n = 0
        self._mp4 = None
        self._gif_frames: list = []
        os.makedirs(os.path.dirname(os.path.abspath(path_stem)), exist_ok=True)
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
        if mp4:
            try:
                # without imageio-ffmpeg, imageio would pick another plugin for ".mp4" (e.g. a TIFF writer)
                import imageio.v2 as iio
                import imageio_ffmpeg

                imageio_ffmpeg.get_ffmpeg_exe()  # raises if no ffmpeg binary is available
                self._mp4 = iio.get_writer(path_stem + ".mp4", format="FFMPEG", fps=fps, codec="libx264",
                                           quality=quality, pixelformat="yuv420p")
            except Exception as exc:
                self._disable_mp4(exc)

    def _disable_mp4(self, exc: Exception) -> None:
        """Switch from MP4 to PNG frames (removing a partial MP4) after ``exc``."""
        if self._mp4 is not None:
            try:
                self._mp4.close()
            except Exception:  # the writer is broken anyway
                pass
            self._mp4 = None
        path = self.stem + ".mp4"
        if os.path.exists(path):
            os.remove(path)
        hint = " (pip install imageio-ffmpeg)" if isinstance(exc, ImportError) else ""
        self.png_dir = self.png_dir or self.stem + "_frames"
        os.makedirs(self.png_dir, exist_ok=True)
        warnings.warn(f"MP4 output disabled: {type(exc).__name__}: {exc}{hint}; writing PNG frames to "
                      f"{self.png_dir} instead", RuntimeWarning, stacklevel=3)

    def add(self, img: np.ndarray) -> None:
        """Append one RGB uint8 frame."""
        if self._mp4 is not None:
            try:
                self._mp4.append_data(img)
            except Exception as exc:  # e.g. a broken ffmpeg; this and later frames go to PNG files
                self._disable_mp4(exc)
        if self.png_dir:
            save_png(os.path.join(self.png_dir, f"frame_{self.n:05d}.png"), img)
        if self.gif and self.n % self.gif_every == 0:
            from PIL import Image

            im = Image.fromarray(img)
            if self.gif_scale != 1.0:
                im = im.resize((max(1, int(im.width * self.gif_scale)), max(1, int(im.height * self.gif_scale))),
                               Image.LANCZOS)
            self._gif_frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        self.n += 1

    def close(self) -> list[str]:
        """Finish the files; returns the paths written (the PNG directory if frames went there instead of the
        MP4)."""
        out = []
        if self._mp4 is not None:
            self._mp4.close()
            out.append(self.stem + ".mp4")
        elif self.png_dir and self.n:
            out.append(self.png_dir)
        if self.gif and self._gif_frames:
            dur = int(round(1000.0 * self.gif_every / self.fps))
            self._gif_frames[0].save(self.stem + ".gif", save_all=True, append_images=self._gif_frames[1:],
                                     duration=dur, loop=0, optimize=False)
            out.append(self.stem + ".gif")
        return out
