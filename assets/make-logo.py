#!/usr/bin/env python3
"""Render the Anivar app mark.

The mark is a single zigzag crossed by one broad bar, and the letters of ANIVAR
are found in it at different angles — so the GEOMETRY below is fixed and must not
be "tidied". Changing a vertex changes which letters can still be read out of it.

What this file controls is the finish, and one choice in it is load-bearing:
the bar is laid down FIRST and the letters over the top. When the bar sits on top
it severs every diagonal at the waist into two floating pieces, and a letter can
no longer be traced through the middle of the icon — which is the one thing the
mark exists to let you do.

The strokes are painted rather than drawn: width breathes along the path, the
spine drifts, the ends taper on lift-off, and the bristles skip at the boundary.
Brush character belongs in the EDGE — laying a highlight down the centre of a
stroke reads as a shiny tube, not paint.

    python assets/make-logo.py            # writes logo-1024.png + logo.png
    npm run tauri icon assets/logo-1024.png   # then regenerates all 51 sizes

TEXTURE tunes how dry the brush is: 0.5 light hand, 1.0 brush, 1.6 dry brush.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

# ── Fixed identity ───────────────────────────────────────────────────────────
ZIGZAG = [(12, 86), (37, 15), (62, 86), (87, 15)]   # do not edit: holds the letters
BAR = [(-12, 56.6), (112, 55.4)]                    # bled past both edges, slight tilt
BONE = (237, 230, 220)
RED = (201, 96, 92)                                  # the app's single accent
TILE_TOP, TILE_BOTTOM = (46, 42, 38), (18, 16, 15)   # warm graphite, lit from above
RIM = (76, 70, 64)

TEXTURE = 1.0
SUPERSAMPLE = 2


def _samples(points, step=0.35):
    """Walk the path at even spacing, carrying position, 0..1 progress and tangent."""
    segs = [(a, b, math.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(points, points[1:])]
    total = sum(s[2] for s in segs)
    out, acc = [], 0.0
    for a, b, length in segs:
        n = max(2, int(length / step))
        for i in range(n + 1):
            f = i / n
            out.append((
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                (acc + length * f) / total,
                ((b[0] - a[0]) / length, (b[1] - a[1]) / length),
            ))
        acc += length
    return out


def _wobble(t, seed, freqs=(23, 57, 113)):
    """Smooth, deterministic 1-D noise: the same mark renders identically every run."""
    amps = (0.55, 0.30, 0.15)
    return sum(a * math.sin(t * f + seed * (i + 1.7))
               for i, (f, a) in enumerate(zip(freqs, amps)))


def brush(n, u, points, width, seed, texture, taper=0.42):
    """A stroke laid down like paint. Returns an L-mode coverage mask."""
    mask = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(mask)
    for x, y, t, (tx, ty) in _samples(points):
        lift = min(1.0, t / 0.06) * min(1.0, (1 - t) / 0.06)
        r = width * 0.5 * (1 - taper + taper * lift ** 0.5) * (1 + 0.17 * texture * _wobble(t, seed)) * u
        drift = 0.11 * texture * width * _wobble(t, seed + 4.3, (17, 41, 83)) * u
        cx, cy = x * u - ty * drift, y * u + tx * drift
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    if texture:
        # Bite irregular chunks out of the EDGE only — that is where a dry brush
        # skips first. Erosion by blur+threshold, not MinFilter: a 60px min-kernel
        # on a 3k canvas takes minutes, this is the same band in milliseconds.
        interior = mask.filter(ImageFilter.GaussianBlur(1.5 * u)).point(lambda v: 255 if v > 238 else 0)
        edge = ImageChops.subtract(mask, interior)
        grit = Image.effect_noise((n, n), 72).filter(ImageFilter.GaussianBlur(0.75 * u))
        grit = grit.point(lambda v: 0 if v > 128 - 36 * texture else 255)
        mask = ImageChops.subtract(mask, ImageChops.multiply(edge, grit))
        mask = mask.filter(ImageFilter.GaussianBlur(0.28 * u))
    return mask


def render(size, texture=TEXTURE):
    n = size * SUPERSAMPLE
    u = n / 100.0
    img = Image.new("RGB", (n, n))
    d = ImageDraw.Draw(img)
    for y in range(n):
        f = y / (n - 1)
        d.line([(0, y), (n, y)],
               fill=tuple(int(TILE_TOP[i] + (TILE_BOTTOM[i] - TILE_TOP[i]) * f) for i in range(3)))

    # Bar first, letters over it — see the module docstring; this is the whole point.
    img.paste(Image.new("RGB", (n, n), RED), (0, 0), brush(n, u, BAR, 13, 4.1, texture, taper=0.14))
    letters = brush(n, u, ZIGZAG, 9.6, 1.7, texture)
    img.paste(Image.new("RGB", (n, n), (0, 0, 0)), (0, 0),
              letters.filter(ImageFilter.GaussianBlur(2.0 * u)).point(lambda v: int(v * 0.48)))
    img.paste(Image.new("RGB", (n, n), BONE), (0, 0), letters)

    ImageDraw.Draw(img).rounded_rectangle(
        [1.2 * u, 1.2 * u, n - 1.2 * u, n - 1.2 * u],
        radius=int(21 * u), outline=RIM, width=max(1, int(1.1 * u)))

    tile = Image.new("L", (n, n), 0)
    ImageDraw.Draw(tile).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(22 * u), fill=255)
    out = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    out.paste(img, (0, 0), tile)
    return out.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    here = Path(__file__).parent
    master = render(1024)
    master.save(here / "logo-1024.png")
    # Every smaller size is DOWNSAMPLED from the master, never re-rendered: fresh
    # noise per size puts texture in the 32px icon that the real pipeline
    # (`tauri icon`, which scales one master) would never produce.
    master.resize((512, 512), Image.LANCZOS).save(here / "logo.png")
    print(f"wrote {here/'logo-1024.png'} and {here/'logo.png'}")
