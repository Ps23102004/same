"""Labelled evaluation sets, generated so ground truth is exact.

Three instances of ONE class (same silhouette, different surface) are composited
into random scenes, plus scenes with no instance at all. That gives the three
labels the product actually has to separate:

    same        instance A  -- must be returned
    same_class  instance B/C -- MUST BE REJECTED. This is the whole test.
    unrelated   no object of the class

Suites (all generated from the same code path, only the knobs differ):

    easy   object at 30% of a 1600px frame, "natural" 1/f fractal surface.
           A well-lit close-ish photo of a textured object. Best case.
    small  same surface, object at 7% of the frame (~110 px). This is the
           stage-1 recall test: a whole-image embedding cannot find this, and
           it is the size a wallet actually occupies in a room photo.
    hard   object at 16%, FLAT low-frequency surface (few strong corners, like
           plain leather), plus blur, JPEG q55, and 25% occlusion. This is the
           number to believe.

Every file gets a distinct mtime spread over four years so the indexer's date
inference produces a real timeline to sort and to compute `last_seen` from.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CANVAS = 1600   # realistic phone-photo scale; a "small" object is still hundreds of px
INSTANCES = ("A", "B", "C")  # A is the target; B and C are the same-class traps


@dataclass
class Suite:
    obj_frac: float          # object width as a fraction of the frame
    texture: str             # "noise" | "painted"
    degrade: bool            # blur + JPEG artefacts
    occlude: float           # fraction of the object hidden behind clutter
    n_per_instance: int = 8
    n_unrelated: int = 12
    quality: int = 92
    twins: bool = False      # also render instance "T": a pixel-identical TWIN
                             # of A. Two of the same mug. See eval.py.


SUITES = {
    "easy": Suite(obj_frac=0.30, texture="natural", degrade=False, occlude=0.0, twins=True),
    "small": Suite(obj_frac=0.07, texture="natural", degrade=False, occlude=0.0),
    "hard": Suite(obj_frac=0.16, texture="flat", degrade=True, occlude=0.25, quality=55),
}


def _silhouette(size: int) -> Image.Image:
    """The CLASS shape. Identical for every instance, on purpose."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    pad = size // 8
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=size // 4, fill=255)
    d.ellipse([size * 0.28, -size * 0.10, size * 0.72, size * 0.30], fill=255)
    return m


def _surface(seed: int, size: int, texture: str) -> np.ndarray:
    """The INSTANCE identity: the surface pattern, unique per seed.

    "natural" is 1/f fractal noise -- octaves of random grids from 4x4 up to
    128x128, amplitude halving each octave. That is the statistics of real
    photographed surfaces, and it is what gives SIFT keypoints at several
    scales. (An earlier version used flat pixel noise; SIFT's DoG scale-space
    smooths that away almost entirely, so it measured nothing useful. Worth
    saying out loud: a synthetic set can be accidentally degenerate in either
    direction.)

    "flat" is low-frequency only -- a plain painted/leather surface with a few
    strokes. Few strong corners, so it is the genuinely hard case.
    """
    rng = np.random.default_rng(seed)
    if texture == "natural":
        # 1/f base for the low frequencies, then a power-law scatter of
        # high-contrast shapes for the mid/high ones. Local contrast at many
        # scales is what SIFT's DoG detector actually keys on; a smooth
        # fractal alone yields almost no keypoints (measured: 7 on a 256px
        # patch), which would have made this suite meaningless.
        acc = np.zeros((size, size, 3), np.float32)
        amp = 1.0
        for octave in (4, 8, 16):
            g = rng.random((octave, octave, 3)).astype(np.float32)
            up = np.asarray(
                Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC),
                dtype=np.float32,
            ) / 255.0
            acc += amp * up
            amp *= 0.6
        acc -= acc.min()
        img = Image.fromarray((255 * acc / max(acc.max(), 1e-6)).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for _ in range(220):
            w = int(size * float(rng.pareto(1.6) + 1) / 60)
            w = max(3, min(w, size // 3))
            x, y = rng.integers(-w, size, 2)
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            shape = int(rng.integers(0, 3))
            box = [int(x), int(y), int(x + w), int(y + w * float(rng.uniform(0.4, 2.0)))]
            if shape == 0:
                d.ellipse(box, fill=c)
            elif shape == 1:
                d.rectangle(box, fill=c)
            else:
                d.line(box, fill=c, width=max(1, w // 5))
        return np.asarray(img)

    low = rng.integers(40, 215, (5, 5, 3), dtype=np.uint8)
    img = Image.fromarray(low).resize((size, size), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    for _ in range(7):
        x, y = rng.integers(0, size, 2)
        w, h = rng.integers(size // 8, size // 3, 2)
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)(
            [int(x), int(y), int(x + w), int(y + h)], fill=c
        )
    a = np.asarray(img).astype(np.float32)
    return np.clip(a + rng.normal(0, 4, a.shape), 0, 255).astype(np.uint8)


def _sprite(seed: int, size: int, texture: str) -> Image.Image:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., :3] = _surface(seed, size, texture)
    arr[..., 3] = np.asarray(_silhouette(size))
    return Image.fromarray(arr, "RGBA")


def _scene(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (CANVAS, CANVAS))
    c1 = np.array([rng.randint(25, 225) for _ in range(3)], np.float32)
    c2 = np.array([rng.randint(25, 225) for _ in range(3)], np.float32)
    t = np.linspace(0, 1, CANVAS, dtype=np.float32)[:, None, None]
    grad = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    img = Image.fromarray(np.repeat(grad, CANVAS, axis=1))
    d = ImageDraw.Draw(img)
    for _ in range(10):
        x, y = rng.randint(0, CANVAS), rng.randint(0, CANVAS)
        w, h = rng.randint(30, 140), rng.randint(30, 140)
        d.ellipse([x, y, x + w, y + h], fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return img


def build(out_dir: str | Path, suite: str = "hard", seed: int = 7,
          cfg: Suite | None = None) -> dict:
    """Render the suite and return its manifest (also written as manifest.json)."""
    cfg = cfg or SUITES[suite]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()

    manifest: dict[str, dict] = {}
    base = datetime(2021, 3, 1)
    clock = 0
    sprites = {k: _sprite(seed + i, 640, cfg.texture) for i, k in enumerate(INSTANCES)}
    order = list(INSTANCES)
    if cfg.twins:
        # "T" is a second physical copy of the same mass-produced item: same
        # mould, same print run, so the same surface down to the pixel. It is a
        # DIFFERENT object in the world and there is no signal in the pixels
        # that says so. eval.py measures what the pipeline does with it.
        sprites["T"] = sprites["A"]
        order.append("T")

    for inst in order:
        for i in range(cfg.n_per_instance):
            rng = random.Random(hash((suite, inst, i)) & 0xFFFF)
            scene = _scene(rng)
            size = int(CANVAS * cfg.obj_frac * rng.uniform(0.8, 1.25))
            obj = sprites[inst].resize((size, size), Image.BICUBIC).rotate(
                rng.uniform(-40, 40), expand=True, resample=Image.BICUBIC
            )
            x = rng.randint(0, CANVAS - obj.width)
            y = rng.randint(0, CANVAS - obj.height)
            scene.paste(obj, (x, y), obj)
            if cfg.occlude > 0:
                d = ImageDraw.Draw(scene)
                oh = int(obj.height * cfg.occlude)
                d.rectangle([x, y + obj.height - oh, x + obj.width, y + obj.height],
                            fill=tuple(rng.randint(0, 255) for _ in range(3)))
            if cfg.degrade:
                scene = scene.filter(ImageFilter.GaussianBlur(1.6))
            name = f"{inst}_{i:02d}.jpg"
            scene.save(out / name, quality=cfg.quality)
            manifest[name] = {
                "instance": inst,
                "box": [x / CANVAS, y / CANVAS,
                        (x + obj.width) / CANVAS, (y + obj.height) / CANVAS],
            }
            clock += 1
            _stamp(out / name, base + timedelta(days=37 * clock))

    for i in range(cfg.n_unrelated):
        rng = random.Random(hash((suite, "u", i)) & 0xFFFF)
        name = f"unrelated_{i:02d}.jpg"
        _scene(rng).save(out / name, quality=cfg.quality)
        manifest[name] = {"instance": None, "box": None}
        clock += 1
        _stamp(out / name, base + timedelta(days=37 * clock))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _stamp(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


if __name__ == "__main__":
    import sys

    s = sys.argv[1] if len(sys.argv) > 1 else "hard"
    d = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/same_eval_{s}"
    m = build(d, s)
    print(f"{len(m)} images -> {d}")
