#!/usr/bin/env python
"""尺寸对比图: 同一张透明图, 多个输出尺寸并排, 方便挑最合适的一档

用法:
  python tools/contact_sheet.py                              # 自带示例图, 48~160
  python tools/contact_sheet.py art.png --sizes 48,64,96,128,160 --colors 32
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pxconv as P                                    # noqa: E402

TARGET_H = 200          # 每个 sprite 放大后统一高度
PAD = 14
LABEL_H = 20


def checker_bg(w: int, h: int, cell: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    t = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    v = np.where(t == 0, 168, 200).astype(np.uint8)
    return np.dstack([v, v, v, np.full_like(v, 255)])


def make(path: str, out: str, sizes: list[int], colors: int,
         sample: str = "core", edge_color: str = "stretch",
         alpha_thr: float = 0.5) -> tuple[str, list[str]]:
    src = P.load_rgba(path)
    tiles = []
    for s in sizes:
        cols, rows = P.target_grid(src.shape[0], src.shape[1], s, None)
        o = argparse.Namespace(size=s, pixel_size=None, colors=colors, alpha_thr=alpha_thr,
                               cov_thr=alpha_thr, core_thr=0.7, rescue="on",
                               palette_method="median", alpha_mode="hard",
                               edge="none", edge_color=edge_color, sample=sample,
                               despike=1, pad=0, no_trim=False, preview_scale=1,
                               no_preview=True)
        small = P.convert(src, cols, rows, o)
        im = Image.fromarray((np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
        k = max(1, round(TARGET_H / im.height))      # 取最近的整数倍, 各档显示高度更接近
        up = im.resize((im.width * k, im.height * k), Image.NEAREST)
        tiles.append((f"{s}px  {im.width}x{im.height}", up))

    width = sum(t[1].width for t in tiles) + PAD * (len(tiles) + 1)
    height = TARGET_H + LABEL_H + PAD * 2
    canvas = Image.fromarray(checker_bg(width, height), "RGBA")
    d = ImageDraw.Draw(canvas)
    x = PAD
    for label, im in tiles:
        y = PAD + (TARGET_H - im.height) // 2
        canvas.alpha_composite(im, (x, max(PAD, y)))
        d.text((x, height - LABEL_H), label, fill=(20, 20, 24, 255))
        x += im.width + PAD
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    canvas.convert("RGB").save(out)
    return out, [t[0] for t in tiles]


def main() -> None:
    ap = argparse.ArgumentParser(description="多尺寸对比图")
    ap.add_argument("images", nargs="*", help="输入透明图（默认 assets/sample_raw.png）")
    ap.add_argument("--sizes", default="48,64,96,128,160")
    ap.add_argument("--colors", type=int, default=24)
    ap.add_argument("--sample", default="core", choices=["core", "mean", "majority", "center"])
    ap.add_argument("--edge-color", default="stretch", choices=["keep", "avg", "stretch"])
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "contact_sheet"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    imgs = a.images or [os.path.join(ROOT, "assets", "sample_raw.png")]
    sizes = [int(x) for x in a.sizes.split(",")]
    for p in imgs:
        nm = os.path.splitext(os.path.basename(p))[0]
        o = os.path.join(a.out, f"contact_{nm}_{a.colors}c.png")
        out, labels = make(p, o, sizes, a.colors, a.sample, a.edge_color)
        print("[OK]", out, "|", " / ".join(labels))


if __name__ == "__main__":
    main()
