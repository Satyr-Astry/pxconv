#!/usr/bin/env python
"""把若干张像素素材并排拼成一张对比图（棋盘格底 + 标签），方便肉眼看哪个算法好

用法:
  python tools/compare_sheet.py                        # 默认对比 benchmark 输出里的 v2/v3
  python tools/compare_sheet.py a.png:标签A b.png:标签B --out out/cmp.png
"""
from __future__ import annotations
import argparse, os
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_H = 300
PAD = 16
LABEL_H = 22


def checker_bg(w: int, h: int, cell: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    t = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    v = np.where(t == 0, 168, 200).astype(np.uint8)
    return np.dstack([v, v, v, np.full_like(v, 255)])


def sheet(files: list[tuple[str, str]], out: str, target_h: int = TARGET_H) -> str:
    tiles = []
    for f, label in files:
        im = Image.open(f).convert("RGBA")
        k = max(1, target_h // im.height)
        tiles.append((label, im.resize((im.width * k, im.height * k), Image.NEAREST)))
    w = sum(t[1].width for t in tiles) + PAD * (len(tiles) + 1)
    h = target_h + LABEL_H + PAD * 2
    canvas = Image.fromarray(checker_bg(w, h), "RGBA")
    d = ImageDraw.Draw(canvas)
    x = PAD
    for label, im in tiles:
        canvas.alpha_composite(im, (x, PAD + (target_h - im.height) // 2))
        d.text((x, h - LABEL_H), label, fill=(16, 16, 20, 255))
        x += im.width + PAD
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    canvas.convert("RGB").save(out)
    print("[OK]", out, os.path.getsize(out), "B")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="算法并排对比图")
    ap.add_argument("items", nargs="*", help="路径:标签 ...")
    ap.add_argument("--stem", default="sample_raw_64px",
                    help="默认查找 out/benchmark/<stem>_<配置>.png")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "compare_sheet", "algorithms.png"))
    a = ap.parse_args()
    if a.items:
        files = [(x.split(":")[0], x.split(":")[1]) for x in a.items]
    else:
        base = os.path.join(ROOT, "out", "benchmark")
        files = [(os.path.join(base, f"{a.stem}_v2_mean(面积平均).png"), "v2 mean"),
                 (os.path.join(base, f"{a.stem}_v2_majority(多数投票).png"), "v2 majority"),
                 (os.path.join(base, f"{a.stem}_v3_core(默认).png"), "v3 core (new)"),
                 (os.path.join(base, f"{a.stem}_v3_core+stretch(去暗边).png"), "v3 core+stretch"),
                 (os.path.join(base, f"{a.stem}_v3_core_关救援.png"), "v3 no rescue")]
        files = [(f, l) for f, l in files if os.path.isfile(f)]
        if not files:
            raise SystemExit(f"没找到 {base}/{a.stem}_*.png —— 先跑 python tools/benchmark.py")
    sheet(files, a.out)


if __name__ == "__main__":
    main()
