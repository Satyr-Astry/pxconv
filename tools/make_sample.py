#!/usr/bin/env python
"""生成一张带软 alpha 边缘的测试用透明图（模拟 AI 出图 + 抠图后的效果）

画的是粗糙几何形状，用 Lanczos 降采样产生抗锯齿半透明边 —— 这正是像素化时
必须清理掉的「毛边」。整张图是程序合成的，不含任何第三方素材。

用法:
  python tools/make_sample.py                       # -> assets/sample_raw.png (256x320)
  python tools/make_sample.py --out out/x.png --size 512x640
"""
from __future__ import annotations
import argparse, os
import numpy as np
from PIL import Image, ImageDraw


def build(w: int = 256, h: int = 320, supersample: int = 4) -> Image.Image:
    S = supersample
    W, H = w * S, h * S
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.ellipse([60 * S, 30 * S, 196 * S, 210 * S], fill=(38, 32, 48, 255))          # 头发
    d.ellipse([80 * S, 60 * S, 176 * S, 170 * S], fill=(255, 222, 200, 255))       # 脸
    d.ellipse([98 * S, 100 * S, 118 * S, 124 * S], fill=(60, 90, 150, 255))        # 眼
    d.ellipse([138 * S, 100 * S, 158 * S, 124 * S], fill=(60, 90, 150, 255))
    d.ellipse([104 * S, 104 * S, 110 * S, 112 * S], fill=(255, 255, 255, 255))     # 高光
    d.ellipse([144 * S, 104 * S, 150 * S, 112 * S], fill=(255, 255, 255, 255))
    d.arc([112 * S, 128 * S, 144 * S, 152 * S], 0, 180, fill=(190, 90, 100, 255), width=3 * S)
    d.polygon([(128 * S, 165 * S), (74 * S, 300 * S), (182 * S, 300 * S)],         # 裙子
              fill=(90, 130, 220, 255))
    d.line([(96 * S, 210 * S), (70 * S, 250 * S)], fill=(255, 222, 200, 255), width=10 * S)
    d.line([(160 * S, 210 * S), (186 * S, 250 * S)], fill=(255, 222, 200, 255), width=10 * S)
    d.rectangle([100 * S, 300 * S, 126 * S, 316 * S], fill=(245, 245, 250, 255))   # 袜/鞋
    d.rectangle([132 * S, 300 * S, 158 * S, 316 * S], fill=(245, 245, 250, 255))

    return img.resize((w, h), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成带软 alpha 边缘的测试透明图")
    ap.add_argument("--out", default=None, help="输出路径（默认 assets/sample_raw.png）")
    ap.add_argument("--size", default="256x320", help="输出尺寸，如 256x320")
    ap.add_argument("--supersample", type=int, default=4)
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = a.out or os.path.join(root, "assets", "sample_raw.png")
    w, h = (int(v) for v in a.size.lower().split("x"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    img = build(w, h, a.supersample)
    img.save(out)
    a8 = np.asarray(img.convert("RGBA"))
    semi = int(((a8[..., 3] > 0) & (a8[..., 3] < 255)).sum())
    print(f"[OK] {out}  {img.width}x{img.height}  半透明像素={semi}")


if __name__ == "__main__":
    main()
