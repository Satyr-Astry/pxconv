#!/usr/bin/env python
"""边缘参数矩阵实测: 同一张透明图, 每组参数跑一遍, 出指标表 + 预览图

指标:
  尺寸      输出像素尺寸(裁边后)
  alpha     输出 PNG 里 alpha 的取值 (像素画应为 0,255)
  半透明    半透明像素个数 (像素画应为 0)
  色数      可见像素的颜色种数 (<= --colors)
  边缘px    有透明邻居的可见像素数 (= 1px 轮廓环的长度)
  边缘色差  边缘像素色 vs 内圈像素色 的平均距离(0~441), 越小 = 暗边/渗色越少
  发灰率    边缘里"发灰"的比例 (会被本体深色/白色干扰, 仅参考)

用法:
  python tools/edge_bench.py                        # assets/sample_raw.png, 64 和 128 两档
  python tools/edge_bench.py art.png --sizes 32,64 --colors 16
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pxconv as P                                    # noqa: E402

OUT = os.path.join(ROOT, "out", "edge_tests")

CONFIGS = [
    ("R0_NEAREST+二值化+限色",    None),
    ("E1_v2_mean 面积平均",       dict(sample="mean",     alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="keep")),
    ("E2_v2_majority",            dict(sample="majority", alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="keep")),
    ("E3_majority+stretch",       dict(sample="majority", alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="stretch")),
    ("E4_majority+avg",           dict(sample="majority", alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="avg")),
    ("E5_shrink 收缩1px",         dict(sample="majority", alpha_mode="hard", alpha_thr=0.5,  edge="shrink", edge_color="keep")),
    ("E6_阈值0.65 瘦一圈",        dict(sample="majority", alpha_mode="hard", alpha_thr=0.65, edge="none",   edge_color="keep")),
    ("E7_v2_center 中心点",       dict(sample="center",   alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="keep")),
    ("E8_keep 保留柔边",          dict(sample="mean",     alpha_mode="keep", alpha_thr=0.5,  edge="none",   edge_color="keep")),
    ("E9_v3_core 默认",           dict(sample="core",     alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="keep",   rescue="on")),
    ("EA_v3_core+stretch",        dict(sample="core",     alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="stretch", rescue="on")),
    ("EB_v3_core 关救援",         dict(sample="core",     alpha_mode="hard", alpha_thr=0.5,  edge="none",   edge_color="keep",   rescue="off")),
]


def stats(small: np.ndarray, name: str, out: str) -> dict:
    a8 = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
    vis = a8[a8[..., 3] > 0]
    fs = P.fringe_stats(small)
    lv = np.unique(a8[..., 3])
    lv_s = ",".join(str(int(v)) for v in lv[:3]) + ("…" if len(lv) > 3 else "")
    return dict(name=name, size=f"{a8.shape[1]}x{a8.shape[0]}", alpha_lv=lv_s,
                alpha_n=int(len(lv)),
                semi=int(((a8[..., 3] > 0) & (a8[..., 3] < 255)).sum()),
                colors=int(len(np.unique(vis[:, :3], axis=0))) if len(vis) else 0,
                boundary=fs["boundary"], edge_delta=round(fs["edge_delta"], 1),
                grayish=round(fs["grayish"], 3), file=out)


def naive_nearest(src: np.ndarray, cols: int, rows: int, colors: int) -> np.ndarray:
    """对照组: 最常见的做法 —— NEAREST 缩放 + alpha 二值化 + 调色板吸附"""
    img = Image.fromarray((np.clip(src, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    small = np.asarray(img.resize((cols, rows), Image.NEAREST)).astype(np.float32) / 255.0
    small = P.harden_alpha(small, 0.5)
    return P.snap_to_palette(small, P.build_palette(src, colors))


def run(src: np.ndarray, name: str, kw: dict | None, size: int, colors: int,
        sub: str) -> dict:
    cols, rows = P.target_grid(src.shape[0], src.shape[1], size, None)
    if kw is None:
        small = P.trim_pad(naive_nearest(src, cols, rows, colors), True, 0)
    else:
        o = argparse.Namespace(size=size, pixel_size=None, colors=colors,
                               alpha_thr=kw["alpha_thr"], alpha_mode=kw["alpha_mode"],
                               edge=kw["edge"], edge_color=kw["edge_color"],
                               sample=kw["sample"], cov_thr=kw["alpha_thr"],
                               core_thr=kw.get("core_thr", 0.7),
                               rescue=kw.get("rescue", "on"), palette_method="median",
                               despike=1, pad=0, no_trim=False, preview_scale=4,
                               no_preview=False)
        small = P.convert(src, cols, rows, o)
    d = os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, name + ".png")
    P.save_rgba(small, out)
    P.checker_preview(small, 4).save(os.path.join(d, name + "_preview.png"))
    return stats(small, name, os.path.join("out", "edge_tests", sub, name + ".png"))


def cjk_pad(s: str, width: int) -> str:
    w = sum(2 if ord(c) > 127 else 1 for c in s)
    return s + " " * max(0, width - w)


def main() -> None:
    ap = argparse.ArgumentParser(description="边缘参数矩阵实测")
    ap.add_argument("image", nargs="?", default=None, help="输入透明图")
    ap.add_argument("--sizes", default="64,128")
    ap.add_argument("--colors", type=int, default=16)
    a = ap.parse_args()

    img = a.image or os.path.join(ROOT, "assets", "sample_raw.png")
    src = P.load_rgba(img)
    print(f"输入 {img}  {src.shape[1]}x{src.shape[0]}")
    all_rows: dict = {}
    for size in [int(x) for x in a.sizes.split(",")]:
        rows = [run(src, n, kw, size, a.colors, f"s{size}") for n, kw in CONFIGS]
        all_rows[size] = rows
        hdr = (cjk_pad("配置", 26) + f"{'尺寸':>9}{'alpha':>10}{'半透明':>7}{'色数':>5}"
               f"{'边缘px':>7}{'边缘色差':>9}{'发灰率':>8}")
        print(f"\n=== 输出宽度 {size}px / {a.colors} 色 ===")
        print(hdr); print("-" * 92)
        for r in rows:
            print(cjk_pad(r["name"], 26) + f"{r['size']:>9}{r['alpha_lv']:>10}{r['semi']:>7}"
                  f"{r['colors']:>5}{r['boundary']:>7}{r['edge_delta']:>9}{r['grayish']:>8}")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=1)
    print(f"\n[OK] 指标 -> {path}")


if __name__ == "__main__":
    main()
