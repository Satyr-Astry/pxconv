#!/usr/bin/env python
"""pxconv 基准测试: 速度 / 细结构保留率 / alpha 与调色板不变量 / 底部裁剪对照

三个可复现实验（全部只依赖仓库内自带资源，任何人克隆后都能跑出同样的量级）:

  1. 速度        同一张图, v2(整数分块) vs v3(积分图) 的「首次」与「改参数(缓存)」耗时
  2. 细结构保留率 源图里"细"的像素(局部 alpha 密度低 = 手指/细腿/发梢)
                 有多少比例在输出里活下来; 对比救援 on/off
  3. 底部裁剪对照 造一张「主体只在图片底部」的 896x1152 图,
                 v2 的整数分块会静默丢掉底部, v3 的分数边界不会

用法:
  python tools/benchmark.py                                  # 自带的 256x320 示例图
  python tools/benchmark.py path/to/art.png --size 128 --colors 24
  python tools/benchmark.py path/to/art.png --crop-control    # 额外跑底部裁剪对照
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pxconv as P                                    # noqa: E402

OUT = os.path.join(ROOT, "out", "benchmark")

CASES = [
    ("v2_mean(面积平均)",        dict(sample="mean",     edge_color="keep")),
    ("v2_majority(多数投票)",     dict(sample="majority", edge_color="stretch")),
    ("v2_center(中心点)",         dict(sample="center",   edge_color="stretch")),
    ("v3_core(默认)",             dict(sample="core",     edge_color="keep", rescue="on")),
    ("v3_core+stretch(去暗边)",   dict(sample="core",     edge_color="stretch", rescue="on")),
    ("v3_core_关救援",            dict(sample="core",     edge_color="keep", rescue="off")),
]


# ---------- 度量工具 ----------

def _sat(mask: np.ndarray) -> np.ndarray:
    T = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.float32)
    T[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.float32), 0), 1)
    return T


def _box_sum(T: np.ndarray, k: int) -> np.ndarray:
    """每个像素周围 k x k 窗口的和（越界算 0）"""
    r = k // 2
    h, w = T.shape[0] - 1, T.shape[1] - 1
    ys0 = np.clip(np.arange(h) - r, 0, h); ys1 = np.clip(np.arange(h) + r + 1, 0, h)
    xs0 = np.clip(np.arange(w) - r, 0, w); xs1 = np.clip(np.arange(w) + r + 1, 0, w)
    return (T[np.ix_(ys1, xs1)] - T[np.ix_(ys0, xs1)]
            - T[np.ix_(ys1, xs0)] + T[np.ix_(ys0, xs0)])


def thin_recall(src: np.ndarray, small_full: np.ndarray, cols: int, rows: int,
                win: int = 9, dens_thr: float = 0.45) -> float:
    """细结构保留率

    "细" = alpha>0.5 且 9x9 窗口内不透明像素密度 < dens_thr —— 手指、细腿、
    尾巴尖、发梢这类 1~3px 宽的结构。输出的可见格覆盖了多少"细"源像素。
    """
    a = src[..., 3]
    dens = _box_sum(_sat(a > 0.5), win) / float(win * win)
    thin = (a > 0.5) & (dens < dens_thr)
    total = int(thin.sum())
    if total == 0:
        return 1.0
    h, w = a.shape
    T = _sat(thin.astype(np.float32))
    ys = np.linspace(0.0, h, rows + 1)
    xs = np.linspace(0.0, w, cols + 1)
    B = P._integ_bounds(T[..., None], ys, xs)
    S = B[1:, 1:] - B[:-1, 1:] - B[1:, :-1] + B[:-1, :-1]
    vis = small_full[..., 3] > 0.5
    r, c = min(rows, vis.shape[0]), min(cols, vis.shape[1])
    kept = float(S[..., 0][:r, :c][vis[:r, :c]].sum())
    return kept / float(total)


def components(mask: np.ndarray) -> int:
    """4-连通块数（小图 BFS）"""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    stack: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                n += 1
                stack.append((y, x)); seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
    return n


def namespace(kw: dict, size: int, colors: int) -> argparse.Namespace:
    return argparse.Namespace(
        size=size, pixel_size=None, colors=colors, alpha_thr=0.5, cov_thr=0.5,
        core_thr=0.7, rescue=kw.get("rescue", "on"), palette_method="median",
        alpha_mode=kw.get("alpha_mode", "hard"), edge=kw.get("edge", "none"),
        edge_color=kw.get("edge_color", "keep"), sample=kw["sample"],
        despike=1, pad=0, no_trim=False, preview_scale=1, no_preview=True)


# ---------- 实验 1+2: 速度与细结构 ----------

def run_case(src: np.ndarray, prep: dict, tag: str, kw: dict, size: int,
             colors: int) -> dict:
    cols, rows = P.target_grid(src.shape[0], src.shape[1], size, None)
    o = namespace(kw, size, colors)

    t0 = time.perf_counter()
    small = P.convert(src, cols, rows, o, prep=prep)
    t_first = time.perf_counter() - t0

    t1 = time.perf_counter()
    small2 = P.convert(src, cols, rows, o, prep=prep)
    t_cached = time.perf_counter() - t1
    assert np.array_equal(small, small2), "convert 必须确定性可复现"

    o2 = namespace(kw, size, colors); o2.no_trim = True
    full = P.convert(src, cols, rows, o2, prep=prep)
    recall = thin_recall(src, full, cols, rows)

    a8 = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
    vis = a8[..., 3] > 0
    fs = P.fringe_stats(small)
    os.makedirs(OUT, exist_ok=True)
    stem = f"{os.path.splitext(os.path.basename(TAG_IMAGE))[0]}_{size}px_{tag}"
    P.save_rgba(small, os.path.join(OUT, stem + ".png"))
    P.checker_preview(small, 4).save(os.path.join(OUT, stem + "_preview.png"))

    return dict(tag=tag, size=f"{a8.shape[1]}x{a8.shape[0]}",
                vispx=int(vis.sum()), comps=components(vis),
                thin=round(recall * 100, 1),
                colors=int(len(np.unique(a8[vis][:, :3], axis=0))) if vis.any() else 0,
                alpha_levels=int(len(np.unique(a8[..., 3]))),
                semi=int(((a8[..., 3] > 0) & (a8[..., 3] < 255)).sum()),
                edge_delta=round(fs["edge_delta"], 1),
                ms_first=round(t_first * 1000, 1), ms_cached=round(t_cached * 1000, 1),
                file=os.path.join("out", "benchmark", stem + ".png"))


# ---------- 实验 3: 底部裁剪对照 ----------

def make_bottom_content(w: int = 896, h: int = 1152) -> np.ndarray:
    """造一张「主体只在图片底部 5%」的图: 上半部分是空气, 底部一条红块。

    v2 的整数分块 (arr[:bh*rows]) 覆盖不到最后几行 -> 红块整块消失。
    """
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([w * 0.3, h * 0.02, w * 0.7, h * 0.10], fill=(70, 110, 200, 255))   # 顶部参照物
    d.rectangle([w * 0.2, h * 0.55, w * 0.8, h * 0.98], fill=(240, 220, 200, 255))  # 身体
    d.rectangle([w * 0.25, h * 0.93, w * 0.75, h - 2], fill=(220, 40, 40, 255))     # 底部红块
    return np.asarray(img).astype(np.float32) / 255.0


def crop_control(size: int = 128) -> list[dict]:
    src = make_bottom_content()
    h, w = src.shape[:2]
    cols, rows = P.target_grid(h, w, size, None)
    a8_src = (np.clip(src, 0, 1) * 255 + 0.5).astype(np.uint8)
    red_src = int(((a8_src[..., 0] > 180) & (a8_src[..., 1] < 90) & (a8_src[..., 3] > 127)).sum())

    rows_out = []
    for tag, kw in (("v2_mean", dict(sample="mean", edge_color="keep")),
                    ("v2_majority", dict(sample="majority", edge_color="keep")),
                    ("v3_core", dict(sample="core", edge_color="keep"))):
        prep = P.prepare(src, 0.7)
        o = namespace(kw, size, colors=32); o.no_trim = True
        small = P.convert(src, cols, rows, o, prep=prep)
        a8 = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
        vis = a8[..., 3] > 127
        red = int(((a8[..., 0] > 180) & (a8[..., 1] < 90) & vis).sum())
        ys = np.where(vis.any(axis=1))[0]
        rows_out.append(dict(case=tag, grid=f"{cols}x{rows}",
                             vis_cells=int(vis.sum()),
                             row_range=f"{int(ys.min())}-{int(ys.max())}" if len(ys) else "-",
                             bottom_red=red,
                             src_red_px=red_src))
    return rows_out


def cjk_pad(s: str, width: int) -> str:
    w = sum(2 if ord(c) > 127 else 1 for c in s)
    return s + " " * max(0, width - w)


TAG_IMAGE = ""


def main() -> None:
    global TAG_IMAGE
    ap = argparse.ArgumentParser(description="pxconv 基准测试")
    ap.add_argument("image", nargs="?", default=None,
                    help="输入透明图（默认 assets/sample_raw.png）")
    ap.add_argument("--size", type=int, default=64, help="输出宽度")
    ap.add_argument("--colors", type=int, default=16)
    ap.add_argument("--crop-control", action="store_true",
                    help="额外跑 896x1152 底部内容对照实验（约 1 秒）")
    a = ap.parse_args()

    img = a.image or os.path.join(ROOT, "assets", "sample_raw.png")
    TAG_IMAGE = img
    t0 = time.perf_counter()
    src = P.load_rgba(img)
    prep = P.prepare(src, 0.7)
    t_prep = (time.perf_counter() - t0) * 1000
    h, w = src.shape[:2]
    print(f"输入 {img}  {w}x{h}  半透明像素={int(((src[...,3]>0)&(src[...,3]<1)).sum())}")
    print(f"预处理(读图+积分图) {t_prep:.0f} ms    输出宽度 {a.size}px / {a.colors} 色\n")

    hdr = (cjk_pad("配置", 24) + f"{'尺寸':>9}{'细结构%':>8}{'可见px':>7}{'连通块':>7}"
           f"{'色数':>6}{'alpha级':>8}{'半透明':>7}{'首次ms':>8}{'缓存ms':>8}")
    print(hdr); print("-" * 96)
    rows = []
    for tag, kw in CASES:
        r = run_case(src, prep, tag, kw, a.size, a.colors)
        rows.append(r)
        print(cjk_pad(r["tag"], 24) + f"{r['size']:>9}{r['thin']:>8}{r['vispx']:>7}"
              f"{r['comps']:>7}{r['colors']:>6}{r['alpha_levels']:>8}{r['semi']:>7}"
              f"{r['ms_first']:>8}{r['ms_cached']:>8}")

    result: dict = dict(image=img, size=a.size, colors=a.colors,
                        src=f"{w}x{h}", prep_ms=round(t_prep, 1), cases=rows)

    if a.crop_control:
        print("\n=== 底部内容对照（896x1152 合成图，主体只在底部 5%）===")
        print(f"{'case':<14}{'输出网格':>10}{'可见格':>8}{'行范围':>10}{'底部红像素':>11}{'源红像素':>10}")
        cc = crop_control(a.size)
        for r in cc:
            print(f"{r['case']:<14}{r['grid']:>10}{r['vis_cells']:>8}{r['row_range']:>10}"
                  f"{r['bottom_red']:>11}{r['src_red_px']:>10}")
        result["crop_control"] = cc

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n[OK] 指标 -> {path}   图 -> {OUT}")


if __name__ == "__main__":
    main()
