#!/usr/bin/env python
"""pxconv 回归测试 —— 每个测试都对应 README 里的一条「算法优点」声明

跑法（两种都行）:
  pytest -q tests/                # 需要 pip install pytest
  python tests/test_pxconv.py     # 零依赖跑法

设计原则: 断言的是**不变量**，不是像素级快照 —— 换 numpy/PIL 版本也不会假失败。
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pxconv as P                                    # noqa: E402

SAMPLE = os.path.join(ROOT, "assets", "sample_raw.png")


# ---------- 小工具 ----------

def ns(**kw):
    base = dict(size=64, pixel_size=None, colors=16, alpha_thr=0.5, cov_thr=0.5,
                core_thr=0.7, rescue="on", palette_method="median",
                alpha_mode="hard", edge="none", edge_color="keep", sample="core",
                despike=1, pad=0, no_trim=False, preview_scale=4, no_preview=True)
    base.update(kw)
    return argparse.Namespace(**base)


def to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def convert_path(path: str, **kw) -> np.ndarray:
    src = P.load_rgba(path)
    o = ns(**kw)
    cols, rows = P.target_grid(src.shape[0], src.shape[1], o.size, None)
    return P.convert(src, cols, rows, o)


def synthetic_bottom_content(w: int = 896, h: int = 1152) -> np.ndarray:
    """主体在图片底部 5% 的合成图（用来抓 v2 整数分块丢底部的 bug）"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([w * 0.30, h * 0.02, w * 0.70, h * 0.10], fill=(70, 110, 200, 255))
    d.rectangle([w * 0.20, h * 0.55, w * 0.80, h * 0.98], fill=(240, 220, 200, 255))
    d.rectangle([w * 0.25, h * 0.93, w * 0.75, h - 2], fill=(220, 40, 40, 255))
    return np.asarray(img).astype(np.float32) / 255.0


def synthetic_thin_structures(w: int = 600, h: int = 600) -> np.ndarray:
    """一块实心身体 + 从身体伸出的 4px 细手指/发丝（典型会丢的细结构）"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([150, 150, 450, 450], fill=(230, 200, 180, 255))          # 脸
    for x in (200, 250, 300, 350, 400):                                  # 5 根细手指
        d.line([(x, 420), (x - 6, 560)], fill=(230, 200, 180, 255), width=4)
    for x in (170, 430):                                                 # 2 根细发丝
        d.line([(x, 200), (x + (-40 if x < 300 else 40), 40)],
               fill=(40, 34, 50, 255), width=4)
    return np.asarray(img).astype(np.float32) / 255.0


def thin_recall(src: np.ndarray, small_full: np.ndarray, cols: int, rows: int,
                win: int = 9, dens_thr: float = 0.45) -> float:
    """细结构保留率（与 tools/benchmark.py 同一实现）"""
    a = src[..., 3]
    T = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float32)
    T[1:, 1:] = np.cumsum(np.cumsum((a > 0.5).astype(np.float32), 0), 1)
    r = win // 2
    h, w = a.shape
    ys0 = np.clip(np.arange(h) - r, 0, h); ys1 = np.clip(np.arange(h) + r + 1, 0, h)
    xs0 = np.clip(np.arange(w) - r, 0, w); xs1 = np.clip(np.arange(w) + r + 1, 0, w)
    dens = (T[np.ix_(ys1, xs1)] - T[np.ix_(ys0, xs1)] - T[np.ix_(ys1, xs0)]
            + T[np.ix_(ys0, xs0)]) / float(win * win)
    thin = (a > 0.5) & (dens < dens_thr)
    total = int(thin.sum())
    if total == 0:
        return 1.0
    Tt = np.zeros((h + 1, w + 1), np.float32)
    Tt[1:, 1:] = np.cumsum(np.cumsum(thin.astype(np.float32), 0), 1)
    ys = np.linspace(0.0, h, rows + 1); xs = np.linspace(0.0, w, cols + 1)
    B = P._integ_bounds(Tt[..., None], ys, xs)
    S = B[1:, 1:] - B[:-1, 1:] - B[1:, :-1] + B[:-1, :-1]
    vis = small_full[..., 3] > 0.5
    rr, cc = min(rows, vis.shape[0]), min(cols, vis.shape[1])
    return float(S[..., 0][:rr, :cc][vis[:rr, :cc]].sum()) / float(total)


# ---------- 1. alpha 必须是硬边 ----------

def test_alpha_is_binary():
    """像素画要求 alpha 只有 0 / 255；普通缩放会留下几十种半透明值"""
    out = convert_path(SAMPLE, size=64, alpha_mode="hard")
    levels = np.unique(to_u8(out)[..., 3])
    assert set(int(v) for v in levels) <= {0, 255}, f"出现半透明 alpha: {levels[:10]}"


def test_naive_nearest_leaves_semi_transparent_fringe():
    """反证：对照组（NEAREST + 二值化前）确实会产生大量半透明像素"""
    src = P.load_rgba(SAMPLE)
    cols, rows = P.target_grid(src.shape[0], src.shape[1], 64, None)
    naive = np.asarray(Image.fromarray(to_u8(src), "RGBA").resize((cols, rows), Image.NEAREST))
    semi = int(((naive[..., 3] > 0) & (naive[..., 3] < 255)).sum())
    levels = len(np.unique(naive[..., 3]))
    assert semi > 0 and levels > 2, "对照组应当留下半透明毛边（本例假设不成立，需换图）"


# ---------- 2. 调色板上限必须真的生效 ----------

def test_palette_cap_is_enforced():
    for colors in (8, 16, 24):
        out = to_u8(convert_path(SAMPLE, size=64, colors=colors))
        vis = out[out[..., 3] > 0]
        n = len(np.unique(vis[:, :3], axis=0)) if len(vis) else 0
        assert n <= colors, f"--colors {colors} 失效：实际 {n} 色"


# ---------- 3. 分数边界覆盖整张图（v2 的整数分块会裁掉底部） ----------

def test_fractional_grid_covers_whole_image():
    src = synthetic_bottom_content()
    h, w = src.shape[:2]
    cols, rows = P.target_grid(h, w, 128, None)
    prep = P.prepare(src, 0.7)
    cov, _cnt, _rgb = P.cells_from_sat(prep, rows, cols)
    # 覆盖率是按面积积出来的：所有格子覆盖的 alpha 总量 ≈ 源图 alpha 总量
    total_alpha = float(src[..., 3].sum())
    covered = float((cov * (h / rows) * (w / cols)).sum())
    assert abs(covered - total_alpha) / total_alpha < 1e-3, \
        f"覆盖率没有覆盖全图: {covered:.1f} vs {total_alpha:.1f}"


def test_v3_keeps_bottom_content_that_v2_crops():
    """v2 用 arr[:bh*rows] 整数分块，896x1152 -> 128px 时只盖 990 行，底部整块消失"""
    src = synthetic_bottom_content()
    h, w = src.shape[:2]
    cols, rows = P.target_grid(h, w, 128, None)

    def red_cells(kw: dict) -> int:
        o = ns(size=128, colors=32, no_trim=True, **kw)
        small = P.convert(src, cols, rows, o, prep=P.prepare(src, 0.7))
        a8 = to_u8(small)
        vis = a8[..., 3] > 127
        return int(((a8[..., 0] > 180) & (a8[..., 1] < 90) & vis).sum())

    v2 = red_cells(dict(sample="mean"))
    v3 = red_cells(dict(sample="core"))
    assert v2 == 0, f"对照组 v2 本应丢掉底部（实际保留 {v2} 格）—— 若 v2 已修好，本条应更新"
    assert v3 >= 20, f"v3 应保住底部红块，实际只保留 {v3} 格"


# ---------- 4. 细结构救援 ----------

def test_rescue_recovers_thin_structures():
    src = synthetic_thin_structures()
    cols, rows = P.target_grid(src.shape[0], src.shape[1], 48, None)

    def recall(rescue: str) -> float:
        o = ns(size=48, colors=32, no_trim=True, rescue=rescue, sample="core")
        full = P.convert(src, cols, rows, o, prep=P.prepare(src, 0.7))
        return thin_recall(src, full, cols, rows)

    on, off = recall("on"), recall("off")
    assert on >= off, f"救援不该降低细结构保留率: on={on:.3f} off={off:.3f}"
    assert on > off + 0.02, f"救援没起作用: on={on:.3f} off={off:.3f}"


# ---------- 5. 确定性 ----------

def test_convert_is_deterministic():
    a = convert_path(SAMPLE, size=64)
    b = convert_path(SAMPLE, size=64)
    assert np.array_equal(a, b), "同一输入同一参数必须逐位一致（调色板/量化不能有随机性）"


# ---------- 6. 裁边 / 留白 / 预览 ----------

def test_trim_and_pad():
    src = np.zeros((40, 40, 4), dtype=np.float32)
    src[10:20, 12:24] = [1.0, 0.5, 0.25, 1.0]
    trimmed = P.trim_pad(src, True, 0)
    assert trimmed.shape[:2] == (10, 12), trimmed.shape
    padded = P.trim_pad(src, True, 3)
    assert padded.shape[:2] == (16, 18), padded.shape
    assert padded[3:-3, 3:-3].max() == 1.0


def test_checker_preview_size():
    small = convert_path(SAMPLE, size=48)
    for scale in (1, 4, 8):
        img = P.checker_preview(small, scale)
        assert img.size == (small.shape[1] * scale, small.shape[0] * scale)


# ---------- 7. --size 是全图宽度（裁边前） ----------

def test_target_grid_matches_requested_width():
    for (h, w), size in (((1152, 896), 128), ((320, 256), 64), ((100, 300), 80)):
        cols, rows = P.target_grid(h, w, size, None)
        assert cols == size
        assert abs(rows - cols * h / w) <= 1, f"{w}x{h} -> {cols}x{rows} 纵横比失真"


def test_pixel_size_mode():
    cols, rows = P.target_grid(320, 256, None, 8)
    assert (cols, rows) == (32, 40)


if __name__ == "__main__":
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
