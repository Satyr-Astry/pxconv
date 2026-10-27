#!/usr/bin/env python
"""pxconv v3 — 透明通道图片 → 像素素材

v3 换了算法核心（默认 --sample core），针对 v2 的三个毛病：
  · 慢/卡      : v2 每换一次参数都要在全分辨率上跑 k-means 聚类指派(百万像素)。
                 v3 一次建好积分图(SAT)，之后每个输出格 O(1) 查表 —— 秒级变毫秒级。
  · 面积平均把半透明像素算进去了 : 半透明边像素带背景色(黑/白毛边)，
                 平均进去边缘就发暗。v3 取色**只用 alpha >= --core-thr 的不透明核心像素**，
                 半透明/透明像素完全不参与计算。
  · 边缘生硬 / 细结构直接没了 : v2 的多数投票、中心点都会丢 1px 细腿/细线。
                 v3 用「精确面积覆盖率 + 细结构救援(hysteresis)」，覆盖率够的细线不会被削掉。

v3 管线:
  1) 预处理一次: alpha / 核心像素掩膜 / 核心RGB 的积分图 (Summed-Area Table)
  2) 每格覆盖率 = SAT 双线性边界求和 (分数边缘, 不要求整数倍缩放, O(1)/格)
  3) 轮廓 = 覆盖率 >= --cov-thr; 细结构救援: 覆盖率 >= 0.4*阈值 且两侧有实心邻居的细线保留
  4) 颜色 = 只对核心像素求面积平均; 没有核心像素的格子(细线/纯边缘)用最近的核心色填充
  5) alpha/轮廓形态/边缘配色/去杂点/裁边 —— 与 v2 同名参数
  6) 调色板收口: 输出色数 <= --colors

用法:
  python pxconv.py in.png -o out.png --size 128 --colors 24                 # 默认 core 模式
  python pxconv.py in.png -o out.png --size 64 --colors 16 --sample core --rescue on
  python pxconv.py in.png -o out.png --size 128 --sample mean              # 老的面积平均(对比用)
  python pxconv.py 素材/ -o 输出/ --size 96 --colors 24                     # 批量
"""
from __future__ import annotations
import argparse, os, sys, glob
import numpy as np
from PIL import Image

VERSION = "3.0"

# ---------- 基础 IO ----------

def load_rgba(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0


def save_rgba(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


def target_grid(h: int, w: int, size: int | None, pixel_size: int | None) -> tuple[int, int]:
    """返回 (cols, rows)"""
    if pixel_size:
        cols = max(1, int(round(w / pixel_size)))
        rows = max(1, int(round(h / pixel_size)))
    elif size:
        cols = max(1, min(size, w))
        rows = max(1, int(round(cols * h / w)))
    else:
        raise SystemExit("需要 --size 或 --pixel-size")
    return cols, rows


def _shift(arr: np.ndarray, i: int, j: int) -> np.ndarray:
    """按邻居偏移 (i,j) 取值(2D/3D, 越界补 0) —— 用切片实现, 比 np.pad 快 6 倍"""
    h, w = arr.shape[:2]
    out = np.zeros_like(arr)
    sy = slice(1, None) if i == 0 else (slice(None, -1) if i == 2 else slice(None))
    dy = slice(0, -1) if i == 0 else (slice(1, None) if i == 2 else slice(None))
    sx = slice(1, None) if j == 0 else (slice(None, -1) if j == 2 else slice(None))
    dx = slice(0, -1) if j == 0 else (slice(1, None) if j == 2 else slice(None))
    out[dy, dx] = arr[sy, sx]
    return out


OFFS8 = [(i, j) for i in range(3) for j in range(3) if not (i == 1 and j == 1)]


# ---------- v3 核心: 积分图 + 精确面积重采样 ----------

def prepare(src: np.ndarray, core_thr: float = 0.7) -> dict:
    """预处理一次(可按图缓存): 建 alpha/核心计数/核心RGB 的积分图

    通道: 0=alpha  1=核心像素个数  2..4=核心像素 RGB 之和
    """
    h, w = src.shape[:2]
    core = (src[..., 3] >= core_thr).astype(np.float32)
    pack = np.zeros((h, w, 5), dtype=np.float32)
    pack[..., 0] = src[..., 3]
    pack[..., 1] = core
    pack[..., 2:] = src[..., :3] * core[..., None]
    T = np.zeros((h + 1, w + 1, 5), dtype=np.float32)
    T[1:, 1:] = np.cumsum(np.cumsum(pack, axis=0), axis=1)
    return dict(T=T, h=h, w=w, core_thr=core_thr)


def _integ_bounds(T: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """在积分图上按浮点坐标 (ys, xs) 网格做双线性取值 -> (len(ys), len(xs), C)"""
    H, W = T.shape[0] - 1, T.shape[1] - 1
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, W - 1)
    ry = np.clip(ys - y0, 0.0, 1.0)[:, None, None]
    rx = np.clip(xs - x0, 0.0, 1.0)[None, :, None]
    a = T[np.ix_(y0, x0)]
    b = T[np.ix_(y0 + 1, x0)]
    c = T[np.ix_(y0, x0 + 1)]
    d = T[np.ix_(y0 + 1, x0 + 1)]
    return (a * ((1 - ry) * (1 - rx)) + b * (ry * (1 - rx))
            + c * ((1 - ry) * rx) + d * (ry * rx))


def cells_from_sat(prep: dict, rows: int, cols: int):
    """每个输出格的: 覆盖率(0~1) / 核心像素数 / 核心像素平均色"""
    T, h, w = prep["T"], prep["h"], prep["w"]
    ys = np.linspace(0.0, h, rows + 1)
    xs = np.linspace(0.0, w, cols + 1)
    B = _integ_bounds(T, ys, xs)                                  # (rows+1, cols+1, 5)
    S = B[1:, 1:] - B[:-1, 1:] - B[1:, :-1] + B[:-1, :-1]        # (rows, cols, 5)
    area = (h / rows) * (w / cols)
    cov = np.clip(S[..., 0] / area, 0.0, 1.0)
    cnt = S[..., 1]
    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    nz = cnt > 0
    rgb[nz] = S[..., 2:][nz] / cnt[nz][:, None]
    return cov, cnt, rgb


def fill_empty_core(rgb: np.ndarray, cnt: np.ndarray, mask: np.ndarray | None = None,
                    rounds: int = 8) -> np.ndarray:
    """没有核心像素的格子 → 用最近的核心色填充(逐层扩散)

    mask != None 时只在 mask(轮廓内)的格子上补色 —— 轮廓外是透明的, 不需要颜色, 省掉大量无用计算。
    """
    have = cnt > 0
    if not have.any():
        return rgb
    need = ~have if mask is None else (mask & ~have)
    if not need.any():
        return rgb
    out = rgb.copy()
    for _ in range(rounds):
        if not need.any():
            break
        acc = np.zeros_like(out)
        nacc = np.zeros(out.shape[:2], dtype=np.float32)
        for (i, j) in OFFS8:
            nb = _shift(have.astype(np.float32), i, j) > 0.5
            m = nb & need
            if not m.any():
                continue
            acc += _shift(out, i, j) * m[..., None]
            nacc += m
        m2 = nacc > 0
        if not m2.any():
            break
        out[m2] = acc[m2] / nacc[m2][:, None]
        have = have | m2
        need = need & ~m2
    return out


def rescue_thin(on: np.ndarray, cov: np.ndarray, thr: float, ratio: float = 0.4) -> np.ndarray:
    """细结构救援: 覆盖率略低于阈值、但两侧有实心邻居的细线/细腿保留下来"""
    weak = (cov >= thr * ratio) & ~on
    if not weak.any():
        return on
    nb = np.zeros(on.shape, dtype=np.float32)
    for (i, j) in OFFS8:
        nb += _shift(on.astype(np.float32), i, j)
    return on | (weak & (nb >= 2))


def sample_core(src: np.ndarray, rows: int, cols: int, prep: dict | None,
                o) -> np.ndarray:
    """v3 主算法: 覆盖率轮廓(带细结构救援) + 只对核心像素取色"""
    prep = prep if prep is not None else prepare(src, float(getattr(o, "core_thr", 0.7)))
    cov, cnt, rgb = cells_from_sat(prep, rows, cols)
    thr = float(getattr(o, "cov_thr", None) or o.alpha_thr)
    on = cov >= thr
    if getattr(o, "rescue", "on") == "on":
        on = rescue_thin(on, cov, thr)
    rgb = fill_empty_core(rgb, cnt, mask=on)      # 只有轮廓内的空格子才补色
    alpha = cov if getattr(o, "alpha_mode", "hard") == "keep" else on.astype(np.float32)
    return np.concatenate([rgb, alpha[..., None]], axis=2)


# ---------- v2 遗留模式 (对比用) ----------

def block_view(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = arr.shape[:2]
    bh, bw = max(1, h // rows), max(1, w // cols)
    return arr[: bh * rows, : bw * cols].reshape(rows, bh, cols, bw, arr.shape[2])


def coverage_alpha(rgba: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return block_view(rgba[..., 3:4], rows, cols).mean(axis=(1, 3))[..., 0]


def mean_blocks(rgba: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """v2 面积平均(预乘): 半透明像素会参与 -> 边缘偏暗, 保留只为对比"""
    a = rgba[..., 3:4]
    premul = np.concatenate([rgba[..., :3] * a, a], axis=2)
    blk = block_view(premul, rows, cols).mean(axis=(1, 3))
    a2 = blk[..., 3:4]
    rgb = np.where(a2 > 1e-6, blk[..., :3] / np.maximum(a2, 1e-6), 0.0)
    return np.concatenate([np.clip(rgb, 0, 1), a2], axis=2)


def kmeans_palette(pixels: np.ndarray, k: int, iters: int = 24, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(pixels)))
    if k == 1:
        return pixels.mean(axis=0, keepdims=True)
    centers = [pixels[rng.integers(len(pixels))]]
    for _ in range(k - 1):
        d = np.min(((pixels[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        d = np.maximum(d, 1e-12)
        centers.append(pixels[rng.choice(len(pixels), p=d / d.sum())])
    C = np.stack(centers)
    for _ in range(iters):
        lab = np.argmin(((pixels[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1)
        for i in range(k):
            m = lab == i
            if m.any():
                C[i] = pixels[m].mean(axis=0)
    return C


def assign(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.empty(len(pixels), dtype=np.int32)
    step = 20000
    for i in range(0, len(pixels), step):
        chunk = pixels[i:i + step]
        out[i:i + step] = np.argmin(
            ((chunk[:, None, :] - palette[None, :, :]) ** 2).sum(-1), axis=1)
    return out


def build_palette(src: np.ndarray, colors: int, core_thr: float = 0.7,
                  method: str = "median") -> np.ndarray:
    """调色板只用**不透明核心像素**拟合(半透明边像素带毛边色, 会污染调色板)

    method: median = PIL 中位切分(C 实现, 快) / kmeans = 感知 k-means(慢但更匀)
    """
    core = src[..., 3] >= core_thr
    px = src[..., :3][core] if core.any() else src[..., :3][src[..., 3] > 0.02]
    if len(px) == 0:
        px = src[..., :3].reshape(-1, 3)
    if len(px) > 120000:
        rng = np.random.default_rng(12345)
        px = px[rng.choice(len(px), 120000, replace=False)]
    if method == "median":
        im = Image.fromarray((np.clip(px, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(-1, 1, 3), "RGB")
        q = im.quantize(colors=max(2, min(256, colors)), method=Image.MEDIANCUT)
        pal = np.asarray(q.getpalette(), dtype=np.float32).reshape(-1, 3)[:colors] / 255.0
        return pal[np.lexsort((pal[:, 2], pal[:, 1], pal[:, 0]))]
    return kmeans_palette(px, colors, iters=12)


def get_palette(prep: dict | None, src: np.ndarray, colors: int, core_thr: float,
                method: str = "median") -> np.ndarray:
    """调色板按 (colors, core_thr, method) 缓存在预处理结果里 —— 改参数时不再重算"""
    if prep is not None:
        key = (int(colors), round(float(core_thr), 3), method)
        if key not in prep.setdefault("palettes", {}):
            prep["palettes"][key] = build_palette(src, colors, core_thr, method)
        return prep["palettes"][key]
    return build_palette(src, colors, core_thr, method)


def snap_to_palette(rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """把可见像素色吸附到调色板 —— 保证色数 <= --colors"""
    h, w = rgba.shape[:2]
    idx = assign(rgba[..., :3].reshape(-1, 3), palette).reshape(h, w)
    vis = rgba[..., 3] > 0.02
    out = rgba.copy()
    out[..., :3] = np.where(vis[..., None], palette[idx], rgba[..., :3])
    return out


def sample_blocks(rgba: np.ndarray, rows: int, cols: int, mode: str,
                  palette: np.ndarray | None) -> np.ndarray:
    """v2 的 mean / majority / center (保留, 效果不如 core)"""
    cov = coverage_alpha(rgba, rows, cols)
    if mode == "mean" or palette is None:
        return mean_blocks(rgba, rows, cols)
    h, w = rgba.shape[:2]
    if mode == "center":
        bh, bw = max(1, h // rows), max(1, w // cols)
        yy = np.clip((np.arange(rows) + 0.5) * bh, 0, h - 1).astype(int)
        xx = np.clip((np.arange(cols) + 0.5) * bw, 0, w - 1).astype(int)
        yy, xx = np.meshgrid(yy, xx, indexing="ij")
        idx = assign(rgba[yy, xx, :3].reshape(-1, 3), palette).reshape(rows, cols)
        return np.concatenate([palette[idx], cov[..., None]], axis=2)
    vis = rgba[..., 3] > 0.05
    bh, bw = max(1, h // rows), max(1, w // cols)
    lab = assign(rgba[..., :3].reshape(-1, 3), palette).reshape(h, w)
    lb = lab[: bh * rows, : bw * cols].reshape(rows, bh, cols, bw)
    vb = vis[: bh * rows, : bw * cols].reshape(rows, bh, cols, bw)
    K = len(palette)
    out_idx = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            l = lb[i, :, j, :].ravel()
            v = vb[i, :, j, :].ravel()
            l = l[v]
            if l.size:
                out_idx[i, j] = int(np.bincount(l, minlength=K).argmax())
    return np.concatenate([palette[out_idx], cov[..., None]], axis=2)


# ---------- alpha / 边缘处理 ----------

def harden_alpha(rgba: np.ndarray, threshold: float) -> np.ndarray:
    out = rgba.copy()
    out[..., 3] = (out[..., 3] >= threshold).astype(np.float32)
    return out


def morph_alpha(rgba: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return rgba
    a = rgba[..., 3] > 0.5
    stack = np.stack([cond_pad(a, i, j) > 0 for i in range(3) for j in range(3)])
    res = rgba.copy()
    res[..., 3] = ((a & stack.all(axis=0)) if mode == "shrink" else (a | stack.any(axis=0))).astype(np.float32)
    return res


def cond_pad(cond: np.ndarray, i: int, j: int) -> np.ndarray:
    return _shift(cond.astype(np.float32), i, j)


def neighbors_color(rgba: np.ndarray, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = cond.shape
    c = np.pad(rgba[..., :3] * cond[..., None], ((1, 1), (1, 1), (0, 0)))
    col = np.zeros((h, w, 3), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for (i, j) in OFFS8:
        col += c[i:i + h, j:j + w]
        cnt += cond_pad(cond, i, j)
    return col, cnt


def edge_color(rgba: np.ndarray, mode: str, passes: int = 2,
               palette: np.ndarray | None = None) -> np.ndarray:
    """avg: 边缘取内圈众数色 | stretch: 内圈色向外推覆盖边缘(去暗边)"""
    if mode == "keep":
        return rgba
    out = rgba.copy()
    for _ in range(passes if mode == "stretch" else 1):
        vis = out[..., 3] > 0.5
        if not vis.any():
            break
        cnt_all = sum(cond_pad(vis, i, j) for (i, j) in OFFS8)
        interior = vis & (cnt_all == 8.0)
        bnd = vis & (cnt_all < 8.0)
        if not bnd.any() or not interior.any():
            break
        if palette is None:
            col_i, cnt_i = neighbors_color(out, interior)
            upd = bnd & (cnt_i > 0)
            if not upd.any():
                break
            out[..., :3] = np.where(upd[..., None], col_i / np.maximum(cnt_i, 1e-6)[..., None], out[..., :3])
            if mode == "avg":
                break
            continue
        K = len(palette)
        h, w = out.shape[:2]
        lab = assign(out[..., :3].reshape(-1, 3), palette).reshape(h, w)
        if mode == "avg":
            counts = np.zeros((h, w, K), dtype=np.float32)
            for (i, j) in OFFS8:
                m = cond_pad(interior, i, j) > 0
                counts[np.arange(h)[:, None], np.arange(w)[None, :], _shift(lab, i, j)] += m
            pick = counts.argmax(-1)
            out[..., :3] = np.where(bnd[..., None], palette[pick], out[..., :3])
            break
        done = np.zeros_like(bnd)
        for (i, j) in OFFS8:
            m = (cond_pad(interior, i, j) > 0) & bnd & ~done
            if m.any():
                out[..., :3] = np.where(m[..., None], _shift(out[..., :3], i, j), out[..., :3])
                done |= m
            if done.all():
                break
    return out


def despike(rgba: np.ndarray, passes: int) -> np.ndarray:
    out = rgba.copy()
    for _ in range(passes):
        a = out[..., 3] > 0.5
        nb = sum(cond_pad(a, i, j) for (i, j) in OFFS8)
        out[..., 3][a & (nb <= 1)] = 0.0
    return out


def trim_pad(rgba: np.ndarray, do_trim: bool, pad: int) -> np.ndarray:
    if do_trim:
        ys, xs = np.where(rgba[..., 3] > 0.5)
        if len(ys):
            rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if pad:
        h, w = rgba.shape[:2]
        out = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.float32)
        out[pad:pad + h, pad:pad + w] = rgba
        rgba = out
    return rgba


# ---------- 诊断 ----------

def fringe_stats(rgba: np.ndarray) -> dict:
    vis = rgba[..., 3] > 0.5
    if not vis.any():
        return dict(boundary=0, grayish=0.0, mean_sat=0.0, edge_delta=0.0)
    cnt_all = sum(cond_pad(vis, i, j) for (i, j) in OFFS8)
    interior = vis & (cnt_all == 8.0)
    bnd = vis & (cnt_all < 8.0)
    edge_delta = 0.0
    col_i, cnt_i = neighbors_color(rgba, interior)
    upd = bnd & (cnt_i > 0)
    if upd.any():
        ref = col_i[upd] / np.maximum(cnt_i[upd], 1e-6)[:, None]
        edge_delta = float((np.abs(rgba[..., :3][upd] - ref).mean(axis=1) * 255).mean())
    px = rgba[..., :3][bnd] * 255
    if not len(px):
        return dict(boundary=0, grayish=0.0, mean_sat=0.0, edge_delta=edge_delta)
    spread = px.max(axis=1) - px.min(axis=1)
    mx = px.max(axis=1)
    sat = np.where(mx > 0, spread / np.maximum(mx, 1e-6), 0.0)
    return dict(boundary=int(bnd.sum()), grayish=float((spread < 40).mean()),
                mean_sat=float(sat.mean()), edge_delta=edge_delta)


# ---------- 主流程 ----------

def convert(src: np.ndarray, cols: int, rows: int, o, prep: dict | None = None) -> np.ndarray:
    sample = getattr(o, "sample", "core")
    colors = int(getattr(o, "colors", 16) or 0)
    core_thr = float(getattr(o, "core_thr", 0.7))
    pmethod = getattr(o, "palette_method", "median")
    palette = get_palette(prep, src, colors, core_thr, pmethod) if colors else None

    if sample == "core":
        rgba = sample_core(src, rows, cols, prep, o)
    else:
        rgba = sample_blocks(src, rows, cols, sample, palette)
        if getattr(o, "alpha_mode", "hard") == "hard":
            rgba = harden_alpha(rgba, o.alpha_thr)

    if getattr(o, "edge", "none") != "none":
        rgba = morph_alpha(rgba, o.edge)
        if getattr(o, "alpha_mode", "hard") == "hard":
            rgba = harden_alpha(rgba, 0.5)
    if getattr(o, "edge_color", "keep") != "keep":
        rgba = edge_color(rgba, o.edge_color, palette=palette)
    if palette is not None:
        rgba = snap_to_palette(rgba, palette)
    if int(getattr(o, "despike", 1) or 0):
        rgba = despike(rgba, int(o.despike))
    return trim_pad(rgba, not getattr(o, "no_trim", False), int(getattr(o, "pad", 0) or 0))


def checker_preview(rgba: np.ndarray, scale: int, cell: int = 8) -> Image.Image:
    h, w = rgba.shape[:2]
    up = np.asarray(Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").resize(
        (w * scale, h * scale), Image.NEAREST), dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:h * scale, 0:w * scale]
    t = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
    bg = 0.78 - 0.13 * t
    a = up[..., 3:4]
    comp = up[..., :3] * a + bg[..., None] * (1 - a)
    img = Image.fromarray((comp * 255).astype(np.uint8), "RGB")
    if scale >= 8:
        px = np.asarray(img).copy()
        px[::scale, :, :] = (px[::scale, :, :] * 0.55).astype(np.uint8)
        px[:, ::scale, :] = (px[:, ::scale, :] * 0.55).astype(np.uint8)
        img = Image.fromarray(px, "RGB")
    return img


def run_one(inp: str, out: str, o) -> tuple[str, int, int, dict]:
    src = load_rgba(inp)
    cols, rows = target_grid(src.shape[0], src.shape[1], o.size, o.pixel_size)
    small = convert(src, cols, rows, o)
    save_rgba(small, out)
    if not o.no_preview:
        base, _ = os.path.splitext(out)
        checker_preview(small, o.preview_scale).save(f"{base}_preview.png")
    return out, small.shape[1], small.shape[0], fringe_stats(small)


# ---------- 归档(已停用): 描边。默认不打描边, 保留备查 ----------
def add_outline(rgba, color=(0, 0, 0), alpha_thr: float = 0.5):
    a = rgba[..., 3] > alpha_thr
    dil = np.zeros_like(a)
    for (i, j) in OFFS8 + [(1, 1)]:
        dil |= cond_pad(a, i, j) > 0
    ring = dil & ~a
    out = rgba.copy()
    for c in range(3):
        out[..., c][ring] = color[c] / 255.0
    out[..., 3][ring] = 1.0
    return out
# ---------- 归档结束 ----------


def main():
    ap = argparse.ArgumentParser(description=f"透明通道图片 -> 像素素材 (v{VERSION})")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--size", type=int, help="输出宽度(像素)")
    ap.add_argument("--pixel-size", type=int, help="每 N 个源像素并成 1 个像素")
    ap.add_argument("--colors", type=int, default=16, help="调色板颜色数(0=不限)")
    ap.add_argument("--sample", choices=["core", "mean", "majority", "center"], default="core",
                    help="core=v3(覆盖率轮廓+核心像素取色,推荐) / 其余为 v2 老算法")
    ap.add_argument("--core-thr", type=float, default=0.7,
                    help="核心像素 alpha 阈值: 只有 alpha>=该值的像素参与取色(默认0.7)")
    ap.add_argument("--palette-method", choices=["median", "kmeans"], default="median",
                    help="调色板拟合方式: median 快(默认) / kmeans 慢但更匀")
    ap.add_argument("--cov-thr", type=float, default=None,
                    help="覆盖率阈值: 覆盖率>=该值的格子成为像素(默认跟随 --alpha-thr)")
    ap.add_argument("--alpha-thr", type=float, default=0.5, help="覆盖率阈值(等价 --cov-thr)")
    ap.add_argument("--rescue", choices=["on", "off"], default="on",
                    help="细结构救援: 保住 1px 细线/细腿(默认 on)")
    ap.add_argument("--alpha-mode", choices=["hard", "keep"], default="hard")
    ap.add_argument("--edge", choices=["none", "shrink", "grow"], default="none")
    ap.add_argument("--edge-color", choices=["keep", "avg", "stretch"], default="keep")
    ap.add_argument("--despike", type=int, default=1, help="清孤立像素遍数, 0=关")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--preview-scale", type=int, default=6)
    ap.add_argument("--no-preview", action="store_true")
    o = ap.parse_args()
    if o.cov_thr is None:
        o.cov_thr = o.alpha_thr

    if os.path.isdir(o.input):
        os.makedirs(o.output, exist_ok=True)
        files = sorted(sum([glob.glob(os.path.join(o.input, e)) for e in
                            ("*.png", "*.webp", "*.jpg", "*.jpeg", "*.bmp")], []))
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            out, w, h, fs = run_one(f, os.path.join(o.output, name + ".png"), o)
            print(f"[OK] {os.path.basename(f)} -> {out}  ({w}x{h})  边缘色差={fs['edge_delta']:.1f}")
    else:
        out, w, h, fs = run_one(o.input, o.output, o)
        print(f"[OK] {o.input} -> {out}  ({w}x{h})  边缘色差={fs['edge_delta']:.1f}")


if __name__ == "__main__":
    main()
