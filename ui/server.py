#!/usr/bin/env python
"""pxconv 本地 UI 服务 (纯标准库 http.server, 无额外依赖)

启动:  .venv/Scripts/python.exe ui/server.py [端口]
然后浏览器打开 http://127.0.0.1:8765
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys, time, zipfile, glob, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import pxconv as P                                   # noqa: E402

DEFAULT_PARAMS = dict(
    size=64, pixel_size=0, colors=16, sample="core", alpha_thr=0.5, cov_thr=0.5,
    core_thr=0.7, rescue="on", palette_method="median",
    alpha_mode="hard", edge="none", edge_color="keep", despike=1,
    no_trim=False, pad=0,
)

PRESETS = {
    "推荐(v3核心像素+救援)": dict(sample="core", core_thr=0.7, cov_thr=0.5, rescue="on",
                          edge_color="keep", alpha_mode="hard", edge="none", despike=1, colors=16),
    "v3 + 去暗边(stretch)": dict(sample="core", core_thr=0.7, cov_thr=0.5, rescue="on",
                          edge_color="stretch", alpha_mode="hard", edge="none", despike=1, colors=16),
    "v3 + 抠图光晕收缩":     dict(sample="core", core_thr=0.7, cov_thr=0.5, rescue="on",
                          edge_color="keep", alpha_mode="hard", edge="shrink", despike=1, colors=16),
    "老算法 面积平均(v2)":    dict(sample="mean", edge_color="keep", alpha_mode="hard", edge="none",
                          despike=1, colors=16),
    "老算法 多数投票(v2)":    dict(sample="majority", edge_color="stretch", alpha_mode="hard", edge="none",
                          despike=1, colors=16),
    "柔边立绘(保留羽化)":     dict(sample="core", alpha_mode="keep", edge_color="keep", despike=1, colors=24),
    "低色数复古(8色)":       dict(sample="core", core_thr=0.7, cov_thr=0.5, rescue="on",
                          edge_color="keep", despike=2, colors=8),
}

# 预处理缓存(读图 + 积分图): 改参数时不用重算, 这是 UI 不卡的关键
_PREP_CACHE: dict = {}
_PREP_MAX = 3


# ---------- 核心 ----------

def ns_from(params: dict):
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if k in DEFAULT_PARAMS})
    p["alpha_thr"] = p.get("cov_thr", p["alpha_thr"])
    return argparse.Namespace(**p)


def source_and_prep(spec: dict, core_thr: float):
    """返回 (src, prep, name) —— path 来源走缓存(prep 与 core_thr 绑定)"""
    if spec.get("b64"):
        src = to_array(base64.b64decode(spec["b64"]))
        return src, P.prepare(src, core_thr), (spec.get("name") or "upload"), False
    path = spec.get("path", "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    try:
        st = os.stat(path)
    except OSError as e:
        raise FileNotFoundError(f"读不到文件: {path} ({e})")
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, round(float(core_thr), 3))
    hit = _PREP_CACHE.get(key)
    cached = hit is not None
    if hit is None:
        src = P.load_rgba(path)
        hit = (src, P.prepare(src, core_thr))
        if len(_PREP_CACHE) >= _PREP_MAX:
            _PREP_CACHE.pop(next(iter(_PREP_CACHE)))
        _PREP_CACHE[key] = hit
    return hit[0], hit[1], os.path.basename(path), cached


def to_array(raw: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA")).astype(np.float32) / 255.0


def to_png_b64(arr: np.ndarray, scale: int = 1) -> str:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def metrics(small: np.ndarray) -> dict:
    a8 = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
    vis = a8[a8[..., 3] > 0]
    fs = P.fringe_stats(small)
    return dict(w=int(a8.shape[1]), h=int(a8.shape[0]),
                colors=int(len(np.unique(vis[:, :3], axis=0))) if len(vis) else 0,
                semi=int(((a8[..., 3] > 0) & (a8[..., 3] < 255)).sum()),
                alpha_levels=int(len(np.unique(a8[..., 3]))),
                boundary=fs["boundary"],
                edge_delta=round(fs["edge_delta"], 2),
                grayish=round(fs["grayish"], 3),
                covered=int(round(100.0 * (a8[..., 3] > 0).mean(), 1)))


def convert_array(src: np.ndarray, params: dict, prep: dict | None = None):
    o = ns_from(params)
    cols, rows = P.target_grid(src.shape[0], src.shape[1], o.size or None,
                               o.pixel_size or None)
    t0 = time.perf_counter()
    small = P.convert(src, cols, rows, o, prep=prep)
    ms = (time.perf_counter() - t0) * 1000
    return small, ms


def read_source(spec: dict) -> tuple[np.ndarray, str]:
    """spec: {"b64": ...} 或 {"path": ...}"""
    if spec.get("b64"):
        raw = base64.b64decode(spec["b64"])
        return to_array(raw), spec.get("name") or "upload"
    path = spec.get("path", "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return P.load_rgba(path), os.path.basename(path)


IMG_EXT = ("*.png", "*.webp", "*.jpg", "*.jpeg", "*.bmp")


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    server_version = "pxconv-ui"

    def log_message(self, fmt, *a):        # 静音访问日志
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?", 1)[0]          # 剥掉查询串: UI 支持 /?in=<路径> 直接载入
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/presets":
            self._json(dict(defaults=DEFAULT_PARAMS, presets=PRESETS))
        elif path == "/favicon.ico":
            self._send(204, b"")
        else:
            self._json(dict(ok=False, error="not found"), 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json(dict(ok=False, error=f"请求解析失败: {e}"), 400)
            return
        try:
            fn = {"convert": self.api_convert, "save": self.api_save,
                  "batch": self.api_batch, "preview": self.api_preview}.get(
                self.path.rsplit("/", 1)[-1])
            if not fn:
                self._json(dict(ok=False, error="unknown api"), 404)
                return
            self._json(fn(data))
        except Exception as e:
            self._json(dict(ok=False, error=f"{type(e).__name__}: {e}"), 500)

    # ---- API ----

    def api_convert(self, d):
        params = d.get("params") or {}
        t_all = time.perf_counter()
        src, prep, name, cached = source_and_prep(d.get("source") or {},
                                                 float(params.get("core_thr", 0.7)))
        small, ms = convert_array(src, params, prep)
        out = dict(ok=True, name=name, sprite=to_png_b64(small),
                   metrics=metrics(small), ms=round(ms, 1),
                   total_ms=round((time.perf_counter() - t_all) * 1000, 1), cached=cached)
        if d.get("want_original"):
            out["original"] = to_png_b64(src)
        return out

    def api_preview(self, d):
        """只返回放大后的预览(可选, 前端一般自己用 CSS 放大)"""
        params = d.get("params") or {}
        src, prep, name, cached = source_and_prep(d.get("source") or {},
                                                 float(params.get("core_thr", 0.7)))
        small, ms = convert_array(src, params, prep)
        return dict(ok=True, name=name, preview=to_png_b64(small, int(d.get("scale") or 4)),
                    metrics=metrics(small), ms=round(ms, 1), cached=cached)

    def api_save(self, d):
        b64 = d.get("b64") or ""
        out = (d.get("path") or "").strip()
        if not b64 or not out:
            return dict(ok=False, error="缺少 b64 或 path")
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "wb") as f:
            f.write(base64.b64decode(b64))
        return dict(ok=True, path=out, bytes=os.path.getsize(out))

    def api_batch(self, d):
        folder = (d.get("folder") or "").strip()
        outdir = (d.get("outdir") or "").strip() or os.path.join(folder, "pixel_out")
        if not os.path.isdir(folder):
            return dict(ok=False, error=f"文件夹不存在: {folder}")
        files = sorted(sum([glob.glob(os.path.join(folder, e)) for e in IMG_EXT], []))
        if not files:
            return dict(ok=False, error="文件夹里没找到图片")
        os.makedirs(outdir, exist_ok=True)
        rows, t0 = [], time.perf_counter()
        for f in files:
            try:
                src = P.load_rgba(f)
                small, ms = convert_array(src, d.get("params") or {})
                name = os.path.splitext(os.path.basename(f))[0] + ".png"
                out = os.path.join(outdir, name)
                P.save_rgba(small, out)
                m = metrics(small)
                rows.append(dict(name=os.path.basename(f), out=name, **m, ms=round(ms, 1)))
            except Exception as e:
                rows.append(dict(name=os.path.basename(f), out="", error=str(e)))
        zpath = os.path.join(outdir, "_全部输出.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for r in rows:
                if r.get("out"):
                    z.write(os.path.join(outdir, r["out"]), r["out"])
        return dict(ok=True, rows=rows, outdir=outdir, zip=zpath,
                    seconds=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/"
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError:
        print(f"端口 {a.port} 已被占用 —— UI 可能已经在跑了，直接打开浏览器。")
        if not a.no_open:
            os.system(f'start "" "{url}"')
        return
    print(f"pxconv UI 已启动: {url}")
    print("按 Ctrl+C 退出")
    if not a.no_open:
        threading.Timer(0.8, lambda: os.system(f'start "" "{url}"')).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("已退出")


if __name__ == "__main__":
    main()
