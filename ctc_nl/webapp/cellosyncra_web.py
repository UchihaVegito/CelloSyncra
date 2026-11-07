#!/usr/bin/env python
"""CelloSyncra web app: upload a 2D or 3D time-lapse, ask questions in plain English, see the answer.

Runs on the GPU node next to the Qwen server (see `sbatch run_hpc.sh web`) and reuses
ctc_nl_pipeline.py unchanged:
  upload -> Cellpose-SAM joint 2D+3D generalist (segmentation) -> IoU tracker -> cell table
  question -> Qwen2.5-72B planner (tool calls on the cell table) -> answer drawn on the image

    python cellosyncra_web.py --out <CTC_OUT> --data <CTC_DATA> --work <upload folder> \
        --llm-url http://localhost:8123/v1 --port 8901

Open the printed URL (it contains an access token) through an SSH tunnel.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctc_nl_pipeline as P  # noqa: E402

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402

CFG: argparse.Namespace = None          # filled in main()
app = FastAPI(title="CelloSyncra", docs_url=None, redoc_url=None, openapi_url=None)


# =============================================================================
# access token (the node port is reachable by other cluster users)
# =============================================================================
def require_token(request: Request):
    tok = request.headers.get("x-token") or request.query_params.get("token")
    if not CFG.token or (tok and secrets.compare_digest(tok, CFG.token)):
        return True
    raise HTTPException(401, "missing or wrong access token: open the URL printed in the job log")


AUTH = [Depends(require_token)]


# =============================================================================
# sequences: uploads (seq folder "01") and the CTC test videos (seq "02", existing segmentation)
# =============================================================================
LOCK = threading.RLock()                 # protects SEQS / meta files
SEQS: dict[str, dict] = {}               # id -> meta
SESS: dict[str, tuple] = {}              # id -> (Session, Lock)
JOBS: deque = deque()                    # segmentation queue (ids)
JOB_EVENT = threading.Event()
BACKEND_NAMES = {"cpsam_joint": "our AI (Cellpose-SAM 2D+3D)", "oracle": "ground truth"}


def seq_root(sid):
    return Path(CFG.work) / "seqs" / sid


def save_meta(m):
    if m["kind"] != "upload":
        return
    p = seq_root(m["id"]) / "meta.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1, default=float))
    tmp.replace(p)


def load_uploads():
    for p in sorted((Path(CFG.work) / "seqs").glob("*/meta.json")):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        if m.get("status") in ("queued", "segmenting", "converting"):   # interrupted by a restart -> redo
            m["status"] = "queued"
            JOBS.append(m["id"])
        SEQS[m["id"]] = m
        if m.get("ndim") == 3 and m.get("spacing"):
            P.VOXEL_UM[m["ds_name"]] = tuple(m["spacing"])
    if JOBS:
        JOB_EVENT.set()


def library():
    """CTC test videos (sequence 02) that already have a segmentation in CTC_OUT."""
    out = []
    if not CFG.data:
        return out
    try:
        ds = P.find_datasets(CFG.data)
    except Exception:
        return out
    for name, d in sorted(ds.items()):
        if not (d / "02").is_dir():
            continue
        for be in ("cpsam_joint", "oracle"):
            rd = P.res_dir(CFG.out, be, name, "02")
            if rd.is_dir() and any(rd.glob("*.tif")):
                sid = f"lib-{be}-{name}"
                if sid not in SEQS:
                    SEQS[sid] = dict(id=sid, kind="library", name=f"{name} (test video 02)", ds_name=name,
                                     ds_dir=str(d), seq="02", res_dir=str(rd), backend=be, status="ready",
                                     created=0, ndim=3 if "3D" in name else 2)
                out.append(SEQS[sid])
    return out


def get_meta(sid):
    with LOCK:
        m = SEQS.get(sid)
        if m is None and sid.startswith("lib-"):
            library()
            m = SEQS.get(sid)
    if m is None:
        raise HTTPException(404, "unknown sequence")
    return m


_SEQ_CACHE: dict = {}


def get_seq(m):
    key = (m["id"], m.get("status") == "ready")
    s = _SEQ_CACHE.get(key)
    if s is None:
        if not (Path(m["ds_dir"]) / m["seq"]).is_dir():
            raise HTTPException(409, "the images are still being prepared")
        s = P.Seq(Path(m["ds_dir"]), m["seq"])
        if not s.frames:
            raise HTTPException(409, "no frames found for this sequence")
        _SEQ_CACHE[key] = s
    return s


def get_session(sid):
    m = get_meta(sid)
    if m["status"] != "ready":
        raise HTTPException(409, f"sequence is not ready yet ({m['status']})")
    with LOCK:
        if sid not in SESS:
            seq = get_seq(m)
            seg = P.SegResult(Path(m["res_dir"]))
            if not seg.files:
                raise HTTPException(409, "no segmentation found")
            SESS[sid] = (P.Session(seq, seg, None, m["ds_name"]), threading.Lock())
            if "ndim" not in m or "shape" not in m:
                m.update(ndim=seq.ndim, shape=list(seq.shape), n_frames=len(seq.frames),
                         frames=[min(seq.frames), max(seq.frames)])
        return SESS[sid]


def public(m):
    keys = ("id", "kind", "name", "ndim", "shape", "n_frames", "frames", "status", "progress", "error",
            "created", "backend", "spacing", "cells_per_frame", "seg_seconds", "ds_name", "source")
    d = {k: m.get(k) for k in keys if k in m}
    d["backend_label"] = BACKEND_NAMES.get(m.get("backend"), m.get("backend"))
    if m["kind"] == "upload":
        d["model"] = model_key(m)
        d["model_label"] = CFG.models[d["model"]]["label"]
        d["backend_label"] = d["model_label"]
    return d


# =============================================================================
# upload -> CTC-style folder -> segmentation + tracking (one GPU job at a time)
# =============================================================================
_NUM = re.compile(r"(\d+)")


def natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in _NUM.split(p.name)]


def gray(a):
    """RGB(A) -> first channel, like the pipeline's Seq.image."""
    return a[..., 0] if a.ndim >= 3 and a.shape[-1] in (3, 4) and a.shape[0] > 4 else a


def frames_from_files(files: list[Path], ndim: int) -> list[np.ndarray]:
    """One file per time point, or one multi-page file holding the whole time-lapse."""
    if len(files) == 1:
        a = tifffile.imread(files[0])
        if ndim == 2:
            if a.ndim == 2 or (a.ndim == 3 and a.shape[-1] in (3, 4)):
                return [gray(a)]
            if a.ndim == 3:
                return [a[t] for t in range(a.shape[0])]
            if a.ndim == 4 and a.shape[-1] in (3, 4):
                return [gray(a[t]) for t in range(a.shape[0])]
            raise ValueError(f"cannot read a 2D time-lapse from an array of shape {a.shape}")
        a = gray(a)
        if a.ndim == 3:
            return [a]
        if a.ndim == 4:
            return [a[t] for t in range(a.shape[0])]
        raise ValueError(f"cannot read a 3D time-lapse from an array of shape {a.shape} (expected Z,Y,X or T,Z,Y,X)")
    out = [gray(tifffile.imread(f)) for f in files]
    want = 2 if ndim == 2 else 3
    for f, a in zip(files, out):
        if a.ndim != want:
            raise ValueError(f"expected one {'2D image' if ndim == 2 else '3D z-stack (Z,Y,X)'} per file; "
                             f"{f.name} has shape {a.shape}")
    return out


def expand_uploads(raw: Path) -> list[Path]:
    files = []
    for f in sorted(raw.iterdir(), key=natural):
        if f.suffix.lower() == ".zip":
            dest = raw / (f.stem + "_unzipped")
            dest.mkdir(exist_ok=True)
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    nm = Path(info.filename).name                  # flatten: no path traversal
                    if info.is_dir() or nm.startswith(".") or Path(nm).suffix.lower() not in (".tif", ".tiff"):
                        continue
                    with z.open(info) as src, open(dest / nm, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            files += sorted(dest.iterdir(), key=natural)
        elif f.suffix.lower() in (".tif", ".tiff"):
            files.append(f)
    return files


def model_key(m):
    k = m.get("model") or "thesis"
    return k if k in CFG.models else CFG.default_model


def make_segmenter(seq, key):
    path = CFG.models[key]["path"]
    if CFG.debug_segmenter or not path:  # testing without Cellpose weights / GPU
        from skimage import filters, measure as sm, morphology
        def fn(t):
            im = seq.image(t).astype(np.float32)
            m = im > filters.threshold_otsu(im)
            m = morphology.remove_small_objects(m, 10)
            return sm.label(m).astype(np.int32)
        return fn
    return P.cpsam_predictor(seq, path, gpu=CFG.gpu, batch_size=CFG.batch_size)


def process(sid):
    m = SEQS[sid]
    root = seq_root(sid)
    try:
        if m.get("source"):                 # re-run of another upload with a different model: images exist
            if m["ndim"] == 3:
                P.VOXEL_UM[m["ds_name"]] = tuple(m["spacing"])
            seq = P.Seq(Path(m["ds_dir"]), "01")
            m.update(status="segmenting", progress=[0, len(seq.frames)])
            save_meta(m)
            return _segment(sid, m, seq)
        m.update(status="converting", progress=[0, 0], error=None)
        save_meta(m)
        files = expand_uploads(root / "raw")
        if not files:
            raise ValueError("no .tif/.tiff images found in the upload")
        frames = frames_from_files(files, m["ndim"])
        shapes = {f.shape for f in frames}
        if len(shapes) > 1:
            raise ValueError(f"all time points must have the same size; found {sorted(shapes)}")
        dsd = root / "data" / m["ds_name"] / "01"
        if dsd.exists():
            shutil.rmtree(dsd)
        dsd.mkdir(parents=True)
        digits = max(3, len(str(len(frames) - 1)))
        for t, a in enumerate(frames):
            tifffile.imwrite(dsd / f"t{t:0{digits}d}.tif", a)
        del frames
        if m["ndim"] == 3:
            P.VOXEL_UM[m["ds_name"]] = tuple(m["spacing"])
        seq = P.Seq(root / "data" / m["ds_name"], "01")
        m.update(shape=list(seq.shape), n_frames=len(seq.frames), frames=[min(seq.frames), max(seq.frames)],
                 status="segmenting", progress=[0, len(seq.frames)])
        save_meta(m)
        return _segment(sid, m, seq)
    except Exception as e:
        traceback.print_exc()
        m.update(status="error", error=f"{type(e).__name__}: {e}")
        save_meta(m)


def _segment(sid, m, seq):
    try:
        fn = make_segmenter(seq, model_key(m))
        rd = Path(m["res_dir"])
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        trk = P.OverlapTracker(spacing=seq.spacing)
        t0, counts = time.time(), []
        for k, t in enumerate(sorted(seq.frames)):          # same loop as P.write_res, with progress
            tl = trk.update(t, fn(t))
            tifffile.imwrite(rd / f"mask{t:0{seq.digits}d}.tif", tl.astype(np.uint16))
            counts.append(int(len(np.unique(tl)) - (1 if (tl == 0).any() else 0)))
            m["progress"] = [k + 1, len(seq.frames)]
            if k % 5 == 0 or k + 1 == len(seq.frames):
                save_meta(m)
        (rd / "res_track.txt").write_text(trk.res_track())
        m.update(status="ready", cells_per_frame=counts, seg_seconds=round(time.time() - t0, 1))
        SESS.pop(sid, None)
        P.log(f"web: {m['name']} segmented: {len(counts)} frames, {np.mean(counts):.1f} cells/frame")
    except Exception as e:
        traceback.print_exc()
        m.update(status="error", error=f"{type(e).__name__}: {e}")
    save_meta(m)


def worker():
    while True:
        JOB_EVENT.wait()
        while JOBS:
            sid = JOBS.popleft()
            if sid in SEQS and SEQS[sid].get("status") == "queued":
                process(sid)
        JOB_EVENT.clear()


# =============================================================================
# rendering
# =============================================================================
_RENDER: OrderedDict = OrderedDict()
_U8: OrderedDict = OrderedDict()
COL_OUT, COL_HL, COL_DIV, COL_SEL = (0, 200, 230), (60, 220, 110), (255, 150, 40), (255, 215, 0)


def _lru(cache, key, make, size):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    v = make()
    cache[key] = v
    if len(cache) > size:
        cache.popitem(last=False)
    return v


def ids_param(s):
    return [int(x) for x in (s or "").split(",") if x.strip().lstrip("-").isdigit()]


def render_png(sid, t, view="max", outlines=True, hl=(), div=(), sel=None, labels=True, maxw=None):
    from PIL import Image, ImageDraw
    from skimage.segmentation import find_boundaries
    m = get_meta(sid)
    seq = get_seq(m)
    if t not in seq.frames:
        raise HTTPException(404, f"frame {t} does not exist")
    z = None if view in (None, "", "max") else int(view)

    def u8():
        img = seq.image(t)
        if img.ndim == 3:
            img = img.max(0) if z is None else img[min(max(z, 0), img.shape[0] - 1)]
        return P.normalize_u8(img)
    g = _lru(_U8, (sid, t, z), u8, 64)
    rgb = np.stack([g] * 3, -1).astype(np.float32)
    lab2 = None
    if m["status"] == "ready" and (outlines or hl or div or sel is not None):
        seg = get_session(sid)[0].seg
        lab = seg.label(t) if t in seg.files else None
        if lab is not None:
            if lab.ndim == 3:
                zz = None if z is None else min(max(z, 0), lab.shape[0] - 1)
                lab2 = lab.max(0) if zz is None else lab[zz]
                fp = (lambda ids: np.isin(lab, ids).any(0)) if zz is None else (lambda ids: np.isin(lab[zz], ids))
            else:
                lab2 = lab
                fp = lambda ids: np.isin(lab, ids)
            if outlines:
                rgb[find_boundaries(lab2, mode="inner")] = COL_OUT
            for ids, col, fill in ((list(hl), COL_HL, 0.42), (list(div), COL_DIV, 0.5),
                                   ([sel] if sel is not None else [], COL_SEL, 0.3)):
                if not ids:
                    continue
                mk = fp(ids)
                rgb[mk] = (1 - fill) * rgb[mk] + fill * np.array(col, np.float32)
                rgb[find_boundaries(np.where(mk, lab2, 0), mode="inner")] = col
    im = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    if labels and lab2 is not None and (hl or div):
        d = ImageDraw.Draw(im)
        seg = get_session(sid)[0].seg
        lab = seg.label(t)
        for i in list(hl) + list(div):
            mk = (lab == i).any(0) if lab.ndim == 3 else lab == i
            ys, xs = np.nonzero(mk)
            if len(xs):
                x, y = int(xs.mean()), int(ys.min())
                txt = str(i)
                w = 7 * len(txt) + 6
                d.rectangle([x - w // 2, y - 16, x + w // 2, y - 3], fill=(20, 24, 40))
                d.text((x - w // 2 + 3, y - 15), txt, fill=(255, 255, 255))
    if maxw and im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)))
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=False, compress_level=3)
    return buf.getvalue()


# =============================================================================
# API
# =============================================================================
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((HERE / "web" / "index.html").read_text(), headers={"Cache-Control": "no-store"})


def llm_status():
    if CFG.planner == "offline":
        return {"ok": True, "model": "offline keyword planner (testing)", "url": None}
    try:
        with urllib.request.urlopen(CFG.llm_url.rstrip("/") + "/models", timeout=2) as r:
            names = [d.get("id") for d in json.loads(r.read()).get("data", [])]
        return {"ok": CFG.llm_model in names, "model": CFG.llm_model, "url": CFG.llm_url, "served": names}
    except Exception as e:
        return {"ok": False, "model": CFG.llm_model, "url": CFG.llm_url, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/status", dependencies=AUTH)
def status():
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return {"llm": llm_status(), "gpu": gpu,
            "models": [dict(key=k, label=v["label"], path=v["path"] or "(debug segmenter)") for k, v in CFG.models.items()],
            "default_model": CFG.default_model,
            "seg_model": CFG.models[CFG.default_model]["path"] or "(debug segmenter)",
            "work": str(CFG.work), "out": str(CFG.out), "planner": CFG.planner,
            "queue": list(JOBS)}


@app.get("/api/seqs", dependencies=AUTH)
def list_seqs():
    with LOCK:
        lib = [public(m) for m in library()]
        ups = [public(m) for m in SEQS.values() if m["kind"] == "upload"]
    return {"uploads": sorted(ups, key=lambda m: -m.get("created", 0)), "library": lib}


@app.post("/api/seqs", dependencies=AUTH)
async def upload(files: list[UploadFile] = File(...), name: str = Form(""), ndim: int = Form(2),
                 vz: float = Form(1.0), vy: float = Form(1.0), vx: float = Form(1.0), model: str = Form("")):
    model = model or CFG.default_model
    if model not in CFG.models:
        raise HTTPException(400, f"unknown model {model!r}; available: {list(CFG.models)}")
    if ndim not in (2, 3):
        raise HTTPException(400, "ndim must be 2 or 3")
    if ndim == 3 and min(vz, vy, vx) <= 0:
        raise HTTPException(400, "voxel sizes must be positive")
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    root = seq_root(sid)
    (root / "raw").mkdir(parents=True)
    n = 0
    for f in files:
        nm = Path(f.filename or "").name
        if Path(nm).suffix.lower() not in (".tif", ".tiff", ".zip"):
            continue
        with open(root / "raw" / nm, "wb") as dst:
            while chunk := await f.read(1 << 22):
                dst.write(chunk)
        n += 1
    if not n:
        shutil.rmtree(root)
        raise HTTPException(400, "upload .tif/.tiff images (one per time point, or one multi-page file) or a .zip of them")
    clean = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name.strip())[:60] or "sequence"
    m = dict(id=sid, kind="upload", name=name.strip() or clean, ds_name=f"upload_{sid}_{clean}",
             ndim=ndim, spacing=[vz, vy, vx] if ndim == 3 else None, status="queued", progress=[0, 0],
             created=time.time(), backend="cpsam_joint", seq="01", n_files=n, model=model)
    m["ds_dir"] = str(root / "data" / m["ds_name"])
    m["res_dir"] = str(root / "res")
    with LOCK:
        SEQS[sid] = m
        save_meta(m)
        JOBS.append(sid)
        JOB_EVENT.set()
    return public(m)


@app.post("/api/seqs/{sid}/resegment", dependencies=AUTH)
def resegment(sid: str, body: dict):
    """Same images, other cell-finding model -> a new sequence next to the original (to compare)."""
    src = get_meta(sid)
    if src["kind"] != "upload" or src["status"] != "ready":
        raise HTTPException(400, "only finished uploads can be re-run")
    key = body.get("model")
    if key not in CFG.models:
        raise HTTPException(400, f"unknown model {key!r}")
    base = SEQS.get(src.get("source")) or src
    nid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    root = seq_root(nid)
    root.mkdir(parents=True)
    plain = re.sub(r"\s+·\s+.*$", "", base["name"])
    m = dict(id=nid, kind="upload", name=f"{plain} · {CFG.models[key]['label']}", ds_name=base["ds_name"],
             ds_dir=base["ds_dir"], ndim=base["ndim"], spacing=base.get("spacing"), shape=base.get("shape"),
             n_frames=base.get("n_frames"), frames=base.get("frames"), status="queued", progress=[0, 0],
             created=time.time(), backend="cpsam_joint", seq="01", model=key, source=base["id"],
             res_dir=str(root / "res"))
    with LOCK:
        SEQS[nid] = m
        save_meta(m)
        JOBS.append(nid)
        JOB_EVENT.set()
    return public(m)


@app.get("/api/seqs/{sid}", dependencies=AUTH)
def seq_info(sid: str):
    m = get_meta(sid)
    if m["status"] == "ready":
        sess = get_session(sid)[0]
        m.setdefault("n_frames", len(sess.frames))
        m.setdefault("frames", [sess.frames[0], sess.frames[-1]])
        m.setdefault("shape", list(sess.seq.shape))
        m["ndim"] = sess.ndim
        if m["ndim"] == 3:
            m["spacing"] = list(sess.spacing)
    return public(m)


@app.delete("/api/seqs/{sid}", dependencies=AUTH)
def delete_seq(sid: str):
    m = get_meta(sid)
    if m["kind"] != "upload":
        raise HTTPException(400, "test videos cannot be removed")
    if m["status"] in ("converting", "segmenting"):
        raise HTTPException(409, "wait until segmentation has finished")
    users = [o["name"] for o in SEQS.values() if o.get("source") == sid]
    if users:
        raise HTTPException(409, f"remove the re-runs that use these images first: {', '.join(users)}")
    with LOCK:
        SEQS.pop(sid, None)
        SESS.pop(sid, None)
        trash = Path(CFG.work) / "trash"            # moved, not deleted: can be restored by hand
        trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(seq_root(sid)), str(trash / sid))
    return {"ok": True, "moved_to": str(trash / sid)}


@app.get("/api/seqs/{sid}/frame/{t}.png", dependencies=AUTH)
def frame_png(sid: str, t: int, view: str = "max", outlines: int = 1, hl: str = "", div: str = "",
              sel: int | None = None, labels: int = 1, maxw: int | None = None):
    key = (sid, t, view, outlines, hl, div, sel, labels, maxw, SEQS.get(sid, {}).get("status"))
    png = _lru(_RENDER, key, lambda: render_png(sid, t, view, bool(outlines), ids_param(hl), ids_param(div),
                                                sel, bool(labels), maxw), 128)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


def _cell_record(sess, t, cid):
    f = sess.feats(t).get(cid)
    if f is None:
        return None
    tr = sess.seg.tracks.get(cid, {})
    rec = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items() if k not in ("bbox", "zrange")}
    rec.update(frame=t, ndim=sess.ndim,
               track=dict(begin=tr.get("B"), end=tr.get("E"), parent=tr.get("P") or None,
                          daughters=tr.get("children") or None))
    return rec


@app.get("/api/seqs/{sid}/cell", dependencies=AUTH)
def pick_cell(sid: str, t: int, x: float, y: float, view: str = "max", cid: int | None = None):
    sess, lk = get_session(sid)
    with lk:
        lab = sess.seg.label(t) if t in sess.seg.files else None
        if lab is None:
            raise HTTPException(404, "no segmentation for this frame")
        if cid is None:
            xi, yi = int(round(x)), int(round(y))
            H, W = lab.shape[-2:]
            if not (0 <= xi < W and 0 <= yi < H):
                return {"cell": None}
            col = (lab[:, yi, xi] if view in ("", "max") else lab[int(view), yi, xi][None]) if lab.ndim == 3 \
                else lab[yi, xi][None]
            col = col[col > 0]
            if not col.size:                                       # near miss: closest cell within 6 px
                l2 = (lab.max(0) if view in ("", "max") else lab[int(view)]) if lab.ndim == 3 else lab
                y0, y1, x0, x1 = max(yi - 6, 0), min(yi + 7, H), max(xi - 6, 0), min(xi + 7, W)
                win = l2[y0:y1, x0:x1]
                ys, xs = np.nonzero(win)
                if not len(xs):
                    return {"cell": None}
                k = int(np.argmin((ys + y0 - yi) ** 2 + (xs + x0 - xi) ** 2))
                cid = int(win[ys[k], xs[k]])
            else:
                cid = int(np.bincount(col).argmax())
        return {"cell": _cell_record(sess, t, int(cid))}


def _reply(sess, ans):
    if ans is None:
        return None
    t = ans["frame"]
    feats = sess.feats(t)
    size = P.size_key(sess.ndim)
    cells = []
    for i in ans["cell_ids"][:300]:
        f = feats.get(i)
        cells.append(dict(id=i, size=round(f[size], 2) if f else None, cx=round(f["cx"], 1) if f else None,
                          cy=round(f["cy"], 1) if f else None, cz=round(f["cz"], 1) if f and "cz" in f else None))
    divs = []
    for p in ans.get("division_parent_ids") or []:
        tr = sess.seg.tracks.get(p)
        if tr:
            kids = tr.get("children") or []
            divs.append(dict(parent=p, last_frame=tr["E"], daughters=kids,
                             frame=min((sess.seg.tracks[k]["B"] for k in kids if k in sess.seg.tracks), default=tr["E"] + 1)))
    return dict(frame=t, cell_ids=ans["cell_ids"], count=ans["count"], cells=cells, divisions=divs,
                note=ans.get("note") or "", size_unit="um^3" if sess.ndim == 3 else "px^2")


def history_path(sid):
    m = get_meta(sid)
    if m["kind"] == "upload":
        return seq_root(sid) / "history.jsonl"
    p = Path(CFG.work) / "library_history" / f"{sid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@app.post("/api/seqs/{sid}/ask", dependencies=AUTH)
def ask(sid: str, body: dict):
    q = str(body.get("question", "")).strip()
    if not q:
        raise HTTPException(400, "empty question")
    if len(q) > 2000:
        raise HTTPException(400, "question too long (max 2000 characters)")
    planner = body.get("planner") or CFG.planner
    sess, lk = get_session(sid)
    with lk:                                           # a Session holds state: one question at a time
        sess.reset()
        t0, err, usage = time.time(), None, {}
        try:
            if planner == "offline":
                usage = P.plan_offline(sess, q) or {}
            else:
                usage = P.plan_openai(sess, q, CFG.llm_model, CFG.llm_url, max_steps=CFG.max_steps)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            usage = usage or getattr(sess, "usage", {}) or {}
            if "Connection" in err or "connect" in err.lower():
                err = f"the assistant (Qwen) is not reachable at {CFG.llm_url} - is the LLM server running? ({err})"
        rec = dict(id=uuid.uuid4().hex[:10], time=time.time(), sequence=sid, question=q,
                   planner=CFG.llm_model if planner != "offline" else "offline",
                   seconds=round(time.time() - t0, 2), steps=sess.steps, error=err,
                   answer=_reply(sess, sess.answer), trace=sess.trace,
                   usage={k: v for k, v in (usage or {}).items() if isinstance(v, (int, float))})
    with LOCK, open(history_path(sid), "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return rec


@app.get("/api/seqs/{sid}/history", dependencies=AUTH)
def history(sid: str):
    p = history_path(sid)
    return {"items": [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []}


@app.get("/api/history", dependencies=AUTH)
def all_history():
    items = []
    with LOCK:
        metas = list(SEQS.values())
    for m in metas:
        try:
            p = history_path(m["id"])
        except HTTPException:
            continue
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    r["sequence_name"] = m["name"]
                    r["backend_label"] = (CFG.models[model_key(m)]["label"] if m["kind"] == "upload"
                                          else BACKEND_NAMES.get(m.get("backend"), ""))
                    r.pop("trace", None)
                    items.append(r)
    return {"items": sorted(items, key=lambda r: -r["time"])}


@app.get("/api/seqs/{sid}/cells.csv", dependencies=AUTH)
def cells_csv(sid: str):
    sess, lk = get_session(sid)
    rows, keys = [], []
    with lk:
        for t in sess.frames:
            for cid in sorted(sess.feats(t)):
                r = _cell_record(sess, t, cid)
                tr = r.pop("track")
                r.pop("ndim", None)
                r.update(track_begin=tr["begin"], track_end=tr["end"], parent=tr["parent"] or 0,
                         daughters=" ".join(map(str, tr["daughters"] or [])))
                rows.append(r)
                keys += [k for k in r if k not in keys]
    first = ["frame", "id"]
    keys = first + [k for k in keys if k not in first]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", get_meta(sid)["name"])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}_cells.csv"'})


@app.get("/api/seqs/{sid}/masks.zip", dependencies=AUTH)
def masks_zip(sid: str):
    m = get_meta(sid)
    rd = Path(m["res_dir"])
    if m["status"] != "ready" or not rd.is_dir():
        raise HTTPException(409, "no segmentation yet")
    tmp = Path(tempfile.mkstemp(suffix=".zip", dir=CFG.work)[1])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(rd.iterdir()):
            z.write(f, f"RES/{f.name}")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", m["name"])
    from starlette.background import BackgroundTask
    return FileResponse(tmp, media_type="application/zip", filename=f"{name}_masks_CTC.zip",
                        background=BackgroundTask(lambda: tmp.unlink(missing_ok=True)))


@app.get("/api/seqs/{sid}/history.json", dependencies=AUTH)
def history_json(sid: str):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", get_meta(sid)["name"])
    return JSONResponse(history(sid), headers={"Content-Disposition": f'attachment; filename="{name}_answers.json"'})


# =============================================================================
def find_joint_model(out, backup):
    try:
        return P._model_file(out, "cpsam_ft_joint")
    except FileNotFoundError:
        pass
    if backup:
        hits = sorted(p for p in Path(backup).rglob("cpsam_ft_joint*") if p.is_file())
        if hits:
            return str(hits[-1])
    return None


def main(argv=None):
    global CFG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("CTC_OUT", ""), help="results folder of the 2D+3D experiment (models/, seg/)")
    ap.add_argument("--data", default=os.environ.get("CTC_DATA", ""), help="CTC data roots (for the test-video library)")
    ap.add_argument("--work", default=os.environ.get("WEB_WORK", ""), help="where uploads and their results are stored")
    ap.add_argument("--model", default=os.environ.get("WEB_SEG_MODEL", ""), help="thesis Cellpose-SAM model (default: cpsam_ft_joint in --out or --backup)")
    ap.add_argument("--model-qpi", default=os.environ.get("WEB_SEG_MODEL_QPI", ""), help="model fine-tuned with QPI (cpsam_ft_joint_qpi); if given it is the default")
    ap.add_argument("--backup", default=os.environ.get("BACKUP", ""))
    ap.add_argument("--llm-url", default=os.environ.get("QWEN_URL", "http://localhost:8000/v1"))
    ap.add_argument("--llm-model", default=os.environ.get("OPEN_LLM", "Qwen/Qwen2.5-72B-Instruct-AWQ"))
    ap.add_argument("--planner", choices=["qwen", "offline"], default="qwen")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--token", default=os.environ.get("WEB_TOKEN", ""), help="access token (default: random)")
    ap.add_argument("--no-token", action="store_true", help="no access token (only on a private machine)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--debug-segmenter", action="store_true", help=argparse.SUPPRESS)   # Otsu threshold, tests only
    CFG = ap.parse_args(argv)
    CFG.gpu = not CFG.cpu
    if not CFG.work:
        CFG.work = str(Path(CFG.out or ".") / "web")
    Path(CFG.work, "seqs").mkdir(parents=True, exist_ok=True)
    if not CFG.debug_segmenter and not CFG.model:
        CFG.model = find_joint_model(CFG.out, CFG.backup)
        if not CFG.model:
            raise SystemExit("ERROR: joint model cpsam_ft_joint not found in --out or --backup; pass --model <file>")
    CFG.models = OrderedDict()
    if CFG.model_qpi:
        if not CFG.debug_segmenter and not Path(CFG.model_qpi).is_file():
            raise SystemExit(f"ERROR: --model-qpi file not found: {CFG.model_qpi}")
        CFG.models["qpi"] = dict(label="With QPI (recommended)", path=CFG.model_qpi)
    CFG.models["thesis"] = dict(label="Original (thesis)" if CFG.model_qpi else "Cellpose-SAM 2D+3D", path=CFG.model)
    CFG.default_model = next(iter(CFG.models))
    CFG.token = "" if CFG.no_token else (CFG.token or secrets.token_urlsafe(16))
    load_uploads()
    library()
    threading.Thread(target=worker, daemon=True).start()
    host = os.uname().nodename
    q = f"/?token={CFG.token}" if CFG.token else "/"
    print("=" * 78, flush=True)
    print(f"CelloSyncra web app on {host}:{CFG.port}", flush=True)
    for k, v in CFG.models.items():
        print(f"  cell finder {k:7s}: {v['path'] or 'debug (Otsu)'}  [{v['label']}]{' (default)' if k == CFG.default_model else ''}", flush=True)
    print(f"  assistant          : {CFG.llm_model if CFG.planner == 'qwen' else 'offline'} @ {CFG.llm_url}", flush=True)
    print(f"  uploads            : {CFG.work}", flush=True)
    print(f"  open               : http://localhost:{CFG.port}{q}", flush=True)
    print("=" * 78, flush=True)
    import uvicorn
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="warning")


if __name__ == "__main__":
    main()