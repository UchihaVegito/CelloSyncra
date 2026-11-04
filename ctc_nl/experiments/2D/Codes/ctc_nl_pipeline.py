#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment
from skimage import measure

warnings.filterwarnings("ignore", category=FutureWarning)

UNMATCHED_ID0 = 60000          # oracle ids for SEG instances without a TRA marker
CLASSES = ["interphase", "dividing", "newborn"]
_IDX = re.compile(r"(\d+)\.tif$")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# =============================================================================
# 1. Data access
# =============================================================================
def find_datasets(root: str | Path, include: list[str] | None = None) -> dict[str, Path]:
    """Map dataset name -> folder that contains 01/, 02/, 01_GT/ ...
    Handles nested layouts such as 2D/DIC-C2DH-HeLa/DIC-C2DH-HeLa/01."""
    root = Path(root)
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.lower().endswith("_test"):
            continue
        if include and d.name not in include:
            continue
        cands = [d, d / d.name] + sorted(p for p in d.iterdir() if p.is_dir())
        for c in cands:
            if (c / "01").is_dir() and (c / "02").is_dir() and (c / "01_GT").is_dir():
                out[d.name] = c
                break
        else:
            log(f"skip {d.name}: no 01/02/01_GT found")
    return out


def _index_dir(d: Path) -> dict[int, Path]:
    if not d.is_dir():
        return {}
    return {int(m.group(1)): p for p in sorted(d.glob("*.tif")) if (m := _IDX.search(p.name))}


def normalize_u8(img: np.ndarray, lo=1.0, hi=99.8) -> np.ndarray:
    img = img.astype(np.float32)
    a, b = np.percentile(img, [lo, hi])
    return (np.clip((img - a) / max(b - a, 1e-6), 0, 1) * 255).astype(np.uint8)


class Seq:
    """One CTC sequence (e.g. DIC-C2DH-HeLa / 02)."""

    def __init__(self, ds_dir: Path, seq: str):
        self.dir, self.seq = Path(ds_dir), seq
        self.name = self.dir.name
        self.frames = _index_dir(self.dir / seq)
        self.gold = _index_dir(self.dir / f"{seq}_GT" / "SEG")
        self.silver = _index_dir(self.dir / f"{seq}_ST" / "SEG")
        self.tra = _index_dir(self.dir / f"{seq}_GT" / "TRA")
        any_f = next(iter(self.frames.values()))
        self.digits = len(_IDX.search(any_f.name).group(1))
        self.tracks = self._lineage()
        self._shape = None

    def _lineage(self):
        f = self.dir / f"{self.seq}_GT" / "TRA" / "man_track.txt"
        tr = {}
        if f.exists():
            for line in f.read_text().split("\n"):
                if line.strip():
                    L, B, E, P = map(int, line.split()[:4])
                    tr[L] = dict(B=B, E=E, P=P, children=[])
            for L, d in tr.items():
                if d["P"] in tr:
                    tr[d["P"]]["children"].append(L)
        return tr

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.image(min(self.frames)).shape
        return self._shape

    def image(self, t):
        return tifffile.imread(self.frames[t])

    def mask(self, t, source="best"):
        for d in {"gold": [self.gold], "silver": [self.silver], "best": [self.gold, self.silver],
                  "dense": [self.silver, self.gold]}[source]:
            if t in d:
                return tifffile.imread(d[t]).astype(np.int32)
        return None

    def tra_mask(self, t):
        return tifffile.imread(self.tra[t]).astype(np.int32) if t in self.tra else None

    def annotated(self, source="best"):
        keys = {"gold": self.gold, "silver": self.silver, "best": {**self.silver, **self.gold}}[source]
        return sorted(set(keys) & set(self.frames))

    def oracle_frame(self, t) -> np.ndarray:
        """Dense reference instances relabelled with TRA ids so labels == GT tracks.
        Silver ST is used first because gold SEG is PARTIAL in several datasets
        (e.g. Fluo-N2DL-HeLa annotates only a subset of nuclei per gold frame);
        gold is used where no silver exists (Fluo-N2DH-SIM+, whose gold is complete)."""
        seg, tra = self.mask(t, "dense"), self.tra_mask(t)
        if seg is None:
            return tra if tra is not None else np.zeros(self.shape, np.int32)
        if tra is None:
            return seg
        out = np.zeros_like(seg)
        used, nxt = set(), UNMATCHED_ID0
        for p in measure.regionprops(seg):
            rr, cc = p.coords[:, 0], p.coords[:, 1]
            ov = tra[rr, cc]
            ov = ov[ov > 0]
            L = int(np.bincount(ov).argmax()) if ov.size else 0
            if L and L not in used:
                used.add(L)
            else:
                L, nxt = nxt, nxt + 1
            out[rr, cc] = L
        for L in np.unique(tra):            # cells missing from SEG: use the marker
            if L and L not in used:
                out[(tra == L) & (out == 0)] = L
        return out


def lineage_class(tracks, L, t) -> int:
    d = tracks.get(int(L))
    if d is None:
        return 0
    if d["children"] and d["E"] - t <= 1:
        return 1
    if d["P"] and t - d["B"] <= 1:
        return 2
    return 0


# =============================================================================
# 2. Features
# =============================================================================
FEATURE_DOC = {
    "area": "area in px^2",
    "perimeter": "perimeter in px",
    "eccentricity": "0 circle .. 1 line",
    "solidity": "area / convex hull area (low = concave outline)",
    "circularity": "4*pi*area/perimeter^2 (1 = circle)",
    "irregularity": "1 - 0.5*(solidity + min(circularity,1)) (0 regular .. 1 irregular)",
    "aspect_ratio": "major / minor axis",
    "mean_intensity": "mean raw intensity",
    "texture_cv": "intensity std / mean inside the cell",
    "cx": "centroid x (px, left->right)", "cy": "centroid y (px, top->bottom)",
    "frames_until_division": "frames until the cell's track ends in a division (only cells that divide; 0 = divides after this frame)",
    "p_dividing": "classifier probability the cell divides within 2 frames (if classifier loaded)",
    "p_newborn": "classifier probability the cell was born within 2 frames (if classifier loaded)",
}


FEATURE_ALIAS = {"x": "cx", "y": "cy", "x_centroid": "cx", "y_centroid": "cy", "centroid_x": "cx",
                 "centroid_y": "cy", "size": "area", "intensity": "mean_intensity", "brightness": "mean_intensity",
                 "elongation": "aspect_ratio", "roundness": "circularity", "texture": "texture_cv"}


def frame_features(label: np.ndarray, image: np.ndarray) -> dict[int, dict]:
    if label.max() == 0:
        return {}

    def _std(m, i):
        return float(np.std(i[m]))

    p = measure.regionprops_table(
        label, intensity_image=image.astype(np.float32),
        properties=("label", "area", "perimeter", "eccentricity", "solidity",
                    "axis_major_length", "axis_minor_length", "centroid", "bbox", "intensity_mean"),
        extra_properties=(_std,))
    out = {}
    for i in range(len(p["label"])):
        a = float(p["area"][i])
        if a < 5:
            continue
        per = max(float(p["perimeter"][i]), 1e-6)
        circ = 4 * math.pi * a / per ** 2
        sol = float(p["solidity"][i])
        mean = float(p["intensity_mean"][i])
        out[int(p["label"][i])] = dict(
            id=int(p["label"][i]), area=a, perimeter=per, eccentricity=float(p["eccentricity"][i]),
            solidity=sol, circularity=circ, irregularity=1 - 0.5 * (sol + min(circ, 1.0)),
            aspect_ratio=float(p["axis_major_length"][i]) / max(float(p["axis_minor_length"][i]), 1e-6),
            mean_intensity=mean, texture_cv=float(p["_std"][i]) / max(abs(mean), 1e-6),
            cx=float(p["centroid-1"][i]), cy=float(p["centroid-0"][i]),
            bbox=[int(p["bbox-1"][i]), int(p["bbox-0"][i]), int(p["bbox-3"][i]), int(p["bbox-2"][i])])
    return out


# =============================================================================
# 3. Tracking
# =============================================================================
class OverlapTracker:
    """IoU-Hungarian linking frame t-1 -> t, with division detection: an unmatched
    new cell within `div_dist` parent diameters of a previous cell whose track was
    continued makes both cells daughters of that previous cell. Produces CTC
    res_track.txt (L B E P)."""

    def __init__(self, iou_min=0.2, div_dist=1.5):
        self.iou_min, self.div_dist = iou_min, div_dist
        self.tracks: dict[int, dict] = {}
        self.prev = None
        self.next_id = 1

    def _new(self, t, parent):
        L = self.next_id
        self.next_id += 1
        self.tracks[L] = dict(B=t, E=t, P=parent)
        return L

    def update(self, t, lab):
        out = np.zeros(lab.shape, np.int32)
        if self.prev is None or lab.max() == 0 or self.prev.max() == 0:
            for p in measure.regionprops(lab):
                out[p.coords[:, 0], p.coords[:, 1]] = self._new(t, 0)
            self.prev = out
            return out
        # pixel coordinates per object, computed once (fast on large, dense frames)
        cco = {p.label: p.coords for p in measure.regionprops(lab)}
        pco = {p.label: p.coords for p in measure.regionprops(self.prev)}
        prv = sorted(pco)
        cur = sorted(cco)
        pidx = {j: b for b, j in enumerate(prv)}
        iou = np.zeros((len(cur), len(prv)))
        for a, i in enumerate(cur):
            ov = self.prev[cco[i][:, 0], cco[i][:, 1]]
            for j, c in zip(*np.unique(ov[ov > 0], return_counts=True)):
                iou[a, pidx[j]] = c / (len(cco[i]) + len(pco[j]) - c)
        r, c = linear_sum_assignment(-iou)
        matched = {a: prv[b] for a, b in zip(r, c) if iou[a, b] >= self.iou_min}
        cc = {i: cco[i].mean(0) for i in cur}
        pc = {j: pco[j].mean(0) for j in prv}
        prad = {j: math.sqrt(len(pco[j]) / math.pi) for j in prv}
        P = np.array([pc[j] for j in prv])
        cont = {L: a for a, L in matched.items()}
        # candidate daughters: unmatched new cells, assigned to their nearest previous cell
        near: dict[int, list[int]] = {}
        for a, i in enumerate(cur):
            if a in matched:
                continue
            d = np.linalg.norm(P - cc[i], axis=1)
            j = prv[int(d.argmin())]
            if d.min() < self.div_dist * 2 * prad[j]:
                near.setdefault(j, []).append(a)
        assigned = {}
        for j, kids in near.items():
            # (a) parent continued by one cell + >=1 unmatched neighbour, or
            # (b) parent lost its match and >=2 unmatched cells appear around it
            if j in cont:
                kids = [cont[j]] + kids
            if len(kids) < 2:
                continue
            kids = sorted(kids, key=lambda a: np.linalg.norm(cc[cur[a]] - pc[j]))[:2]
            for a in kids:
                matched.pop(a, None)
                assigned[a] = self._new(t, j)
            self.tracks[j]["E"] = t - 1
            # the parent track must END here: a cell still linked to j (e.g. a 3rd candidate that
            # was not one of the two daughters) starts a new track, otherwise the parent would
            # continue after its children start (invalid CTC lineage)
            for a2 in [a2 for a2, L in matched.items() if L == j]:
                matched.pop(a2)
                assigned[a2] = self._new(t, 0)
        for a, i in enumerate(cur):
            if a in assigned:
                L = assigned[a]
            elif a in matched:
                L = matched[a]
                self.tracks[L]["E"] = t
            else:
                L = self._new(t, 0)
            out[cco[i][:, 0], cco[i][:, 1]] = L
        self.prev = out
        return out

    def res_track(self):
        return "".join(f"{L} {d['B']} {d['E']} {d['P']}\n" for L, d in sorted(self.tracks.items()))


# =============================================================================
# 4. Segmentation backends  (each returns fn(t) -> instance label image)
# =============================================================================
CPSAM_BASE = os.environ.get("CPSAM_BASE", "cpsam")   # same base weights as the zero-shot run


def cpsam_predictor(seq: Seq, model_path: str | None, gpu=True, batch_size=8, params=None):
    from cellpose import models
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_path or CPSAM_BASE)
    kw = dict(params or {})

    def fn(t):
        out = model.eval(seq.image(t).astype(np.float32), batch_size=batch_size, **kw)
        return np.asarray(out[0]).astype(np.int32)
    return fn


CPSAM_GRID = [dict(cellprob_threshold=c, flow_threshold=f) for c in (-3.0, -2.0, -1.0, 0.0, 1.0) for f in (0.4, 0.8)]


def cpsam_val_split(n):
    idx = list(range(n))
    random.Random(0).shuffle(idx)
    nv = max(1, int(0.1 * n))
    return idx[nv:], idx[:nv]


def cpsam_params_path(out, ds_name):
    return Path(out) / "models" / "cpsam" / f"params_{ds_name}.json"


def st_tune_cpsam(a):
    """Choose cellprob/flow thresholds per dataset on the seq-01 VALIDATION frames
    (the same 10% held out during fine-tuning). Seq 02 is never used."""
    from cellpose import models
    ds = find_datasets(a.data, a.datasets)
    for name, d in ds.items():
        s1 = Seq(d, "01")
        frames = s1.annotated("best")
        _, va = cpsam_val_split(len(frames))
        vf = [frames[i] for i in va]
        model = models.CellposeModel(gpu=True, pretrained_model=cpsam_ft_path(a.out, name))
        scores = []
        for g in CPSAM_GRID:
            sc = [seg_measure(s1.mask(t, "best"),
                              np.asarray(model.eval(s1.image(t).astype(np.float32), batch_size=a.batch_size, **g)[0]))[0]
                  for t in vf]
            scores.append(float(np.nanmean(sc)))
            log(f"  {name} {g}: SEG(seq01 val)={scores[-1]:.3f}")
        best = CPSAM_GRID[int(np.argmax(scores))]
        cpsam_params_path(a.out, name).write_text(json.dumps(dict(best=best, val_frames=vf,
                                                                   grid=list(zip(map(json.dumps, CPSAM_GRID), scores))), indent=1))
        log(f"{name}: tuned {best} (default 0/0.4 gave {scores[CPSAM_GRID.index(dict(cellprob_threshold=0.0, flow_threshold=0.4))]:.3f}, best {max(scores):.3f})")


class Sam3Image:
    """facebookresearch/sam3 image model. Tiled inference + mask NMS because dense
    microscopy fields exceed SAM 3's fixed detection-query budget."""

    def __init__(self, checkpoint=None, conf=0.3, tile=512, overlap=64, device="cuda"):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        kw = {"checkpoint_path": checkpoint} if checkpoint else {}
        self.model = build_sam3_image_model(**kw).to(device).eval()
        self.proc = Sam3Processor(self.model, confidence_threshold=conf)
        self.tile, self.overlap = tile, overlap

    def _tile(self, rgb, prompt):
        import torch
        from PIL import Image
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = self.proc.set_image(Image.fromarray(rgb))
            o = self.proc.set_text_prompt(state=st, prompt=prompt)
        m = o["masks"]
        m = m.squeeze(1) if m.ndim == 4 else m
        return m.float().cpu().numpy() > 0.5, o["scores"].float().cpu().numpy()

    def label(self, img, prompt):
        u8 = normalize_u8(img)
        rgb = np.stack([u8] * 3, -1)
        H, W = u8.shape
        masks, scores = [], []
        step = self.tile - self.overlap
        ys = range(0, max(H - self.overlap, 1), step) if H > self.tile else [0]
        xs = range(0, max(W - self.overlap, 1), step) if W > self.tile else [0]
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + self.tile, H), min(x0 + self.tile, W)
                m, s = self._tile(rgb[y0:y1, x0:x1], prompt)
                for mi, si in zip(m, s):
                    f = np.zeros((H, W), bool)
                    f[y0:y1, x0:x1] = mi
                    masks.append(f)
                    scores.append(si)
        lab = np.zeros((H, W), np.int32)
        k = 1
        for i in np.argsort(-np.array(scores)) if scores else []:
            free = masks[i] & (lab == 0)
            if free.sum() >= 0.6 * masks[i].sum() and free.sum() >= 10:   # NMS by overlap
                lab[free] = k
                k += 1
        return lab


SAM3_PROMPTS = ["cell", "cells", "nucleus", "cell nucleus", "bright spot", "round object", "blob"]


def sam3_choose_prompt(sam: Sam3Image, seq01: Seq, n_frames=5):
    frames = seq01.annotated("gold")[:n_frames] or seq01.annotated("best")[:n_frames]
    scores = {}
    for p in SAM3_PROMPTS:
        s = [seg_measure(seq01.mask(t, "best"), sam.label(seq01.image(t), p))[0] for t in frames]
        scores[p] = float(np.nanmean(s))
        log(f"  sam3 prompt '{p}': SEG(seq01)={scores[p]:.3f}")
    best = max(scores, key=scores.get)
    return best, scores


# =============================================================================
# 5. Segmentation metrics
# =============================================================================
def seg_measure(ref, pred):
    """Official CTC SEG: per reference object, J with the predicted object covering
    >50% of it, else 0; mean over reference objects."""
    ids = [i for i in np.unique(ref) if i]
    js = []
    for i in ids:
        r = ref == i
        lab, cnt = np.unique(pred[r], return_counts=True)
        j = 0.0
        for l, c in zip(lab, cnt):
            if l and c > 0.5 * r.sum():
                s = pred == l
                j = np.logical_and(r, s).sum() / np.logical_or(r, s).sum()
        js.append(j)
    return (float(np.mean(js)) if js else float("nan")), len(ids)


def det_vs_markers(markers, pred):
    """CTC-style detection: a marker is detected by the predicted object that covers
    > 50% of the marker's pixels; each predicted object may detect one marker
    (extra markers on the same object are misses = under-segmentation).
    Returns (tp, fp, fn). Valid when the SEG reference is only partial."""
    used, tp = set(), 0
    for p in measure.regionprops(markers):
        v = pred[p.coords[:, 0], p.coords[:, 1]]
        lab, cnt = np.unique(v, return_counts=True)
        k = int(np.argmax(cnt))
        if lab[k] and cnt[k] > 0.5 * len(v) and lab[k] not in used:
            used.add(int(lab[k]))
            tp += 1
    n_pred = len([i for i in np.unique(pred) if i])
    n_mark = len([i for i in np.unique(markers) if i])
    return tp, n_pred - tp, n_mark - tp


def match_instances(ref, pred, thr=0.5):
    """Hungarian IoU matching -> {pred_id: ref_id} for IoU >= thr."""
    R = [int(i) for i in np.unique(ref) if i]
    P = [int(i) for i in np.unique(pred) if i]
    if not R or not P:
        return {}, R, P
    ri = {r: a for a, r in enumerate(R)}
    pi = {p: b for b, p in enumerate(P)}
    inter = {}
    m = (ref > 0) & (pred > 0)
    for (r, p), c in zip(*np.unique(np.stack([ref[m], pred[m]], 1), axis=0, return_counts=True)):
        inter[(int(r), int(p))] = c
    ra = dict(zip(*np.unique(ref[ref > 0], return_counts=True)))
    pa = dict(zip(*np.unique(pred[pred > 0], return_counts=True)))
    iou = np.zeros((len(R), len(P)))
    for (r, p), c in inter.items():
        iou[ri[r], pi[p]] = c / (ra[r] + pa[p] - c)
    a, b = linear_sum_assignment(-iou)
    return {P[j]: R[i] for i, j in zip(a, b) if iou[i, j] >= thr}, R, P


# =============================================================================
# 6. Stored segmentation results (CTC RES format)
# =============================================================================
class SegResult:
    def __init__(self, res_dir: Path):
        self.dir = Path(res_dir)
        self.files = _index_dir(self.dir)
        self.tracks = {}
        f = self.dir / "res_track.txt"
        if f.exists():
            for line in f.read_text().split("\n"):
                if line.strip():
                    L, B, E, P = map(int, line.split()[:4])
                    self.tracks[L] = dict(B=B, E=E, P=P, children=[])
            for L, d in self.tracks.items():
                if d["P"] in self.tracks:
                    self.tracks[d["P"]]["children"].append(L)
        self._cache = {}

    def label(self, t):
        if t not in self._cache:
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[t] = tifffile.imread(self.files[t]).astype(np.int32)
        return self._cache[t]


def res_dir(out, backend, ds, seq="02"):
    return Path(out) / "seg" / backend / ds / f"{seq}_RES"


# =============================================================================
# 7. Attribute classifier (ViT on cell crops, labels from TRA lineage)
# =============================================================================
CROP = 128


def make_crop(img_u8, lab, cell_id, cx, cy, size):
    half = size // 2
    H, W = img_u8.shape
    pad = half + 1
    ip = np.pad(img_u8, pad, mode="reflect")
    mp = np.pad((lab == cell_id).astype(np.uint8) * 255, pad)
    x, y = int(round(cx)) + pad, int(round(cy)) + pad
    a = ip[y - half:y + half, x - half:x + half]
    m = mp[y - half:y + half, x - half:x + half]
    from PIL import Image
    a = np.asarray(Image.fromarray(a).resize((CROP, CROP), Image.BILINEAR))
    m = np.asarray(Image.fromarray(m).resize((CROP, CROP), Image.NEAREST))
    return np.stack([a, a, m], 0)  # 3 x CROP x CROP uint8


def crop_size_for(seq: Seq, frames=None):
    frames = frames or seq.annotated("best")[:10]
    d = []
    for t in frames:
        m = seq.mask(t, "best")
        if m is not None:
            for p in measure.regionprops(m):
                d.append(2 * math.sqrt(p.area / math.pi))
    return int(np.clip(2.5 * np.median(d) if d else 64, 32, 256))


def collect_crops(seq: Seq, size, max_ratio=3.0, rng=None, frames=None):
    rng = rng or np.random.default_rng(0)
    X, y, meta = [], [], []
    for t in frames or sorted(seq.frames):
        if t not in seq.tra:
            continue
        lab = seq.oracle_frame(t)
        u8 = normalize_u8(seq.image(t))
        for p in measure.regionprops(lab):
            if p.label >= UNMATCHED_ID0:
                continue
            c = lineage_class(seq.tracks, p.label, t)
            X.append((t, p.label, p.centroid[1], p.centroid[0], u8, lab))
            y.append(c)
    y = np.array(y)
    n_min = max(1, int((y > 0).sum()))
    keep = np.flatnonzero(y > 0).tolist()
    inter = np.flatnonzero(y == 0)
    keep += rng.choice(inter, min(len(inter), int(max_ratio * n_min)), replace=False).tolist() if len(inter) else []
    crops = np.stack([make_crop(X[i][4], X[i][5], X[i][1], X[i][2], X[i][3], size) for i in keep]) \
        if keep else np.zeros((0, 3, CROP, CROP), np.uint8)
    meta = [dict(t=X[i][0], id=int(X[i][1])) for i in keep]
    return crops, y[keep], meta


def build_vit(pretrained=True, name="vit_small_patch16_224"):
    import timm
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES))
    except Exception as e:  # no internet on compute node etc.
        log(f"WARNING pretrained weights unavailable ({e}); random init")
        return timm.create_model(name, pretrained=False, num_classes=len(CLASSES))


def _to_input(xb, device):
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(xb, device=device).float() / 255.0
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5


class AttributeClassifier:
    def __init__(self, ckpt, device=None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = build_vit(False, st["arch"])
        self.model.load_state_dict(st["state_dict"])
        self.model.to(self.device).eval()
        self.sizes = st["crop_sizes"]
        # label-shift (prior) correction: training sampled ~max_ratio interphase per mitotic
        # cell; real sequences are far more imbalanced. Priors come from seq-01 lineage only.
        pp = Path(ckpt).with_name("attr_priors.json")
        self.priors = json.loads(pp.read_text()) if pp.exists() else {}

    def predict(self, crops, ds=None, corrected=True):
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(crops), 128):
                out.append(torch.softmax(self.model(_to_input(crops[i:i + 128], self.device)), 1).cpu().numpy())
        p = np.concatenate(out) if out else np.zeros((0, len(CLASSES)))
        pr = self.priors.get(ds) if ds else None
        if corrected and pr and len(p):
            w = np.array(pr["true"]) / np.maximum(np.array(pr["train"]), 1e-9)
            p = p * w
            p = p / p.sum(1, keepdims=True)
        return p


def lineage_priors(seq: Seq, max_ratio=3.0):
    """Class frequencies in seq 01 (all TRA cells) and in the balanced training sample."""
    c = np.zeros(3)
    for t in sorted(seq.tra):
        tra = seq.tra_mask(t)
        for L in np.unique(tra):
            if L:
                c[lineage_class(seq.tracks, L, t)] += 1
    mit = max(c[1] + c[2], 1)
    train = np.array([min(c[0], max_ratio * mit), c[1], c[2]])
    return dict(true=(c / c.sum()).tolist(), train=(train / train.sum()).tolist(), counts=c.astype(int).tolist())


def ensure_priors(out, ds_map, max_ratio=3.0):
    pp = Path(out) / "models" / "attr_priors.json"
    pri = json.loads(pp.read_text()) if pp.exists() else {}
    for name, d in ds_map.items():
        if name not in pri:
            pri[name] = lineage_priors(Seq(d, "01"), max_ratio)
            log(f"{name}: seq-01 class counts {pri[name]['counts']} -> prior correction")
    pp.write_text(json.dumps(pri, indent=1))
    return pri


# =============================================================================
# 8. Agent session over stored results
# =============================================================================
class Session:
    """Tool layer the planners call. Operates on one stored segmentation result."""

    def __init__(self, seq: Seq, seg: SegResult, classifier: AttributeClassifier | None = None, ds_name=""):
        self.seq, self.seg, self.cls, self.ds = seq, seg, classifier, ds_name
        self.frames = sorted(seg.files)
        H, W = seq.shape
        self.H, self.W = H, W
        self._feat = {}
        self.reset()

    def reset(self):
        self.t = self.frames[0]
        self.selection: list[int] = []
        self.answer = None
        self.steps = 0
        self.trace: list[dict] = []

    # ---- data
    def feats(self, t):
        if t not in self._feat:
            if len(self._feat) > 200:
                self._feat.pop(next(iter(self._feat)))
            lab = self.seg.label(t)
            img = self.seq.image(t)
            f = frame_features(lab, img)
            for i, r in f.items():   # lineage: frames until this cell's track ends in a division
                tr = self.seg.tracks.get(i)
                if tr and tr.get("children"):
                    r["frames_until_division"] = int(tr["E"] - t)
            if self.cls is not None and f:
                u8 = normalize_u8(img)
                size = self.cls.sizes.get(self.ds) or int(np.median(list(self.cls.sizes.values())))
                ids = list(f)
                crops = np.stack([make_crop(u8, lab, i, f[i]["cx"], f[i]["cy"], size) for i in ids])
                pr = self.cls.predict(crops, self.ds)
                for i, p in zip(ids, pr):
                    f[i]["p_dividing"], f[i]["p_newborn"] = float(p[1]), float(p[2])
            self._feat[t] = f
        return self._feat[t]

    def _rows(self):
        f = self.feats(self.t)
        return [f[i] for i in self.selection if i in f]

    # ---- tools
    def tool_set_frame(self, t):
        t = int(t)
        self.t = min(self.frames, key=lambda f: abs(f - t))
        self.selection = sorted(self.feats(self.t))   # a new frame starts with ALL its cells selected
        return {"frame": self.t, "first": self.frames[0], "last": self.frames[-1], "n_cells_selected": len(self.selection)}

    def tool_select_all_cells(self):
        self.selection = sorted(self.feats(self.t))
        return {"frame": self.t, "n_cells": len(self.selection), "summary": _summary(self._rows())}

    def _feature(self, f):
        """Map a feature name to a column; unknown names raise (reported to the planner as an error)
        instead of silently matching no cells."""
        f = FEATURE_ALIAS.get(str(f).strip().lower(), str(f).strip().lower())
        valid = [k for k in FEATURE_DOC if not k.startswith("p_") or self.cls is not None]
        if f not in valid:
            raise ValueError(f"unknown feature '{f}'. Valid features: {', '.join(valid)}")
        return f

    def tool_filter(self, conditions=None, roi=None, region=None):
        rows = self._rows()
        pop = list(self.feats(self.t).values())
        applied = []
        if isinstance(region, str) and region.strip().lower() in ("", "none", "all", "any", "whole", "null"):
            region = None                                  # "no region" written as a value
        if region and region not in ("lower", "upper", "left", "right"):
            raise ValueError(f"unknown region '{region}'. Valid: lower, upper, left, right")
        if roi is not None and len(roi) not in (0, 4):
            raise ValueError("roi must be [x0, y0, x1, y1]")
        conds = []
        for c in conditions or []:
            if str(c.get("feature", "")).lower() in ("region", "side", "half"):
                region = region or str(c.get("value", "")).lower()   # common mistake: region written as a condition
            else:
                conds.append(c)
        if region and region not in ("lower", "upper", "left", "right"):
            raise ValueError(f"unknown region '{region}'. Valid: lower, upper, left, right")
        for c in conds:
            f, op, v = self._feature(c["feature"]), c["op"], c["value"]
            if isinstance(v, str) and v.lower().startswith("p") and v[1:].replace(".", "").isdigit():
                vals = [r[f] for r in pop if f in r]
                thr = float(np.percentile(vals, float(v[1:]))) if vals else 0.0
            else:
                thr = float(v)
            fn = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}[op]
            rows = [r for r in rows if f in r and fn(r[f], thr)]
            applied.append(f"{f} {op} {thr:.4g}")
        if region:
            rows = [r for r in rows if in_region(r, region, self.H, self.W)]
            applied.append(f"region {region}")
        if roi:
            x0, y0, x1, y1 = [float(v) for v in roi]
            rows = [r for r in rows if x0 <= r["cx"] <= x1 and y0 <= r["cy"] <= y1]
            applied.append(f"roi {roi}")
        self.selection = [r["id"] for r in rows]
        return {"n": len(rows), "ids": self.selection[:200], "applied": applied}

    def tool_rank(self, feature, order="desc", k=1):
        feature = self._feature(feature)
        rows = sorted(self._rows(), key=lambda r: r.get(feature, -np.inf), reverse=(order == "desc"))[:int(k)]
        self.selection = [r["id"] for r in rows]
        return {"ids": self.selection, feature: [round(r.get(feature, float("nan")), 4) for r in rows]}

    def tool_count(self):
        return {"frame": self.t, "count": len(self.selection)}

    def tool_get_cell(self, cell_id):
        r = self.feats(self.t).get(int(cell_id))
        if r is None:
            return {"error": f"cell {cell_id} not in frame {self.t}"}
        tr = self.seg.tracks.get(int(cell_id), {})
        return {"frame": self.t, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                "track": {"begin": tr.get("B"), "end": tr.get("E"), "parent": tr.get("P") or None,
                          "daughters": tr.get("children") or None}}

    def tool_track(self, ids=None):
        out = {}
        for i in ids or self.selection:
            tr = self.seg.tracks.get(int(i))
            if tr:
                out[int(i)] = {"begin": tr["B"], "end": tr["E"], "parent": tr["P"] or None,
                               "daughters": tr["children"] or None,
                               "divides": bool(tr["children"]),
                               "frames_until_division": (tr["E"] - self.t) if tr["children"] else None}
        return {"tracks": out}

    def tool_divisions(self, start, end):
        ev = division_events(self.seg.tracks, lambda t: self.feats(t), int(start), int(end))
        return {"n": len(ev), "events": ev[:100]}

    def tool_list_features(self):
        return {k: v for k, v in FEATURE_DOC.items() if not k.startswith("p_") or self.cls is not None}

    def tool_submit_answer(self, cell_ids=None, count=None, division_parent_ids=None, note=""):
        if cell_ids is None:      # omitted -> the current selection (documented in the tool schema)
            cell_ids = list(self.selection)
        if isinstance(cell_ids, (int, float)):
            cell_ids = [cell_ids]
        if not isinstance(cell_ids, list):
            raise ValueError("cell_ids must be a list of integers copied from a tool result")
        if count is None:
            count = len(self.selection)
        self.answer = dict(frame=self.t, cell_ids=[int(i) for i in cell_ids],
                           count=int(count),
                           division_parent_ids=[int(i) for i in (division_parent_ids or [])], note=note)
        return {"ok": True}

    def call(self, name, args):
        self.steps += 1
        try:
            out = getattr(self, f"tool_{name}")(**(args or {}))
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        self.trace.append({"tool": name, "args": args or {}, "out": json.dumps(out, default=float)[:300]})
        return out


def in_region(r, region, H, W):
    return {"lower": r["cy"] > H / 2, "upper": r["cy"] <= H / 2,
            "left": r["cx"] <= W / 2, "right": r["cx"] > W / 2}[region]


def _summary(rows):
    if not rows:
        return {}
    s = {}
    for f in ("area", "aspect_ratio", "circularity", "irregularity"):
        v = np.array([r[f] for r in rows])
        s[f] = {"p25": round(float(np.percentile(v, 25)), 3), "median": round(float(np.median(v)), 3),
                "p75": round(float(np.percentile(v, 75)), 3)}
    return s


def division_events(tracks, feats_fn, a, b):
    """Division = parent with >=1 daughter whose first frame is in [a, b].
    Event position = mean centroid of the daughters at their first frame."""
    ev = []
    for L, d in tracks.items():
        kids = d.get("children") or []
        if not kids:
            continue
        t0 = min(tracks[k]["B"] for k in kids)
        if not (a <= t0 <= b):
            continue
        f = feats_fn(t0)
        pts = [(f[k]["cx"], f[k]["cy"]) for k in kids if k in f]
        if not pts:
            continue
        ev.append(dict(parent=int(L), daughters=[int(k) for k in kids], frame=int(t0),
                       x=round(float(np.mean([p[0] for p in pts])), 1),
                       y=round(float(np.mean([p[1] for p in pts])), 1)))
    return sorted(ev, key=lambda e: e["frame"])


# =============================================================================
# 9. Planners
# =============================================================================
TOOLS = [
    {"name": "set_frame", "description": "Go to frame t. The selection is reset to ALL cells of that frame.",
     "input_schema": {"type": "object", "properties": {"t": {"type": "integer"}}, "required": ["t"]}},
    {"name": "select_all_cells", "description": "Select every segmented cell in the current frame. Call this before filter/rank/count.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "filter", "description": "Keep selected cells meeting ALL conditions. value is a number or a percentile of all cells in the frame, e.g. 'p50' for the median. region is one of lower|upper|left|right (image halves by centroid; y grows downward). roi is [x0,y0,x1,y1] on the centroid.",
     "input_schema": {"type": "object", "properties": {
         "conditions": {"type": "array", "items": {"type": "object", "properties": {
             "feature": {"type": "string"}, "op": {"enum": [">", ">=", "<", "<="]},
             "value": {"type": ["number", "string"]}}, "required": ["feature", "op", "value"]}},
         "region": {"enum": ["lower", "upper", "left", "right"]},
         "roi": {"type": "array", "items": {"type": "number"}}}}},
    {"name": "rank", "description": "Keep the top-k selected cells by a feature (order desc = largest values first).",
     "input_schema": {"type": "object", "properties": {"feature": {"type": "string"},
                      "order": {"enum": ["asc", "desc"]}, "k": {"type": "integer"}}, "required": ["feature", "k"]}},
    {"name": "count", "description": "Number of cells in the current selection.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_cell", "description": "Features and track info for one cell id in the current frame.",
     "input_schema": {"type": "object", "properties": {"cell_id": {"type": "integer"}}, "required": ["cell_id"]}},
    {"name": "track", "description": "Track info (begin, end, parent, daughters, frames_until_division) for the selected or given cell ids.",
     "input_schema": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"}}}}},
    {"name": "divisions", "description": "All division events whose daughters first appear in frames [start, end].",
     "input_schema": {"type": "object", "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                      "required": ["start", "end"]}},
    {"name": "list_features", "description": "Feature names and definitions.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "submit_answer", "description": "Submit the final answer exactly once, ALONE in its turn, after you have seen the results you need. cell_ids: cells in the CURRENT frame (omit to submit the current selection); count: a number (omit to submit the size of the current selection); division_parent_ids: parent ids from `divisions`. Copy values from tool results; never guess them.",
     "input_schema": {"type": "object", "properties": {
         "cell_ids": {"type": "array", "items": {"type": "integer"}}, "count": {"type": "integer"},
         "division_parent_ids": {"type": "array", "items": {"type": "integer"}}, "note": {"type": "string"}}}},
]

SYSTEM = """You answer questions about a time-lapse microscopy sequence ({ds}, frames {f0}-{f1}, image {W}x{H} px) using tools only.
Conventions: x grows left->right, y grows top->bottom. "lower/bottom half" = centroid y > {H2}; "upper/top" = y <= {H2}; "left" = x <= {W2}; "right" = x > {W2}.
Size = area. Elongated = aspect_ratio. Round = circularity. Irregular = irregularity. Median cell size = area percentile p50.
"About to divide" = the cell's track ends in a division within the next 2 frames: filter frames_until_division <= 1{cls_note}.
Workflow: set_frame (selects all cells of the frame) -> filter/rank/count -> submit_answer. For division questions use divisions(start,end) and submit the parent ids.
Work step by step: call a tool, read its result, then decide the next call. Never guess numbers or ids.
Always finish with exactly one submit_answer call, alone in its turn. Do not ask questions; take the most reasonable reading."""


def system_prompt(sess: Session):
    return SYSTEM.format(ds=sess.ds, f0=sess.frames[0], f1=sess.frames[-1], W=sess.W, H=sess.H,
                         H2=sess.H // 2, W2=sess.W // 2,
                         cls_note="; p_dividing from the classifier is also available" if sess.cls else "")


def plan_claude(sess: Session, text, model, max_steps=15):
    import anthropic
    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": text}]
    usage = dict(input_tokens=0, output_tokens=0, nudges=0, deferred_submits=0)
    for _ in range(max_steps):
        r = client.messages.create(model=model, max_tokens=1500, system=system_prompt(sess), tools=TOOLS, messages=msgs)
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            if sess.answer is None and not usage.get("nudges"):
                usage["nudges"] = 1
                msgs.append({"role": "user", "content": "You have not called submit_answer yet. Continue using the tools and finish by calling submit_answer."})
                continue
            break
        blocks = [b for b in r.content if b.type == "tool_use"]
        outs = run_turn(sess, [(b.name, b.input or {}) for b in blocks], usage)
        res = [{"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(o, default=float)[:15000]}
               for b, o in zip(blocks, outs)]
        msgs.append({"role": "user", "content": res})
        if sess.answer is not None:
            break
    return usage


def _calls_from_text(content):
    """Recover tool calls a small model wrote as text (e.g. Qwen '<tool_call>{...}{...}</tool_call>'
    with several JSON objects in one block, which vLLM's hermes parser rejects)."""
    calls, dec = [], json.JSONDecoder()
    if not content or "{" not in content:
        return calls
    i = 0
    while True:
        j = content.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(content, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        i = end
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append((obj["name"], args if isinstance(args, dict) else {}))
    return calls


DEFERRED_SUBMIT = {"error": "submit_answer was NOT executed: it was sent in the same turn as other tool calls, "
                            "so it could not use their results. Read the results, then call submit_answer alone, "
                            "copying the actual ids/count from them."}


def run_turn(sess, calls, usage):
    """Execute one model turn [(name, args), ...] in order. A model that plans the whole chain in one
    turn (e.g. filter -> rank -> submit_answer) has to guess the final answer before seeing any
    result; such a submit_answer is held back and the model is asked to submit again."""
    multi = len(calls) > 1 and any(n != "submit_answer" for n, _ in calls)
    outs = []
    for n, a in calls:
        if n == "submit_answer" and multi:
            usage["deferred_submits"] = usage.get("deferred_submits", 0) + 1
            outs.append(DEFERRED_SUBMIT)
        else:
            outs.append(sess.call(n, a))
    return outs


def plan_openai(sess: Session, text, model, base_url, max_steps=15, max_tokens=1024):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]
    tool_names = {t["name"] for t in TOOLS}
    msgs = [{"role": "system", "content": system_prompt(sess)}, {"role": "user", "content": text}]
    usage = dict(input_tokens=0, output_tokens=0, recovered_text_calls=0, nudges=0, truncated=0, deferred_submits=0)
    sess.usage = usage  # kept if a request fails, so the counters still reach the result file
    for _ in range(max_steps):
        # max_tokens caps a reply: a small model can fall into a repetition loop and otherwise
        # generate until the context is full (minutes per question, then a 400 on the next request)
        r = client.chat.completions.create(model=model, messages=msgs, tools=tools, temperature=0,
                                           max_tokens=max_tokens)
        if r.usage:
            usage["input_tokens"] += r.usage.prompt_tokens
            usage["output_tokens"] += r.usage.completion_tokens
        m = r.choices[0].message
        if r.choices[0].finish_reason == "length":
            usage["truncated"] += 1
        if m.tool_calls:
            msgs.append(m.model_dump(exclude_none=True))
            calls = []
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append((tc.function.name, args if isinstance(args, dict) else {}))
            for tc, out in zip(m.tool_calls, run_turn(sess, calls, usage)):
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=float)[:15000]})
        else:
            calls = [(n, a) for n, a in _calls_from_text(m.content) if n in tool_names]
            if not calls:
                msgs.append({"role": "assistant", "content": m.content or ""})
                if sess.answer is None and not usage["nudges"]:
                    usage["nudges"] = 1
                    msgs.append({"role": "user", "content": "You have not called submit_answer yet. Continue using the tools and finish by calling submit_answer."})
                    continue
                break
            # the model meant to call tools but wrote them as text: execute them and report back
            usage["recovered_text_calls"] += len(calls)
            msgs.append({"role": "assistant", "content": m.content or ""})
            results = [{"tool": n, "result": o} for (n, _), o in zip(calls, run_turn(sess, calls, usage))]
            msgs.append({"role": "user", "content": "Tool results:\n" + json.dumps(results, default=float)[:15000]})
        if sess.answer is not None:
            break
    return usage


NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "a single": 1, "single": 1}
SUPERL = [  # (keywords, feature, order)
    (("largest", "biggest", "greatest area"), "area", "desc"),
    (("smallest", "tiniest"), "area", "asc"),
    (("most elongated", "most stretched", "longest"), "aspect_ratio", "desc"),
    (("roundest", "most circular"), "circularity", "desc"),
    (("most irregular", "most ragged"), "irregularity", "desc"),
]
REGW = {"lower": ("lower", "bottom"), "upper": ("upper", "top"), "left": ("left",), "right": ("right",)}


def plan_offline(sess: Session, text):
    """Keyword planner: the ablation baseline without an LLM."""
    s = text.lower()
    m = re.search(r"frames?\s+(\d+)\s*(?:and|to|-|–)\s*(\d+)", s)
    if m and re.search(r"divid|division|mitos|split", s):
        ev = sess.call("divisions", {"start": int(m.group(1)), "end": int(m.group(2))})
        sess.call("submit_answer", {"division_parent_ids": [e["parent"] for e in ev.get("events", [])]})
        return {}
    m = re.search(r"frame\s+(\d+)", s)
    if m:
        sess.call("set_frame", {"t": int(m.group(1))})
    sess.call("select_all_cells", {})
    box = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", s)
    s_wo = re.sub(r"frame\s+\d+", " ", s)
    s_wo = re.sub(r"\[[^\]]*\]", " ", s_wo)
    kw = {}
    for reg, words in REGW.items():
        if any(re.search(rf"\b{w}\b", s_wo) for w in words):
            kw["region"] = reg
            break
    if box:
        kw["roi"] = [float(box.group(i)) for i in range(1, 5)]
    conds = []
    if "median" in s or "typical" in s:
        conds.append({"feature": "area", "op": ">", "value": "p50"})
    if kw or conds:
        sess.call("filter", {"conditions": conds, **kw})
    if re.search(r"about to divide|will (?:soon )?divide|mitosis in the next|undergo mitosis", s):
        if sess.cls is not None:
            sess.call("filter", {"conditions": [{"feature": "p_dividing", "op": ">", "value": 0.5}]})
        else:
            tr = sess.call("track", {})["tracks"]
            sess.selection = [i for i, d in tr.items() if d["divides"] and d["frames_until_division"] is not None
                              and 0 <= d["frames_until_division"] <= 1]
    for words, feat, order in SUPERL:
        if any(w in s_wo for w in words):
            k = 1
            mk = re.search(r"\b(\d+|one|two|three|four|five)\b", s_wo)
            if mk:
                k = int(mk.group(1)) if mk.group(1).isdigit() else NUMW[mk.group(1)]
            sess.call("rank", {"feature": feat, "order": order, "k": k})
            break
    if re.search(r"how many|count|number of", s):
        sess.call("submit_answer", {"count": len(sess.selection)})
    else:
        sess.call("submit_answer", {"cell_ids": list(sess.selection)})
    return {}


# =============================================================================
# 10. Benchmark generation (from seq-02 oracle) and scoring
# =============================================================================
REGION_TXT = {"lower": ("lower half", "bottom part"), "upper": ("upper half", "top part"),
              "left": ("left half", "left side"), "right": ("right half", "right side")}
SUP_TXT = {("area", "desc"): ("largest", "biggest"), ("area", "asc"): ("smallest", "tiniest"),
           ("aspect_ratio", "desc"): ("most elongated", "most stretched-out"),
           ("circularity", "desc"): ("roundest", "most circular"),
           ("irregularity", "desc"): ("most irregularly shaped", "most ragged-looking")}
KW = {1: "", 2: "2 ", 3: "3 "}
KW2 = {1: "the single", 2: "two", 3: "three"}


def generate_bench(sess: Session, n_per_type, seed=0):
    rng = random.Random(seed)
    H, W = sess.H, sess.W
    frames = [t for t in sess.frames if len(sess.feats(t)) >= 5]
    if not frames:
        return []
    qs = []

    def add(qtype, texts, gold):
        for style, txt in enumerate(texts):
            qs.append(dict(qid=f"{sess.ds}:{qtype}:{len(qs)}", dataset=sess.ds, type=qtype, style=style,
                           text=txt, gold=gold))

    def cells(t):
        return list(sess.feats(t).values())

    for _ in range(n_per_type):  # top-k in region
        for _try in range(20):
            t, reg, k = rng.choice(frames), rng.choice(list(REGION_TXT)), rng.choice([1, 2, 3])
            rows = [r for r in cells(t) if in_region(r, reg, H, W)]
            if len(rows) >= k + 1:
                ids = [r["id"] for r in sorted(rows, key=lambda r: -r["area"])[:k]]
                a, b = REGION_TXT[reg]
                noun = "cell" if k == 1 else "cells"
                verb = "is" if k == 1 else "are"
                add("topk_region", [f"Which {verb} the {KW[k]}largest {noun} in the {a} of frame {t}?",
                                    f"In frame {t}, select {KW2[k]} biggest {noun} located in the {b} of the image."],
                    dict(kind="cells", frame=t, ids=ids))
                break
    for _ in range(n_per_type):  # superlatives
        t, (feat, order), k = rng.choice(frames), rng.choice(list(SUP_TXT)), rng.choice([1, 2, 3])
        rows = sorted(cells(t), key=lambda r: r[feat], reverse=(order == "desc"))
        ids = [r["id"] for r in rows[:k]]
        a, b = SUP_TXT[(feat, order)]
        noun = "cell" if k == 1 else "cells"
        add("superlative", [f"Find the {KW[k]}{a} {noun} in frame {t}.",
                            f"Show me {KW2[k]} {b} {noun} in frame {t}."], dict(kind="cells", frame=t, ids=ids))
    for _ in range(n_per_type):  # count
        t = rng.choice(frames)
        add("count_all", [f"How many cells are there in frame {t}?", f"Count the cells visible in frame {t}."],
            dict(kind="count", frame=t, value=len(cells(t))))
    for _ in range(n_per_type):
        t, reg = rng.choice(frames), rng.choice(list(REGION_TXT))
        n = sum(in_region(r, reg, H, W) for r in cells(t))
        a, b = REGION_TXT[reg]
        add("count_region", [f"How many cells are in the {a} of frame {t}?",
                             f"Give me the number of cells located in the {b} of frame {t}."],
            dict(kind="count", frame=t, value=n))
    for _ in range(n_per_type):  # box
        for _try in range(30):
            t = rng.choice(frames)
            bw, bh = int(W * rng.uniform(0.3, 0.5)), int(H * rng.uniform(0.3, 0.5))
            x0, y0 = rng.randint(0, W - bw), rng.randint(0, H - bh)
            box = [x0, y0, x0 + bw, y0 + bh]
            ids = [r["id"] for r in cells(t) if x0 <= r["cx"] <= x0 + bw and y0 <= r["cy"] <= y0 + bh]
            if ids:
                add("box", [f"Select all cells whose centre lies inside the box {box} in frame {t}.",
                            f"In frame {t}, which cells are within region {box} (x0, y0, x1, y1)?"],
                    dict(kind="cells", frame=t, ids=ids))
                break
    for _ in range(n_per_type):  # above median
        t = rng.choice(frames)
        rows = cells(t)
        med = np.median([r["area"] for r in rows])
        add("above_median", [f"Select the cells in frame {t} that are larger than the median cell size.",
                             f"In frame {t}, which cells are bigger than a typical (median-sized) cell?"],
            dict(kind="cells", frame=t, ids=[r["id"] for r in rows if r["area"] > med]))
    all_ev = division_events(sess.seg.tracks, sess.feats, sess.frames[0], sess.frames[-1])
    if all_ev:
        diam = np.median([2 * math.sqrt(r["area"] / math.pi) for r in cells(frames[len(frames) // 2])])
        for _ in range(n_per_type):  # divisions
            e = rng.choice(all_ev)
            span = rng.randint(8, 30)
            a = max(sess.frames[0], e["frame"] - rng.randint(0, span))
            b = min(sess.frames[-1], a + span)
            ev = [x for x in all_ev if a <= x["frame"] <= b]
            add("divisions", [f"Which cells divide between frames {a} and {b}?",
                              f"List the mitotic events (cell splits) that happen from frame {a} to {b}."],
                dict(kind="divisions", start=a, end=b, tol_t=2, tol_px=float(max(15, 0.75 * diam)),
                     events=[dict(frame=x["frame"], x=x["x"], y=x["y"]) for x in ev]))
        # about to divide
        cand = {}
        for L, d in sess.seg.tracks.items():
            if d["children"]:
                for t in (d["E"], d["E"] - 1):
                    if t in sess.seg.files and L in sess.feats(t):
                        cand.setdefault(t, set()).add(L)
        for t in rng.sample(sorted(cand), min(n_per_type, len(cand))):
            add("about_to_divide", [f"Which cells in frame {t} are about to divide (within the next 2 frames)?",
                                    f"In frame {t}, mark the cells that will undergo mitosis in the next couple of frames."],
                dict(kind="cells", frame=t, ids=sorted(cand[t])))
    return qs


class Scorer:
    def __init__(self, oracle: SegResult, seq: Seq):
        self.oracle, self.seq = oracle, seq
        self._m = {}

    def mapping(self, sys_seg: SegResult, key, t):
        k = (key, t)
        if k not in self._m:
            self._m[k] = match_instances(self.oracle.label(t), sys_seg.label(t))[0]
        return self._m[k]

    def score(self, q, ans, sys_seg, key, sess):
        g = q["gold"]
        if ans is None:
            return dict(answered=False, f1=0.0, exact=0.0)
        if g["kind"] == "cells":
            gold = set(g["ids"])
            pred = ans["cell_ids"]
            if ans["frame"] != g["frame"] and pred:
                mapped, unm = set(), len(pred)
            else:
                mp = self.mapping(sys_seg, key, g["frame"])
                mapped = {mp[i] for i in pred if i in mp}
                unm = sum(1 for i in pred if i not in mp)
            tp = len(mapped & gold)
            P = tp / len(pred) if pred else (1.0 if not gold else 0.0)
            R = tp / len(gold) if gold else (1.0 if not pred else 0.0)
            f1 = 2 * P * R / (P + R) if P + R else 0.0
            return dict(answered=True, precision=P, recall=R, f1=f1,
                        exact=float(mapped == gold and unm == 0 and len(pred) == len(gold)))
        if g["kind"] == "count":
            if ans["count"] is None:
                c = len(ans["cell_ids"]) if ans["cell_ids"] else None
            else:
                c = ans["count"]
            if c is None:
                return dict(answered=False, f1=0.0, exact=0.0)
            err = abs(c - g["value"])
            return dict(answered=True, exact=float(err == 0), abs_err=err,
                        rel_err=err / max(g["value"], 1), f1=float(err == 0))
        if g["kind"] == "divisions":
            par = set(ans["division_parent_ids"])
            pred = [e for e in division_events(sys_seg.tracks, sess.feats, g["start"], g["end"]) if e["parent"] in par]
            G = g["events"]
            if not pred and not G:
                return dict(answered=True, precision=1.0, recall=1.0, f1=1.0, exact=1.0)
            if not pred or not G:
                return dict(answered=True, precision=float(not pred), recall=float(not G), f1=0.0, exact=0.0)
            C = np.full((len(pred), len(G)), 1e9)
            for i, p in enumerate(pred):
                for j, q2 in enumerate(G):
                    d = math.hypot(p["x"] - q2["x"], p["y"] - q2["y"])
                    if abs(p["frame"] - q2["frame"]) <= g["tol_t"] and d <= g["tol_px"]:
                        C[i, j] = d
            a, b = linear_sum_assignment(C)
            tp = int(sum(C[i, j] < 1e9 for i, j in zip(a, b)))
            P, R = tp / len(pred), tp / len(G)
            f1 = 2 * P * R / (P + R) if P + R else 0.0
            return dict(answered=True, precision=P, recall=R, f1=f1, exact=float(tp == len(pred) == len(G)))
        raise ValueError(g["kind"])


# =============================================================================
# 11. Stages
# =============================================================================
def st_prepare(a):
    ds = find_datasets(a.data, a.datasets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for name, d in ds.items():
        row = {"path": str(d)}
        for s in ("01", "02"):
            q = Seq(d, s)
            nd = sum(1 for v in q.tracks.values() if v["children"])
            row[s] = dict(frames=len(q.frames), gold=len(q.gold), silver=len(q.silver), tra=len(q.tra),
                          tracks=len(q.tracks), divisions=nd, shape=list(q.shape))
        summ[name] = row
        log(name, json.dumps(row))
    (out / "data_summary.json").write_text(json.dumps(summ, indent=2))


def st_train_cpsam(a):
    from cellpose import models, train
    ds = find_datasets(a.data, a.datasets)
    outdir = Path(a.out) / "models" / "cpsam"
    outdir.mkdir(parents=True, exist_ok=True)
    if a.mode == "per_dataset":          # specialists: one model per dataset
        groups = {n: [n] for n in ds}
    elif a.mode == "pooled":             # generalist: one model on seq 01 of ALL datasets (the deployable model)
        groups = {"pooled": list(ds)}
    else:                                # leave-one-dataset-out: estimates zero-shot performance on unseen data
        groups = {f"lodo_without_{n}": [m for m in ds if m != n] for n in ds}
    for gname, names in groups.items():
        done = sorted(outdir.rglob(f"cpsam_ft_{gname}"))
        if done and not a.overwrite:
            log(f"cpsam fine-tune {gname}: already trained ({done[-1]}), skipping (use --overwrite to retrain)")
            continue
        X, Y = [], []
        for n in names:
            s1 = Seq(ds[n], "01")
            for t in s1.annotated("best"):
                X.append(s1.image(t).astype(np.float32))
                Y.append(s1.mask(t, "best"))
        tr, va = cpsam_val_split(len(X))   # validation = 10% of seq-01 frames (never seq 02)
        log(f"cpsam fine-tune {gname}: {len(tr)} train / {len(va)} val frames from seq 01")
        model = models.CellposeModel(gpu=True, pretrained_model=CPSAM_BASE)
        res = train.train_seg(model.net, train_data=[X[i] for i in tr], train_labels=[Y[i] for i in tr],
                              test_data=[X[i] for i in va], test_labels=[Y[i] for i in va],
                              learning_rate=a.lr, weight_decay=a.weight_decay, n_epochs=a.epochs,
                              batch_size=a.batch_size, model_name=f"cpsam_ft_{gname}", save_path=str(outdir))
        path = res[0] if isinstance(res, (tuple, list)) else res
        log("saved", path)


def _model_file(out, model_name):
    hits = sorted((Path(out) / "models" / "cpsam").rglob(model_name))
    hits = [h for h in hits if h.is_file()]
    if not hits:
        raise FileNotFoundError(f"model {model_name} not found under {Path(out) / 'models' / 'cpsam'}; run train_cpsam")
    return str(hits[-1])


def cpsam_ft_path(out, ds_name):
    """Specialist model of this dataset (no silent fallback to another model)."""
    return _model_file(out, f"cpsam_ft_{ds_name}")


def cpsam_backend_model(out, backend, ds_name):
    return {"cpsam_ft": lambda: cpsam_ft_path(out, ds_name),
            "cpsam_pooled": lambda: _model_file(out, "cpsam_ft_pooled"),
            "cpsam_lodo": lambda: _model_file(out, f"cpsam_ft_lodo_without_{ds_name}"),
            "cpsam": lambda: None}[backend]()


def write_res(seq: Seq, fn, rdir: Path):
    rdir.mkdir(parents=True, exist_ok=True)
    trk = OverlapTracker()
    for t in sorted(seq.frames):
        lab = fn(t)
        tl = trk.update(t, lab)
        tifffile.imwrite(rdir / f"mask{t:0{seq.digits}d}.tif", tl.astype(np.uint16))
    (rdir / "res_track.txt").write_text(trk.res_track())


def write_oracle(seq: Seq, rdir: Path):
    rdir.mkdir(parents=True, exist_ok=True)
    for t in sorted(seq.frames):
        tifffile.imwrite(rdir / f"mask{t:0{seq.digits}d}.tif", seq.oracle_frame(t).astype(np.uint16))
    (rdir / "res_track.txt").write_text(
        "".join(f"{L} {d['B']} {d['E']} {d['P']}\n" for L, d in sorted(seq.tracks.items())))


def st_segment(a):
    ds = find_datasets(a.data, a.datasets)
    sam = None
    meta_path = Path(a.out) / "seg" / a.backend / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for name, d in ds.items():
        s2 = Seq(d, a.seq)
        rd = res_dir(a.out, a.backend, name, a.seq)
        t0 = time.time()
        if a.backend == "oracle":
            write_oracle(s2, rd)
        elif a.backend == "noisy_oracle":    # selftest only: emulates an imperfect segmenter
            from skimage.morphology import disk, erosion
            rng = np.random.default_rng(1)

            def fn(t):
                lab = s2.oracle_frame(t)
                lab = erosion(lab, disk(1))
                for i in np.unique(lab):
                    if i and rng.random() < 0.1:
                        lab[lab == i] = 0
                return _relabel(lab)
            write_res(s2, fn, rd)
        elif a.backend in ("cpsam", "cpsam_ft", "cpsam_pooled", "cpsam_lodo"):
            # specialists use thresholds tuned on their seq-01 validation frames; the generalist and
            # leave-one-out models use DEFAULT thresholds, as on a new dataset without annotations
            path = cpsam_backend_model(a.out, a.backend, name)
            meta.setdefault(name, {})["model"] = path or CPSAM_BASE
            pp = cpsam_params_path(a.out, name)
            params = json.loads(pp.read_text())["best"] if (a.backend == "cpsam_ft" and pp.exists()) else None
            meta.setdefault(name, {})["cellpose_params"] = params or "default"
            write_res(s2, cpsam_predictor(s2, path, batch_size=a.batch_size, params=params), rd)
        elif a.backend in ("sam3_zs", "sam3_ft"):
            if sam is None:
                ck = a.sam3_ckpt if a.backend == "sam3_ft" else None
                if a.backend == "sam3_ft" and not ck:
                    raise SystemExit("sam3_ft needs --sam3-ckpt (fine-tune with the sam3 repo trainer on export_sam3_coco output)")
                sam = Sam3Image(ck, conf=a.sam3_conf, tile=a.sam3_tile)
            prompt = a.sam3_prompt
            if not prompt:
                prompt, sweep = sam3_choose_prompt(sam, Seq(d, "01"))
                meta[name] = dict(prompt=prompt, sweep_seq01=sweep)
            write_res(s2, lambda t: sam.label(s2.image(t), prompt), rd)
            log(f"{name}: SAM3 prompt '{prompt}' (chosen on seq 01)")
        else:
            raise SystemExit(f"unknown backend {a.backend}")
        meta.setdefault(name, {})["seconds"] = round(time.time() - t0, 1)
        log(f"{a.backend}/{name}: {len(s2.frames)} frames -> {rd} ({time.time() - t0:.0f}s)")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


def st_predict(a):
    """Zero-shot use on a NEW, unannotated time-lapse: a folder of 2D .tif frames (sorted by name).
    Segments with the generalist (pooled) model and default thresholds, tracks, and writes
    CTC-format masks + res_track.txt + a PNG overlay per frame. No ground truth needed."""
    from cellpose import models
    from PIL import Image
    files = sorted(Path(a.images).glob("*.tif")) + sorted(Path(a.images).glob("*.tiff"))
    if not files:
        raise SystemExit(f"no .tif frames in {a.images}")
    model_path = a.model or _model_file(a.out, "cpsam_ft_pooled")
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    dest = Path(a.dest)
    (dest / "masks").mkdir(parents=True, exist_ok=True)
    (dest / "overlays").mkdir(parents=True, exist_ok=True)
    trk = OverlapTracker()
    rows = []
    for t, f in enumerate(files):
        img = tifffile.imread(f)
        img = img if img.ndim == 2 else img[..., 0] if img.shape[-1] <= 4 else img[0]
        lab = np.asarray(model.eval(img.astype(np.float32), batch_size=a.batch_size)[0]).astype(np.int32)
        tl = trk.update(t, lab)
        tifffile.imwrite(dest / "masks" / f"mask{t:03d}.tif", tl.astype(np.uint16))
        u8 = normalize_u8(img)
        rgb = np.stack([u8] * 3, -1)
        Image.fromarray(_contours(rgb, tl, (40, 200, 90))).save(dest / "overlays" / f"{f.stem}.png")
        for r in frame_features(tl, img).values():
            rows.append(dict(frame=t, file=f.name, **{k: v for k, v in r.items() if k != "bbox"}))
        log(f"{f.name}: {len([i for i in np.unique(tl) if i])} cells")
    (dest / "masks" / "res_track.txt").write_text(trk.res_track())
    import pandas as pd
    pd.DataFrame(rows).to_csv(dest / "cells.csv", index=False)
    log(f"model {model_path}\n -> {dest}/masks (CTC format), overlays/*.png, cells.csv")


def _relabel(lab):
    out = np.zeros_like(lab)
    for k, i in enumerate([i for i in np.unique(lab) if i], 1):
        out[lab == i] = k
    return out


def st_eval_seg(a):
    import csv
    ds = find_datasets(a.data, a.datasets)
    rows, per_frame = [], []
    for be in a.backends:
        for name, d in ds.items():
            rd = res_dir(a.out, be, name, a.seq)
            if not rd.exists():
                log(f"missing {rd}; skip")
                continue
            s2, sr = Seq(d, a.seq), SegResult(rd)
            segs, w, tp, fp, fn = [], [], 0, 0, 0
            for t in s2.annotated("gold"):          # SEG: gold only (CTC definition; partial gold is fine)
                seg_t, n = seg_measure(s2.mask(t, "gold"), sr.label(t))
                segs.append(seg_t)
                w.append(n)
                per_frame.append(dict(backend=be, dataset=name, t=t, SEG=seg_t, n_gold=n))
            for t in sorted(set(s2.tra) & set(sr.files)):   # DET: every frame, vs TRA markers
                dtp, dfp, dfn = det_vs_markers(s2.tra_mask(t), sr.label(t))
                tp, fp, fn = tp + dtp, fp + dfp, fn + dfn
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            row = dict(backend=be, dataset=name, SEG=np.average(segs, weights=w) if segs else float("nan"),
                       DET_precision=prec, DET_recall=rec, DET_F1=2 * prec * rec / max(prec + rec, 1e-9),
                       gold_frames=len(segs), gold_objects=int(sum(w)))
            if a.official and be != "oracle":
                row.update(official_ctc(d / f"{a.seq}_GT", rd))
            rows.append(row)
            log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    out = Path(a.out) / "results" / "segmentation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if per_frame:                                   # per-frame SEG, for diagnosing failures over time
        import pandas as pd
        pf = out.with_name("segmentation_per_frame.csv")
        old_pf = pd.read_csv(pf) if pf.exists() else pd.DataFrame()
        if len(old_pf):
            old_pf = old_pf[~old_pf.backend.isin(a.backends)]
        pd.concat([old_pf, pd.DataFrame(per_frame)]).to_csv(pf, index=False)
    old = []
    if out.exists():
        with open(out) as f:
            old = [r for r in csv.DictReader(f) if r["backend"] not in a.backends]
    keys = sorted({k for r in old + rows for k in r}, key=lambda k: (k not in ("backend", "dataset"), k))
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(old + rows)
    log("wrote", out)


def official_ctc(gt_dir: Path, rdir: Path):
    """Official TRA/DET/SEG via py-ctcmetrics (pip install py-ctcmetrics)."""
    exe = shutil.which("ctc_evaluate")
    if not exe:
        return {"official": "py-ctcmetrics not installed"}
    try:
        r = subprocess.run([exe, "--gt", str(gt_dir), "--res", str(rdir)], capture_output=True, text=True, timeout=3600)
        txt = r.stdout + r.stderr
        res = {}
        for k in ("TRA", "DET", "SEG", "LNK", "OP_CTB", "OP_CSB"):
            m = re.search(rf"['\"]?{k}['\"]?\s*[:=]\s*([0-9.]+)", txt)
            if m:
                res[f"official_{k}"] = float(m.group(1))
        return res or {"official_raw": txt[-500:]}
    except Exception as e:
        return {"official": f"error {e}"}


def st_train_cls(a):
    import torch
    import torch.nn as nn
    ds = find_datasets(a.data, a.datasets)
    rng = np.random.default_rng(0)
    Xtr, ytr, Xva, yva, sizes = [], [], [], [], {}
    for name, d in ds.items():
        s1 = Seq(d, "01")
        sizes[name] = crop_size_for(s1)
        fr = sorted(s1.frames)
        cut = fr[int(0.85 * len(fr))]          # temporal hold-out inside seq 01 for model selection
        X, y, _ = collect_crops(s1, sizes[name], a.max_ratio, rng, [t for t in fr if t < cut])
        Xv, yv, _ = collect_crops(s1, sizes[name], a.max_ratio, rng, [t for t in fr if t >= cut])
        log(f"{name}: crop {sizes[name]}px  train {np.bincount(y, minlength=3)}  val {np.bincount(yv, minlength=3)}")
        Xtr.append(X), ytr.append(y), Xva.append(Xv), yva.append(yv)
    Xtr, ytr, Xva, yva = map(np.concatenate, (Xtr, ytr, Xva, yva))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_vit(not a.no_pretrained, a.arch).to(dev)
    cnt = np.bincount(ytr, minlength=3).astype(float)
    wts = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / 3, dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=wts)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, out = -1, Path(a.out) / "models" / "attr_vit.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), a.batch_size):
            b = perm[i:i + a.batch_size]
            xb = Xtr[b].copy()
            if rng.random() < 0.5:
                xb = xb[..., ::-1]
            if rng.random() < 0.5:
                xb = xb[..., ::-1, :]
            xb = np.rot90(xb, rng.integers(4), axes=(2, 3)).copy()
            loss = lossf(model(_to_input(xb, dev)), torch.as_tensor(ytr[b], device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        pv = _predict(model, Xva, dev)
        m = cls_metrics(yva, pv)
        log(f"epoch {ep + 1}/{a.epochs} loss {tot / max(len(perm), 1):.4f}  val macroF1 {m['macro_f1']:.3f}  balAcc {m['balanced_acc']:.3f}")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(dict(state_dict=model.state_dict(), arch=a.arch, crop_sizes=sizes, classes=CLASSES,
                            val=m, epoch=ep + 1), out)
    log(f"best val macro-F1 {best:.3f} -> {out}")


def _predict(model, X, dev):
    import torch
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), 128):
            out.append(torch.softmax(model(_to_input(X[i:i + 128], dev)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def cls_metrics(y, p):
    pred = p.argmax(1) if len(p) else np.array([], int)
    res, f1s, recs = {}, [], []
    for c, n in enumerate(CLASSES):
        tp = int(((pred == c) & (y == c)).sum())
        P = tp / max((pred == c).sum(), 1)
        R = tp / max((y == c).sum(), 1)
        F = 2 * P * R / max(P + R, 1e-9)
        res[n] = dict(precision=P, recall=R, f1=F, support=int((y == c).sum()))
        f1s.append(F)
        recs.append(R)
    res["macro_f1"], res["balanced_acc"] = float(np.mean(f1s)), float(np.mean(recs))
    res["confusion"] = [[int(((y == i) & (pred == j)).sum()) for j in range(3)] for i in range(3)]
    try:
        from sklearn.metrics import roc_auc_score
        res["auroc_dividing"] = float(roc_auc_score(y == 1, p[:, 1])) if 0 < (y == 1).sum() < len(y) else None
        from sklearn.metrics import average_precision_score
        ok = 0 < (y == 1).sum() < len(y)
        res["auprc_dividing"] = float(average_precision_score(y == 1, p[:, 1])) if ok else None
        res["prevalence_dividing"] = float((y == 1).mean()) if len(y) else None
    except Exception:
        pass
    return res


def st_eval_cls(a):
    import torch
    ds = find_datasets(a.data, a.datasets)
    st = torch.load(Path(a.out) / "models" / "attr_vit.pt", map_location="cpu", weights_only=False)
    ensure_priors(a.out, ds)
    clf = AttributeClassifier(Path(a.out) / "models" / "attr_vit.pt")
    allres = {}
    for name, d in ds.items():
        s2 = Seq(d, "02")
        size = st["crop_sizes"].get(name) or crop_size_for(s2)
        # all cells (no subsampling) -> realistic class imbalance
        X, y, _ = collect_crops(s2, size, max_ratio=1e9)
        p = clf.predict(X, name, corrected=True)
        allres[name] = cls_metrics(y, p)
        allres[name]["uncorrected"] = {k: v for k, v in cls_metrics(y, clf.predict(X, name, corrected=False)).items()
                                       if k in ("macro_f1", "balanced_acc", "confusion")}
        allres[name]["n_train_mitotic_seq01"] = clf.priors.get(name, {}).get("counts")
        log(name, json.dumps({k: allres[name][k] for k in ("macro_f1", "balanced_acc", "auroc_dividing", "auprc_dividing")}))
    out = Path(a.out) / "results" / "classifier_seq02.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(allres, indent=2))
    log("wrote", out)


def st_make_bench(a):
    ds = find_datasets(a.data, a.datasets)
    qs = []
    for name, d in ds.items():
        s2 = Seq(d, "02")
        rd = res_dir(a.out, "oracle", name)
        if not rd.exists():
            write_oracle(s2, rd)
        sess = Session(s2, SegResult(rd), None, name)
        q = generate_bench(sess, a.n_per_type, a.seed)
        log(f"{name}: {len(q)} queries")
        qs += q
    out = Path(a.out) / "bench" / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(q) + "\n" for q in qs))
    log(f"wrote {len(qs)} queries -> {out}")
    write_handwritten_template(qs, out.parent / "handwritten_template.csv")


def st_run_bench(a):
    ds = find_datasets(a.data, a.datasets)
    qs = [json.loads(l) for l in (Path(a.out) / "bench" / "queries.jsonl").read_text().split("\n") if l.strip()]
    qs = [q for q in qs if q["dataset"] in ds]
    if a.limit:
        qs = qs[:a.limit]
    clf = None
    if a.use_classifier and (Path(a.out) / "models" / "attr_vit.pt").exists():
        ensure_priors(a.out, ds)
        clf = AttributeClassifier(Path(a.out) / "models" / "attr_vit.pt")
    tag = a.planner if a.planner == "offline" else f"{a.planner}-{a.model.replace('/', '_')}"
    tag += "+cls" if clf else ""
    outp = Path(a.out) / "bench" / f"results_{a.backend}_{tag}.jsonl"
    done = set()
    if outp.exists() and not a.overwrite:
        # resume: keep finished answers, drop records that failed with an error (e.g. server down) so they rerun
        old = [json.loads(l) for l in outp.read_text().split("\n") if l.strip()]
        keep = [r for r in old if not r.get("error")]
        if len(keep) < len(old):
            log(f"resume: re-running {len(old) - len(keep)} questions that ended with an error")
            outp.write_text("".join(json.dumps(r, default=float) + "\n" for r in keep))
        done = {r["qid"] for r in keep}
    sessions, scorers = {}, {}
    with open(outp, "a" if not a.overwrite else "w") as f:
        for q in qs:
            if q["qid"] in done:
                continue
            name = q["dataset"]
            if name not in sessions:
                s2 = Seq(ds[name], "02")
                sr = SegResult(res_dir(a.out, a.backend, name))
                if not sr.files:
                    raise SystemExit(f"no segmentation for {a.backend}/{name}; run segment first")
                sessions[name] = Session(s2, sr, clf, name)
                scorers[name] = Scorer(SegResult(res_dir(a.out, "oracle", name)), s2)
            sess = sessions[name]
            sess.reset()
            t0, usage, err = time.time(), {}, None
            try:
                if a.planner == "offline":
                    usage = plan_offline(sess, q["text"])
                elif a.planner == "claude":
                    usage = plan_claude(sess, q["text"], a.model, a.max_steps)
                else:
                    usage = plan_openai(sess, q["text"], a.model, a.base_url, a.max_steps)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                usage = usage or getattr(sess, "usage", {})
            sc = scorers[name].score(q, sess.answer, sess.seg, a.backend, sess)
            rec = dict(qid=q["qid"], dataset=name, type=q["type"], style=q["style"], backend=a.backend,
                       planner=tag, steps=sess.steps, seconds=round(time.time() - t0, 2),
                       answer=sess.answer, error=err, text=q["text"], trace=sess.trace,
                       **(usage or {}), **sc)
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
    log("wrote", outp)


def st_report(a):
    import pandas as pd
    res = Path(a.out) / "results"
    res.mkdir(parents=True, exist_ok=True)
    lines = ["# Results\n"]
    seg = res / "segmentation.csv"
    if seg.exists():
        df = pd.read_csv(seg)
        cols = [c for c in df.columns if c in ("SEG", "DET_F1", "official_TRA", "official_DET", "official_SEG")]
        piv = df.pivot_table(index="dataset", columns="backend", values=cols).round(3)
        piv.to_csv(res / "table_segmentation.csv")
        lines += ["## 1. Segmentation / tracking (seq 02)\n", _md(piv), ""]
    cj = res / "classifier_seq02.json"
    if cj.exists():
        c = json.loads(cj.read_text())
        rows = [{"dataset": k, "macro_F1": v["macro_f1"], "balanced_acc": v["balanced_acc"],
                 "AUROC_dividing": v.get("auroc_dividing"), "AUPRC_dividing": v.get("auprc_dividing"),
                 "prevalence_dividing": v.get("prevalence_dividing"), "F1_dividing": v["dividing"]["f1"],
                 "F1_newborn": v["newborn"]["f1"]} for k, v in c.items()]
        t = pd.DataFrame(rows).set_index("dataset").round(3)
        t.to_csv(res / "table_classifier.csv")
        lines += ["## Attribute classifier (seq 02, all cells)\n", _md(t), ""]
    files = sorted((Path(a.out) / "bench").glob("results_*.jsonl"))
    if files:
        df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
        df["score"] = df["f1"].fillna(0)
        overall = df.groupby(["planner", "backend"]).agg(
            n=("qid", "count"), score=("score", "mean"), exact=("exact", "mean"),
            answered=("answered", "mean"), steps=("steps", "mean"), sec=("seconds", "mean"),
            **({"count_rel_err": ("rel_err", "mean")} if "rel_err" in df else {})).round(3)
        overall.to_csv(res / "table_bench_overall.csv")
        byds = df.pivot_table(index=["planner", "backend"], columns="dataset", values="score").round(3)
        byds.to_csv(res / "table_bench_by_dataset.csv")
        bytype = df.pivot_table(index=["planner", "backend"], columns="type", values="score").round(3)
        bytype.to_csv(res / "table_bench_by_type.csv")
        df["style_name"] = df["style"].map(STYLE_NAMES)
        bystyle = df.pivot_table(index=["planner", "backend"], columns="style_name", values="score").round(3)
        bystyle.to_csv(res / "table_bench_by_paraphrase.csv")
        lines += ["## 2. Language benchmark (score = F1 for selections/divisions, exact for counts)\n", _md(overall), "",
                  "### By dataset\n", _md(byds), "", "### By query type\n", _md(bytype), "", "### By wording style (template / template paraphrase / LLM free paraphrase / hand-written)\n",
                  _md(bystyle), ""]
        # 3. error attribution
        att = []
        for pl, g in df.groupby("planner"):
            o = g[g.backend == "oracle"].set_index("qid")["score"]
            for be, gb in g[g.backend != "oracle"].groupby("backend"):
                s = gb.set_index("qid")["score"]
                common = o.index.intersection(s.index)
                if len(common):
                    att.append(dict(planner=pl, backend=be, n=len(common), oracle_score=o[common].mean(),
                                    backend_score=s[common].mean(),
                                    planner_loss=1 - o[common].mean(),
                                    segmentation_loss=o[common].mean() - s[common].mean()))
        if att:
            t = pd.DataFrame(att).round(3)
            t.to_csv(res / "table_error_attribution.csv", index=False)
            lines += ["## 3. Error attribution\n", _md(t.set_index(["planner", "backend"])), ""]
    (res / "report.md").write_text("\n".join(lines))
    log("wrote", res / "report.md")
    print("\n".join(lines))


def _md(df):
    try:
        return df.to_markdown()
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def st_export_sam3_coco(a):
    """COCO export of seq 01 (all datasets) for fine-tuning SAM 3.1 with the sam3 repo trainer."""
    from PIL import Image
    ds = find_datasets(a.data, a.datasets)
    out = Path(a.out) / "sam3_coco"
    coco = dict(images=[], annotations=[], categories=[dict(id=1, name="cell")])
    iid = aid = 0
    for name, d in ds.items():
        s1 = Seq(d, "01")
        (out / name).mkdir(parents=True, exist_ok=True)
        for t in s1.annotated("best"):
            lab = s1.mask(t, "best")
            fn = f"{name}/t{t:04d}.png"
            Image.fromarray(normalize_u8(s1.image(t))).convert("RGB").save(out / fn)
            iid += 1
            coco["images"].append(dict(id=iid, file_name=fn, height=lab.shape[0], width=lab.shape[1]))
            for p in measure.regionprops(lab):
                m = lab == p.label
                aid += 1
                y0, x0, y1, x1 = p.bbox
                coco["annotations"].append(dict(id=aid, image_id=iid, category_id=1, iscrowd=0,
                                                area=int(p.area), bbox=[x0, y0, x1 - x0, y1 - y0],
                                                segmentation=_rle(m)))
    (out / "train.json").write_text(json.dumps(coco))
    log(f"{iid} images, {aid} instances -> {out}/train.json")


def _rle(m):
    flat = m.reshape(-1, order="F").astype(np.uint8)
    ch = np.flatnonzero(np.diff(np.concatenate([[0], flat])) != 0)
    return {"counts": np.diff(np.concatenate([[0], ch, [len(flat)]])).tolist(), "size": list(m.shape)}


# =============================================================================
# 11b. Free-text queries: LLM paraphrases (style 2) and hand-written prompts (style 3)
# =============================================================================
STYLE_NAMES = {0: "template", 1: "template paraphrase", 2: "LLM free paraphrase", 3: "hand-written"}

PARA_SYSTEM = """You rewrite benchmark questions about time-lapse microscopy images into natural, varied
free text, the way different biologists would actually type them into an analysis tool.
Rules:
- Keep the meaning EXACTLY: same frame number(s), same box coordinates (copy them verbatim),
  same number of cells asked for, same image region, same ranking criterion, same time window.
- Do NOT reuse the original sentence structure. Vary vocabulary (e.g. biggest/largest/most spread out,
  bottom/lower/southern part), register (terse, chatty, formal), and form (question, command, request).
- No explanations, no quotes. Reply with ONLY a JSON list: [{"qid": "...", "text": "..."}, ...]"""


def required_numbers(q):
    """Numbers a paraphrase must keep: frame(s), division window and box coordinates."""
    g = q["gold"]
    req = set()
    if "frame" in g:
        req.add(str(g["frame"]))
    if g["kind"] == "divisions":
        req |= {str(g["start"]), str(g["end"])}
    m = re.search(r"\[([^\]]+)\]", q["text"])
    if m:
        req |= {x.strip() for x in m.group(1).split(",")}
    return req


def paraphrase_ok(q, text):
    nums = set(re.findall(r"\d+", text))
    return required_numbers(q) <= nums and len(text) > 10


def _llm_json(planner, model, base_url, system, user):
    if planner == "claude":
        import anthropic
        r = anthropic.Anthropic().messages.create(model=model, max_tokens=4000, system=system,
                                                  messages=[{"role": "user", "content": user}])
        txt = "".join(b.text for b in r.content if b.type == "text")
    else:
        from openai import OpenAI
        r = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")).chat.completions.create(
            model=model, temperature=0.9, max_tokens=4000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        txt = r.choices[0].message.content
    m = re.search(r"\[.*\]", txt, re.S)
    return json.loads(m.group(0)) if m else []


def st_paraphrase_bench(a):
    """Adds style-2 free paraphrases (LLM) and style-3 hand-written prompts to queries.jsonl.
    The gold answer is copied from the source query, so scoring is unchanged."""
    qpath = Path(a.out) / "bench" / "queries.jsonl"
    qs = [json.loads(l) for l in qpath.read_text().split("\n") if l.strip()]
    by_id = {q["qid"]: q for q in qs}
    have = {q["qid"] for q in qs}
    new = []
    if a.planner != "none":
        src = [q for q in qs if q["style"] == 0 and f"{q['qid']}:free" not in have]
        for i in range(0, len(src), a.batch):
            chunk = src[i:i + a.batch]
            user = "Rewrite each question once:\n" + json.dumps([{"qid": q["qid"], "text": q["text"]} for q in chunk])
            for attempt in range(3):
                try:
                    out = {o["qid"]: o["text"] for o in _llm_json(a.planner, a.model, a.base_url, PARA_SYSTEM, user)
                           if isinstance(o, dict) and "qid" in o and "text" in o}
                    break
                except Exception as e:
                    log(f"paraphrase batch {i}: {type(e).__name__}: {e} (retry {attempt + 1})")
                    time.sleep(5 * (attempt + 1))
            else:
                out = {}
            kept = 0
            for q in chunk:
                t = out.get(q["qid"])
                if t and paraphrase_ok(q, t):
                    new.append({**q, "qid": f"{q['qid']}:free", "style": 2, "text": t.strip(),
                                "source_qid": q["qid"], "paraphraser": f"{a.planner}:{a.model}"})
                    kept += 1
            log(f"paraphrases {i + len(chunk)}/{len(src)}: kept {kept}/{len(chunk)} (rejected = lost a number)")
    if a.user_file:
        import csv
        n = 0
        with open(a.user_file, newline="") as f:
            for k, row in enumerate(csv.DictReader(f)):
                txt = (row.get("your_text") or row.get("text") or "").strip()
                qid = (row.get("qid") or "").strip()
                if not txt or qid not in by_id:
                    continue
                nq = f"{qid}:hand{k}"
                if nq in have:
                    continue
                new.append({**by_id[qid], "qid": nq, "style": 3, "text": txt, "source_qid": qid})
                n += 1
        log(f"hand-written prompts added: {n}")
    with open(qpath, "a") as f:
        for q in new:
            f.write(json.dumps(q) + "\n")
    log(f"added {len(new)} free-text queries -> {qpath}")


def write_handwritten_template(qs, path: Path):
    """CSV for writing your own wording of a subset of queries (one per dataset x type)."""
    import csv
    seen, rows = set(), []
    for q in qs:
        key = (q["dataset"], q["type"])
        if q["style"] == 0 and key not in seen:
            seen.add(key)
            rows.append(dict(qid=q["qid"], dataset=q["dataset"], type=q["type"], original=q["text"], your_text=""))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["qid", "dataset", "type", "original", "your_text"])
        w.writeheader()
        w.writerows(rows)
    log(f"hand-written template ({len(rows)} rows) -> {path}  (fill 'your_text', keep numbers identical)")


# =============================================================================
# 11c. Figures (PNG) for the thesis
# =============================================================================
# Validated categorical order (light surface); every bar also carries a value label
# because slots 3-5 are below 3:1 contrast against the surface.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948", "#008300"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OK_C, BAD_C, MISS_C, PAR_C = (12, 163, 12), (208, 59, 59), (250, 178, 25), (42, 120, 214)
BACKEND_ORDER = ["oracle", "cpsam_ft", "cpsam_pooled", "cpsam_lodo", "cpsam", "sam3_zs", "sam3_ft", "noisy_oracle"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
    return plt


def _color_map(keys):
    """Colour follows the entity (fixed order), never its rank."""
    order = [k for k in BACKEND_ORDER if k in keys] + sorted(k for k in keys if k not in BACKEND_ORDER)
    return {k: PAL[i % len(PAL)] for i, k in enumerate(order)}, order


def _grouped_bars(ax, groups, series, values, cmap, ylabel, ymax=1.0, fmt="{:.2f}"):
    n = len(series)
    w = 0.8 / max(n, 1)
    for j, s_ in enumerate(series):
        xs = [i - 0.4 + w * (j + 0.5) for i in range(len(groups))]
        ys = [values.get((g, s_), np.nan) for g in groups]
        bars = ax.bar(xs, ys, width=w * 0.92, color=cmap[s_], label=s_, zorder=2)
        for b, y in zip(bars, ys):
            if not np.isnan(y):
                ax.text(b.get_x() + b.get_width() / 2, y + ymax * 0.01, fmt.format(y), ha="center", va="bottom",
                        fontsize=7, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=0)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig_segmentation(res, fdir, plt, dpi):
    import pandas as pd
    f = res / "segmentation.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    cmap, order = _color_map(set(df.backend))
    groups = sorted(df.dataset.unique())
    metrics = [m for m in ("SEG", "DET_F1", "official_TRA") if m in df and df[m].notna().any()]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(7, 1.8 * len(groups)), 2.8 * len(metrics)), squeeze=False)
    for ax, m in zip(axes[:, 0], metrics):
        vals = {(r.dataset, r.backend): float(r[m]) for _, r in df.iterrows() if pd.notna(r[m])}
        _grouped_bars(ax, groups, order, vals, cmap, m)
        ax.set_title({"SEG": "SEG on sequence 02 (vs gold SEG)",
                      "DET_F1": "DET-F1 on sequence 02 (vs TRA markers, all frames)",
                      "official_TRA": "TRA on sequence 02 (official CTC measure)"}.get(m, m), loc="left")
    axes[0, 0].legend(ncol=len(order), loc="lower left", bbox_to_anchor=(0, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "fig1_segmentation.png", dpi=dpi)
    plt.close(fig)


def fig_classifier(res, fdir, plt, dpi):
    f = res / "classifier_seq02.json"
    if not f.exists():
        return
    c = json.loads(f.read_text())
    names = list(c)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.3), squeeze=False)
    for ax, n in zip(axes[0], names):
        cm = np.array(c[n]["confusion"], float)
        rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j])}\n{rn[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if rn[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), CLASSES, rotation=30)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true (lineage)")
        ax.grid(False)
        ax.set_title(f"{n}\nmacro-F1 {c[n]['macro_f1']:.2f} · AUROC(div) "
                     f"{(c[n].get('auroc_dividing') or float('nan')):.2f}", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "fig2_classifier_confusion.png", dpi=dpi)
    plt.close(fig)


def _bench_df(out):
    import pandas as pd
    files = sorted((Path(out) / "bench").glob("results_*.jsonl"))
    if not files:
        return None
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df["score"] = df["f1"].fillna(0)
    df["style_name"] = df["style"].map(STYLE_NAMES)
    return df


def fig_bench(out, fdir, plt, dpi):
    df = _bench_df(out)
    if df is None:
        return
    cmap, border = _color_map(set(df.backend))
    planners = sorted(df.planner.unique())
    # fig3: overall score per planner, bars per backend
    vals = df.groupby(["planner", "backend"]).score.mean().to_dict()
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(planners)), 3.4))
    _grouped_bars(ax, planners, border, vals, cmap, "mean score (F1 / exact count)")
    ax.set_title("Language benchmark: overall score by planner and segmentation backend", loc="left")
    ax.legend(ncol=len(border), loc="lower left", bbox_to_anchor=(0, 1.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "fig3_bench_overall.png", dpi=dpi)
    plt.close(fig)
    # fig4: heatmap by query type
    piv = df.pivot_table(index=["planner", "backend"], columns="type", values="score")
    fig, ax = plt.subplots(figsize=(1.0 * piv.shape[1] + 3, 0.42 * piv.shape[0] + 1.6))
    ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.55 else INK)
    ax.set_xticks(range(piv.shape[1]), [c.replace("_", " ") for c in piv.columns], rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), [f"{p} · {b}" for p, b in piv.index])
    ax.grid(False)
    ax.set_title("Score by query type (1 = perfect)", loc="left")
    fig.tight_layout()
    fig.savefig(fdir / "fig4_bench_by_type.png", dpi=dpi)
    plt.close(fig)
    # fig5: wording robustness, one panel per backend, bars = planner
    styles = [STYLE_NAMES[k] for k in sorted(STYLE_NAMES) if STYLE_NAMES[k] in set(df.style_name)]
    backends = [b for b in border if b in set(df.backend)]
    pmap = {p: PAL[i % len(PAL)] for i, p in enumerate(planners)}
    fig, axes = plt.subplots(1, len(backends), figsize=(max(4.2 * len(backends), 6), 3.4), squeeze=False, sharey=True)
    for ax, b in zip(axes[0], backends):
        sub = df[df.backend == b]
        vals = sub.groupby(["style_name", "planner"]).score.mean().to_dict()
        _grouped_bars(ax, styles, [p for p in planners if p in set(sub.planner)], vals, pmap, "mean score")
        ax.set_xticklabels([s.replace(" ", "\n") for s in styles])
        ax.set_title(f"backend: {b}", loc="left")
    axes[0, 0].legend(ncol=min(len(planners), 4), loc="lower left", bbox_to_anchor=(0, 1.1), fontsize=8)
    fig.suptitle("Robustness to wording: template vs free text", x=0.01, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(fdir / "fig5_bench_by_wording.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    # fig6: error attribution, 100% stacked bars
    rows = []
    for pl, g in df.groupby("planner"):
        o = g[g.backend == "oracle"].set_index("qid")["score"]
        for b, gb in g[g.backend != "oracle"].groupby("backend"):
            sc = gb.set_index("qid")["score"]
            com = o.index.intersection(sc.index)
            if len(com):
                rows.append((f"{pl} · {b}", sc[com].mean(), o[com].mean() - sc[com].mean(), 1 - o[com].mean()))
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rows) + 1.6))
        parts = [("correct", PAL[0]), ("lost to segmentation", PAL[1]), ("lost to planner", PAL[5])]
        for i, (lab, sc, sl, pl_) in enumerate(rows):
            left = 0
            for (name, col), v in zip(parts, (sc, max(sl, 0), pl_)):
                ax.barh(i, v, left=left, color=col, height=0.6, edgecolor=SURF, linewidth=2,
                        label=name if i == 0 else None)
                if v > 0.06:
                    ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff")
                left += v
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("share of the maximum score")
        ax.grid(axis="y", visible=False)
        ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8)
        ax.set_title("Error attribution (oracle masks isolate planner errors)", loc="left", pad=24)
        fig.tight_layout()
        fig.savefig(fdir / "fig6_error_attribution.png", dpi=dpi)
        plt.close(fig)


def _overlay(rgb, lab, ids, col, fill=0.35):
    from skimage.segmentation import find_boundaries
    out = rgb.astype(np.float32)
    for i in ids:
        m = lab == i
        if not m.any():
            continue
        out[m] = (1 - fill) * out[m] + fill * np.array(col, np.float32)
        out[find_boundaries(m, mode="inner")] = col
    return out.clip(0, 255).astype(np.uint8)


def _contours(rgb, lab, col):
    from skimage.segmentation import find_boundaries
    out = rgb.copy()
    out[find_boundaries(lab, mode="inner")] = col
    return out


def fig_seg_examples(a, fdir, plt, dpi):
    """Per dataset: raw gold frame, gold contours, and each backend's contours on the same frame."""
    ds = find_datasets(a.data, a.datasets)
    root = Path(a.out) / "seg"
    backends = [b for b in BACKEND_ORDER if (root / b).is_dir()] + \
        sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in BACKEND_ORDER) if root.exists() else []
    for name, d in ds.items():
        s2 = Seq(d, "02")
        gold = s2.annotated("gold")
        if not gold:
            continue
        t = gold[len(gold) // 2]
        u8 = normalize_u8(s2.image(t))
        rgb = np.stack([u8] * 3, -1)
        panels = [("input", rgb), ("gold truth", _contours(rgb, s2.mask(t, "gold"), MISS_C))]
        for b in backends:
            rd = res_dir(a.out, b, name)
            if rd.exists() and t in _index_dir(rd):
                lab = SegResult(rd).label(t)
                sc, _ = seg_measure(s2.mask(t, "gold"), lab)
                panels.append((f"{b}  (SEG {sc:.2f}, n={len(np.unique(lab)) - 1})", _contours(rgb, lab, (42, 200, 120))))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6 * u8.shape[0] / u8.shape[1] + 0.5))
        for ax, (title, im) in zip(np.atleast_1d(axes), panels):
            ax.imshow(im)
            ax.set_title(title, loc="left", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{name} · sequence 02 · frame {t}", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(fdir / f"seg_examples_{name}.png", dpi=dpi)
        plt.close(fig)


def fig_query_examples(a, fdir, plt, dpi):
    """Qualitative prompt -> tool calls -> answer figures, successes and failures per query type."""
    import textwrap
    df = _bench_df(a.out)
    if df is None or "trace" not in df:
        return
    ds = find_datasets(a.data, a.datasets)
    qs = {json.loads(l)["qid"]: json.loads(l) for l in (Path(a.out) / "bench" / "queries.jsonl").read_text().split("\n") if l.strip()}
    edir = fdir / "query_examples"
    edir.mkdir(exist_ok=True)
    cache = {}
    n_done = 0
    for (pl, b), g in df.groupby(["planner", "backend"]):
        for typ, gt in g.groupby("type"):
            for outcome, gg in (("success", gt[gt.score >= 0.999]), ("failure", gt[gt.score < 0.999])):
                for _, r in gg.sort_values("style", ascending=False).head(a.n_examples).iterrows():
                    q = qs.get(r.qid)
                    if q is None or r.dataset not in ds:
                        continue
                    key = (r.dataset, b)
                    if key not in cache:
                        s2 = Seq(ds[r.dataset], "02")
                        cache[key] = (s2, SegResult(res_dir(a.out, b, r.dataset)), SegResult(res_dir(a.out, "oracle", r.dataset)))
                    s2, sys_seg, orc = cache[key]
                    gold, ans = q["gold"], r.answer if isinstance(r.answer, dict) else None
                    if gold["kind"] == "divisions":
                        t = gold["events"][0]["frame"] if gold["events"] else gold["start"]
                    else:
                        t = gold["frame"]
                    if t not in sys_seg.files:
                        continue
                    rgb = np.stack([normalize_u8(s2.image(t))] * 3, -1)
                    out = rgb.copy()
                    legend = ""
                    if gold["kind"] == "cells" and ans is not None:
                        mp = match_instances(orc.label(t), sys_seg.label(t))[0]
                        pred = ans.get("cell_ids", []) if ans.get("frame") == t else []
                        ok = [i for i in pred if mp.get(i) in set(gold["ids"])]
                        bad = [i for i in pred if i not in ok]
                        miss = [i for i in gold["ids"] if i not in {mp[i] for i in ok}]
                        out = _overlay(out, sys_seg.label(t), ok, OK_C)
                        out = _overlay(out, sys_seg.label(t), bad, BAD_C)
                        out = _overlay(out, orc.label(t), miss, MISS_C, fill=0.0)
                        legend = f"green = correct ({len(ok)})  red = wrong ({len(bad)})  yellow outline = missed ({len(miss)})"
                    elif gold["kind"] == "count":
                        out = _contours(out, sys_seg.label(t), (42, 200, 120))
                        got = (ans or {}).get("count")
                        legend = f"answer {got}  ·  truth {gold['value']}  (green = segmented cells)"
                    elif gold["kind"] == "divisions" and ans is not None:
                        sess = Session(s2, sys_seg, None, r.dataset)
                        evs = [e for e in division_events(sys_seg.tracks, sess.feats, gold["start"], gold["end"])
                               if e["parent"] in set(ans.get("division_parent_ids", [])) and e["frame"] == t]
                        for e in evs:
                            out = _overlay(out, sys_seg.label(t), e["daughters"], OK_C)
                        legend = f"green = reported daughters at frame {t}; yellow circle = true division"
                    fig = plt.figure(figsize=(11, 4.2))
                    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.95, 1])
                    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
                    for ax, im, ttl in ((a0, rgb, f"INPUT · {r.dataset} · frame {t}"), (a2, out, "OUTPUT")):
                        ax.imshow(im)
                        ax.set_title(ttl, loc="left", fontsize=8)
                        ax.axis("off")
                    box = re.search(r"\[([\d.\s,]+)\]", q["text"])
                    if box and gold["kind"] != "divisions":
                        x0, y0, x1, y1 = [float(v) for v in box.group(1).split(",")]
                        for ax in (a0, a2):
                            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#9085e9", lw=1.2))
                    if gold["kind"] == "divisions":
                        for e in gold["events"]:
                            if e["frame"] == t:
                                a2.add_patch(plt.Circle((e["x"], e["y"]), gold["tol_px"], fill=False, ec="#fab219", lw=1.5))
                    a2.text(0, -0.04, legend, transform=a2.transAxes, fontsize=7, color=INK2, va="top")
                    a1.axis("off")
                    lines = [("PROMPT", True)] + [(l, False) for l in textwrap.wrap(f'"{q["text"]}"', 52)]
                    lines += [("", False), (f"{STYLE_NAMES.get(q['style'], q['style'])} · {pl} · {b}", False), ("", False),
                              ("TOOL CALLS", True)]
                    for st in (r.trace or [])[:10]:
                        args = json.dumps(st["args"])[:70]
                        lines += [(l, False) for l in textwrap.wrap(f"{st['tool']}({args[1:-1]})", 58)[:2]]
                    lines += [("", False), ("SCORE vs ground truth", True),
                              ("  ".join(f"{k}={r[k]:.2f}" for k in ("precision", "recall", "f1")
                                         if k in r and isinstance(r[k], float) and not np.isnan(r[k])), False)]
                    y = 1.0
                    for txt, bold in lines:
                        a1.text(0, y, txt, transform=a1.transAxes, fontsize=7.5, va="top", family="monospace" if not bold else None,
                                fontweight="bold" if bold else "normal", color="#6250d6" if bold else INK)
                        y -= 0.052
                    fig.tight_layout()
                    fn = f"{b}__{pl.replace(':', '_')}__{typ}__{outcome}__{r.qid.replace(':', '_')}.png"
                    fig.savefig(edir / fn, dpi=dpi)
                    plt.close(fig)
                    n_done += 1
    log(f"{n_done} qualitative example figures -> {edir}")


def st_figures(a):
    plt = _plt()
    res = Path(a.out) / "results"
    fdir = res / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    for fn in (lambda: fig_segmentation(res, fdir, plt, a.dpi), lambda: fig_classifier(res, fdir, plt, a.dpi),
               lambda: fig_bench(a.out, fdir, plt, a.dpi), lambda: fig_seg_examples(a, fdir, plt, a.dpi),
               lambda: fig_query_examples(a, fdir, plt, a.dpi)):
        try:
            fn()
        except Exception as e:  # one broken figure must not stop the others
            import traceback
            log(f"figure failed: {type(e).__name__}: {e}")
            traceback.print_exc()
    log("figures ->", fdir, sorted(p.name for p in fdir.glob("*.png")))


# =============================================================================
# 12. Self-test on synthetic CTC-format data (CPU, no API key)
# =============================================================================
def make_synthetic(root: Path, T=24, H=240, W=300):
    from skimage.draw import ellipse, polygon
    for seq, seed in (("01", 0), ("02", 1)):
        rng = np.random.default_rng(seed)
        base = root / "Synth-C2DL-Test"
        for sub in (seq, f"{seq}_GT/SEG", f"{seq}_GT/TRA", f"{seq}_ST/SEG"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        cells, nxt = [], 1
        for _ in range(10):
            cells.append(dict(L=nxt, x=rng.uniform(25, W - 25), y=rng.uniform(25, H - 25), vx=rng.uniform(-1, 1),
                              vy=rng.uniform(-1, 1), r=rng.uniform(8, 15), irr=rng.random() < 0.3, B=0, E=T - 1, P=0))
            nxt += 1
        for c in rng.choice(cells[:10], 3, replace=False):     # three divisions
            td = int(rng.integers(5, T - 5))
            c["E"] = td - 1
            for s in (-1, 1):
                cells.append(dict(L=nxt, x=c["x"] + c["vx"] * td + s * 0.8 * c["r"], y=c["y"] + c["vy"] * td,
                                  vx=s * 0.8, vy=0, r=c["r"] * 0.75, irr=False, B=td, E=T - 1, P=c["L"]))
                nxt += 1
        for t in range(T):
            lab = np.zeros((H, W), np.uint16)
            for c in cells:
                if not c["B"] <= t <= c["E"]:
                    continue
                dt = t - c["B"]
                x, y = c["x"] + c["vx"] * dt, c["y"] + c["vy"] * dt
                if c["irr"]:
                    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                    rad = c["r"] * (0.55 + 0.45 * np.random.default_rng(c["L"]).random(12))
                    rr, cc = polygon(y + rad * np.sin(ang), x + rad * np.cos(ang), (H, W))
                else:
                    rr, cc = ellipse(y, x, c["r"], c["r"] * 0.8, shape=(H, W))
                lab[rr, cc] = c["L"]
            img = (300 + 1200 * (lab > 0) + rng.normal(0, 80, lab.shape)).clip(0, 65535).astype(np.uint16)
            tifffile.imwrite(base / seq / f"t{t:03d}.tif", img)
            tifffile.imwrite(base / f"{seq}_GT/TRA" / f"man_track{t:03d}.tif", lab)
            tifffile.imwrite(base / f"{seq}_ST/SEG" / f"man_seg{t:03d}.tif", lab)
            if t % 4 == 0:
                tifffile.imwrite(base / f"{seq}_GT/SEG" / f"man_seg{t:03d}.tif", lab)
        (base / f"{seq}_GT/TRA/man_track.txt").write_text(
            "".join(f"{c['L']} {c['B']} {c['E']} {c['P']}\n" for c in cells))
    (root / "Synth-C2DL-Test_test").mkdir(exist_ok=True)       # must be ignored


def st_selftest(a):
    import tempfile
    root = Path(a.out) if a.out else Path(tempfile.mkdtemp())
    data, out = root / "data", root / "out"
    make_synthetic(data)
    base = ["--data", str(data), "--out", str(out)]
    run = lambda *x: main(list(x) + base)
    run("prepare")
    run("segment", "--backend", "oracle")
    run("segment", "--backend", "noisy_oracle")
    run("eval_seg", "--backends", "oracle", "noisy_oracle")
    run("make_bench", "--n-per-type", "3")
    run("run_bench", "--backend", "oracle", "--planner", "offline", "--overwrite")
    run("run_bench", "--backend", "noisy_oracle", "--planner", "offline", "--overwrite")
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
        run("train_cls", "--epochs", "2", "--no-pretrained", "--batch-size", "16")
        run("eval_cls")
        run("run_bench", "--backend", "oracle", "--planner", "offline", "--use-classifier", "--overwrite")
    except ImportError:
        log("torch/timm not installed: classifier stages skipped")
    run("report")
    run("figures", "--n-examples", "1", "--dpi", "110")
    df = [json.loads(l) for l in (out / "bench" / "results_oracle_offline.jsonl").read_text().split("\n") if l]
    sc = np.mean([r.get("f1", 0) for r in df])
    log(f"SELFTEST oracle+offline mean score {sc:.3f} over {len(df)} queries")
    assert sc > 0.8, "offline planner on oracle should be near-perfect on template queries"
    log("SELFTEST PASSED ->", out)


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--data", default=os.environ.get("CTC_DATA"), help="folder with the 2D datasets")
        p.add_argument("--out", default=os.environ.get("CTC_OUT", "out"))
        p.add_argument("--datasets", nargs="*", default=None, help="subset of dataset names")
        return p

    common(sub.add_parser("prepare"))
    p = common(sub.add_parser("train_cpsam"))
    p.add_argument("--mode", choices=["per_dataset", "pooled", "lodo"], default="pooled",
                   help="per_dataset = specialists; pooled = one generalist; lodo = leave-one-dataset-out (zero-shot estimate)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("predict", help="zero-shot segmentation + tracking of a new unannotated tif folder")
    p.add_argument("--images", required=True, help="folder of 2D .tif frames, sorted by file name")
    p.add_argument("--dest", required=True, help="output folder")
    p.add_argument("--out", default=os.environ.get("CTC_OUT", "out"), help="pipeline output folder holding the models")
    p.add_argument("--model", default=None, help="model file (default: the generalist cpsam_ft_pooled)")
    p.add_argument("--batch-size", type=int, default=8)
    p = common(sub.add_parser("tune_cpsam"))
    p.add_argument("--batch-size", type=int, default=8)
    p = common(sub.add_parser("segment"))
    p.add_argument("--backend", required=True,
                   choices=["oracle", "cpsam", "cpsam_ft", "cpsam_pooled", "cpsam_lodo", "sam3_zs", "sam3_ft", "noisy_oracle"])
    p.add_argument("--seq", default="02")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sam3-ckpt", default=None)
    p.add_argument("--sam3-prompt", default=None, help="fix the prompt instead of sweeping on seq 01")
    p.add_argument("--sam3-conf", type=float, default=0.3)
    p.add_argument("--sam3-tile", type=int, default=512)
    p = common(sub.add_parser("eval_seg"))
    p.add_argument("--backends", nargs="+", default=["oracle", "cpsam_ft", "cpsam_pooled", "cpsam_lodo", "cpsam", "sam3_zs", "sam3_ft"])
    p.add_argument("--seq", default="02")
    p.add_argument("--official", action="store_true", help="also run py-ctcmetrics (TRA/DET/SEG)")
    p = common(sub.add_parser("train_cls"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--arch", default="vit_small_patch16_224")
    p.add_argument("--max-ratio", type=float, default=3.0, help="interphase:mitotic sampling ratio")
    p.add_argument("--no-pretrained", action="store_true")
    common(sub.add_parser("eval_cls"))
    p = common(sub.add_parser("make_bench"))
    p.add_argument("--n-per-type", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p = common(sub.add_parser("run_bench"))
    p.add_argument("--backend", required=True)
    p.add_argument("--planner", choices=["offline", "claude", "openai"], default="offline")
    p.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--use-classifier", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p = common(sub.add_parser("paraphrase_bench"))
    p.add_argument("--planner", choices=["claude", "openai", "none"], default="claude",
                   help="LLM that writes the free paraphrases ('none' = only add --user-file)")
    p.add_argument("--model", default=os.environ.get("PARAPHRASE_MODEL", os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--user-file", default=None, help="CSV with qid,your_text (see bench/handwritten_template.csv)")
    common(sub.add_parser("report"))
    p = common(sub.add_parser("figures"))
    p.add_argument("--n-examples", type=int, default=2, help="qualitative examples per query type and outcome")
    p.add_argument("--dpi", type=int, default=200)
    common(sub.add_parser("export_sam3_coco"))
    p = sub.add_parser("selftest")
    p.add_argument("--out", default=None)

    a = ap.parse_args(argv)
    if a.stage not in ("selftest", "predict") and not a.data:
        ap.error("--data (or $CTC_DATA) is required")
    {"prepare": st_prepare, "train_cpsam": st_train_cpsam, "segment": st_segment, "tune_cpsam": st_tune_cpsam, "predict": st_predict, "eval_seg": st_eval_seg,
     "train_cls": st_train_cls, "eval_cls": st_eval_cls, "make_bench": st_make_bench, "run_bench": st_run_bench,
     "report": st_report, "paraphrase_bench": st_paraphrase_bench, "figures": st_figures,
     "export_sam3_coco": st_export_sam3_coco, "selftest": st_selftest}[a.stage](a)


if __name__ == "__main__":
    main()