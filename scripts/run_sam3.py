"""SAM3-layer ridge-contour pipeline.

Instead of edge-everything, use SAM3 to segment the scene into ridge LAYERS (text
prompt 'mountain ridge' → instance masks, depth-ordered front→back). The ridge contours
we want are exactly the **boundaries between adjacent layers** (where a nearer range
occludes a farther one) plus the sky/terrain skyline. Anything interior to a single
layer is texture and is dropped — this directly removes the "extra" inner polylines.

The corrected-DexiNed evidence (cached) is used only to *gate* layer boundaries (keep a
boundary segment where it is corroborated by an edge) and could later snap them.

Stages 01..10 saved per image + panel; boundary-F vs the red annotation.
Reads results/filter_lab/cache/*.npz (rgb/depth/dexined/red); runs SAM3 live.
"""
from __future__ import annotations

import glob, json, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, median_filter
from skimage.morphology import skeletonize

from peakle.segmenters import load_segmenter
from peakle.depth import load_learned_depth, estimate_depth
from peakle.depth_cluster import normalize_depth, filter_by_ridge_signal, keep_by_ridge_signal

_BASE = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repo root
CACHE = f"{_BASE}/local/derived/cache"
OUT = f"{_BASE}/local/output/sam3-pipeline"
DE_DIR = f"{_BASE}/local/derived/diffusionedge"
SEGMENTER = os.environ.get("SEGMENTER", "sam3")  # swap backend: sam3 | mobile_sam | sam2
EDGE = os.environ.get("EDGE", "diffusionedge")   # crisp contour model: diffusionedge | dexined
DISCARD = os.environ.get("DISCARD", "1") != "0"   # discard 'clearly texture' contours (multi-signal)
RS = dict(sky_t=0.15, edge_t=0.30, sil_t=0.55)  # keep: near skyline, or strong edge, or strong silhouette
PAL = np.array([(230,60,60),(60,200,90),(70,130,240),(240,180,40),(200,80,220),(40,210,210),(250,250,250)], float)/255
_DEPTH_STOPS = np.array([(0.85,0.12,0.12),(0.95,0.80,0.20),(0.20,0.70,0.35),(0.20,0.45,0.92)])

seg = load_segmenter(SEGMENTER)
if seg is None:
    raise SystemExit(f"segmenter {SEGMENTER!r} unavailable (for sam3: huggingface-cli login w/ facebook/sam3 access)")
print(f"segmenter: {seg.name}", flush=True)
_depther = None


def st(v, lo=1, hi=99):
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((v - a) / max(b - a, 1e-6), 0, 1)


def depth_color(d01):
    pos = np.linspace(0, 1, len(_DEPTH_STOPS))
    return np.stack([np.interp(d01, pos, _DEPTH_STOPS[:, c]) for c in range(3)], -1)


def save_rgb(a, p): Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8),"RGB").save(p)
def save_gray(a, p): Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8),"L").save(p)


# ---- tracing (shared with run.py) ----
def walk(skel, max_turn=0.70):
    P = set(zip(*np.where(skel), strict=False))
    def nb(p):
        y,x=p; return [(y+a,x+b) for a in (-1,0,1) for b in (-1,0,1) if (a,b)!=(0,0) and (y+a,x+b) in P]
    def unit(a,b):
        v=np.array([b[0]-a[0],b[1]-a[1]],float); n=np.linalg.norm(v); return v/n if n else v
    seen,out=set(),[]
    for s in [p for p in P if len(nb(p))==1]+list(P):
        if s in seen: continue
        s0=[q for q in nb(s) if q not in seen]
        if not s0: continue
        path=[s,s0[0]]; seen.update(path); prev,cur=unit(s,s0[0]),s0[0]
        while True:
            cand=[q for q in nb(cur) if q not in seen]
            if not cand: break
            sc,nxt=max((float(np.dot(prev,unit(cur,q))),q) for q in cand)
            if sc<max_turn: break
            prev=unit(cur,nxt); seen.add(nxt); path.append(nxt); cur=nxt
        out.append(path)
    return out


def plen(s): return sum(np.hypot(s[i][0]-s[i-1][0], s[i][1]-s[i-1][1]) for i in range(1,len(s)))
def straightness(s):
    a,b=s[0],s[-1]; return np.hypot(a[0]-b[0],a[1]-b[1])/max(plen(s),1.0)
def smooth(s,sig=2.5):
    ys=gaussian_filter1d(np.array([p[0] for p in s],float),sig)
    xs=gaussian_filter1d(np.array([p[1] for p in s],float),sig)
    return list(zip(ys,xs,strict=True))
def trace_mask(mask, min_len=32, min_straight=0.30):
    """Trace, keep long polylines, drop curly ones (the closed-loop instance artifacts)."""
    out=[]
    for s in walk(skeletonize(mask)):
        if plen(s) >= min_len and straightness(s) >= min_straight:
            out.append(smooth(s))
    return out


def _endtan(p, at_end, k=6):
    a=np.asarray(p,float); k=min(k,len(a)-1)
    v=(a[-1]-a[-1-k]) if at_end else (a[0]-a[k]); n=np.hypot(*v); return v/n if n else v
def merge_polylines(polys, max_gap=14.0, min_cos=0.80):
    """Join fragments whose endpoints are close AND collinear (each end points toward the
    other) into longer smooth lines — recovers junction-fragmented ridges and improves both
    continuity and precision (fewer, longer, cleaner polylines)."""
    polys=[list(map(tuple,p)) for p in polys if len(p)>=2]
    changed=True
    while changed:
        changed=False; ends=[]
        for i,p in enumerate(polys):
            ends.append((i,True,np.asarray(p[-1],float),_endtan(p,True)))
            ends.append((i,False,np.asarray(p[0],float),_endtan(p,False)))
        best=None
        for a in range(len(ends)):
            ia,ea,pa,ta=ends[a]
            for b in range(a+1,len(ends)):
                ib,eb,pb,tb=ends[b]
                if ia==ib: continue
                g=pb-pa; d=np.hypot(*g)
                if d>max_gap or d<1e-6: continue
                gh=g/d
                if np.dot(ta,gh)>=min_cos and np.dot(tb,-gh)>=min_cos:
                    sc=np.dot(ta,gh)+np.dot(tb,-gh)-d/max_gap
                    if best is None or sc>best[0]: best=(sc,ia,ea,ib,eb)
        if best:
            _,ia,ea,ib,eb=best
            A=polys[ia] if ea else polys[ia][::-1]
            B=polys[ib][::-1] if eb else polys[ib]
            for k in sorted((ia,ib),reverse=True): polys.pop(k)
            polys.append(A+B); changed=True
    return [smooth(p, sig=3.5) for p in polys]   # heavier smoothing → less ragged/curly


def boundary_f(polys, gt_skel, dg, shape):
    img=Image.fromarray(np.zeros(shape,np.uint8)); dr=ImageDraw.Draw(img)
    for s in polys:
        if len(s)>1: dr.line([(int(x),int(y)) for y,x in s], fill=1, width=1)
    pred=skeletonize(np.asarray(img)>0)
    if pred.sum()==0: return {"F@10":0.0,"prec@10":0.0,"rec@10":0.0}
    dp=distance_transform_edt(~pred)
    P=float(np.mean(dg[pred]<=10)); R=float(np.mean(dp[gt_skel]<=10))
    return {"F@10":round(2*P*R/(P+R),3) if P+R>0 else 0.0,"prec@10":round(P,3),"rec@10":round(R,3)}


def load_edge(name, shape, c):
    """Contour evidence map: DiffusionEdge (crisp, AAAI'24) if available, else corrected DexiNed."""
    if EDGE == "diffusionedge":
        for ext in (".png", ".jpg"):
            p = os.path.join(DE_DIR, name + ext)
            if os.path.exists(p):
                e = np.asarray(Image.open(p).convert("L").resize((shape[1], shape[0]), Image.LANCZOS), float) / 255
                return st(e), "DiffusionEdge"
    if "dexined" in c.files:
        return st(c["dexined"].astype(float)), "DexiNed"
    return st(c["combined"].astype(float)), "DexiNed"


def process(name):
    global _depther
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    c = np.load(os.path.join(CACHE, name+".npz"), allow_pickle=True)
    rgb = c["rgb"].astype(float); depth = c["depth"].astype(float)
    red = c["red"]; shape = rgb.shape[:2]   # masked to terrain after the SAM3 mask is built (drops t-shirt etc.)
    E, esrc = load_edge(name, shape, c)                                   # advanced crisp source (DiffusionEdge)
    Edense = st(c["dexined"].astype(float)) if "dexined" in c.files else st(c["combined"].astype(float))  # dense gate
    depthn = normalize_depth(depth)

    # 01 input
    save_rgb(rgb, f"{d}/01_input.png")
    # 02 terrain via SAM3 — prompt 'mountain' DIRECTLY (not ~sky), so sky AND foreground
    # non-terrain (people, etc.) are excluded in one step and never become spurious contours.
    sky = seg.sky_mask(rgb)                                       # still needed for the skyline
    mountain = ndimage.binary_dilation(seg.terrain_mask(rgb, threshold=0.12), iterations=4)
    b=4; mountain[:, :b]=mountain[:, -b:]=mountain[-b:,:]=False
    red = red & ndimage.binary_dilation(mountain, iterations=6)   # GT ridges on terrain only (drops red t-shirt etc.)
    s2=rgb*0.4
    s2[~mountain]=s2[~mountain]*0.45+np.array([0.30,0.40,0.6])*0.55   # non-terrain (sky / people) dimmed blue
    s2[mountain]=s2[mountain]*0.4+np.array([0.5,0.35,0.2])*0.6
    save_rgb(s2, f"{d}/02_terrain.png")
    # 03 contour evidence (advanced model: DiffusionEdge crisp edges, else DexiNed)
    save_gray(E*mountain, f"{d}/03_contour_{esrc}.png")
    # 04 skyline = lowest sky pixel per column = the sky/terrain boundary. Robust to clouds at
    # the very top (which the segmenter excludes from 'sky', so a top-down run would spike).
    # Columns with no sky are interpolated from neighbours.
    has = sky.any(0); last = (shape[0] - 1 - sky[::-1, :].argmax(0)).astype(float)
    last[~has] = np.nan; cols = np.arange(shape[1]); g = ~np.isnan(last)
    rows = np.interp(cols, cols[g], last[g]) if g.any() else np.full(shape[1], shape[0] - 1.0)
    skyline=gaussian_filter1d(median_filter(rows,25),3.0)
    im=Image.fromarray((rgb*0.6*255).astype(np.uint8),"RGB"); dr=ImageDraw.Draw(im)
    dr.line([(int(x),int(r)) for x,r in enumerate(skyline)], fill=(255,235,90), width=3); im.save(f"{d}/04_skyline.png")
    # 05 depth
    save_rgb(depth_color(st(depth)), f"{d}/05_depth.png")
    # 06 SAM3 ridge-layer instances (front→back colored) — for visualization
    layers = seg.terrain_layers(rgb, depth=depth)
    lo=rgb*0.5
    for i,m in enumerate(layers): lo[m & mountain]=lo[m & mountain]*0.45+PAL[i%len(PAL)]*0.55
    save_rgb(lo, f"{d}/06_layers.png")
    # 07 ridge candidates = SAM3 multi-scale silhouettes  ∪  DiffusionEdge crisp edges.
    #    SAM3 gives clean layer boundaries; the advanced edge model adds foreground ridge crests
    #    SAM3's coarse layers miss. (Texture the edge model adds is removed by depth, stage 08.)
    sil = seg.silhouette_map(rgb, mountain, thresholds=(0.3, 0.15, 0.08))
    save_gray(st(gaussian_filter(sil, 0.6)) if sil.max() else sil, f"{d}/07_silhouettes.png")
    # UNgated: the DexiNed dense-gate was throwing away SAM3's own silhouette recall (the
    # face counterforts) — sil already covers ~0.75 of the annotations. Keep all silhouettes
    # ∪ crisp DiffusionEdge crests; cleanliness is handled downstream by the discard rule.
    ridge = ((sil > 0) | (E > 0.40)) & mountain
    save_gray(ridge.astype(float), f"{d}/08_ridge_candidates.png")
    # 08b discard: classify each traced contour, drop only 'clearly texture'. Loose trace
    # (min_len 24, straightness 0.22) preserves the foreground face-ridges the strict trace lost.
    raw_polys = merge_polylines(trace_mask(ridge, min_len=24, min_straight=0.22))
    siln = st(sil) if sil.max() else sil
    if DISCARD:
        internal, labels = filter_by_ridge_signal(raw_polys, siln, E, skyline, **RS)
    else:
        internal, labels = raw_polys, ["kept"] * len(raw_polys)
    COL = {"silhouette": (60,235,255), "crest": (70,240,120), "region": (205,140,255),
           "long": (60,235,255), "skyline": (255,235,90), "kept": (60,235,255)}
    ov = (rgb*0.45*255).astype(np.uint8); im=Image.fromarray(ov,"RGB"); dr=ImageDraw.Draw(im)
    for s in raw_polys:  # discarded ('clearly texture' — no ridge signal) shown faint red
        if not keep_by_ridge_signal(s, siln, E, skyline, **RS)[0]:
            dr.line([(int(x),int(y)) for y,x in s], fill=(150,45,45), width=1)
    for s, lb in zip(internal, labels, strict=True):
        dr.line([(int(x),int(y)) for y,x in s], fill=COL.get(lb,(60,235,255)), width=2)
    im.save(f"{d}/08b_discard_classified.png")
    # 09 final polylines (skyline + depth-kept contours) over annotation
    base=(rgb*0.5*255).astype(np.uint8); base[red]=(255,55,55)
    im=Image.fromarray(base,"RGB"); dr=ImageDraw.Draw(im)
    dr.line([(int(x),int(r)) for x,r in enumerate(skyline)], fill=(255,235,90), width=3)
    for s in internal: dr.line([(int(x),int(y)) for y,x in s], fill=(60,235,255), width=2)
    im.save(f"{d}/09_polylines.png")
    # cache final + pre-filter polylines + evidence maps (for the optimizer and offline analysis)
    np.savez(f"{d}/_trace.npz", polys=np.array(raw_polys, dtype=object),
             internal=np.array(internal, dtype=object), labels=np.array(labels, dtype=object),
             depthn=depthn.astype(np.float32), edge=E.astype(np.float32),
             sil=(st(sil) if sil.max() else sil).astype(np.float32), mountain=mountain,
             red=red, skyline=skyline.astype(np.float32), shape=shape)
    # metrics: report depth-filtered vs unfiltered
    gt=skeletonize(red); dg=distance_transform_edt(~gt)
    m=boundary_f(internal, gt, dg, shape)
    m_raw=boundary_f(raw_polys, gt, dg, shape)
    m["raw_F@10"]=m_raw["F@10"]; m["raw_prec@10"]=m_raw["prec@10"]; m["raw_rec@10"]=m_raw["rec@10"]
    m["edge_source"]=esrc; m["n_layers"]=len(layers); m["n_internal"]=len(internal); m["n_removed"]=len(raw_polys)-len(internal)
    m["labels"]={k: labels.count(k) for k in set(labels)}
    json.dump(m, open(f"{d}/10_metrics.json","w"), indent=2)
    # panel
    tiles=["02_terrain",f"03_contour_{esrc}","08_ridge_candidates","07_silhouettes","08b_discard_classified","09_polylines"]
    ims=[Image.open(f"{d}/{t}.png").convert("RGB").resize((360,round(360*shape[0]/shape[1]))) for t in tiles]
    w,h=ims[0].size; panel=Image.new("RGB",(w*3,h*2),(15,15,15))
    for i,t in enumerate(ims): panel.paste(t,((i%3)*w,(i//3)*h))
    panel.save(f"{d}/panel.png")
    print(f"{name}: layers={len(layers)} internal={len(internal)} F@10={m['F@10']} P/R={m['prec@10']}/{m['rec@10']}", flush=True)
    return m


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    summary={}
    for p in sorted(glob.glob(f"{CACHE}/*.npz")):
        summary[os.path.basename(p)[:-4]] = process(os.path.basename(p)[:-4])
    json.dump(summary, open(f"{OUT}/summary.json","w"), indent=2)
    print("MEAN F@10:", round(float(np.mean([m["F@10"] for m in summary.values()])),3))
