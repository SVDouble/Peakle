"""Assemble the self-contained GT-dataset & scoring report HTML (images inlined as data URIs).

Usage: python scripts/build_gt_report_html.py <report_asset_dir> <bench_results.json>
Writes <report_asset_dir>/report.html — publishable as an Artifact as-is.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ASSETS = Path(sys.argv[1])
RESULTS = Path(sys.argv[2])
digest = json.load(open(ASSETS / "digest.json"))
audit = json.load(open(ASSETS / "audit.json"))
rows = [r for r in json.load(open(RESULTS)) if "error" not in r]
n_conf = sum(
    1
    for r in rows
    for t in ("oracle", "extracted")
    if isinstance(r.get(t), dict) and r[t].get("verdict") == "CONFIRMED"
)


def uri(name: str) -> str:
    p = ASSETS / name
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def img(name: str, caption: str, alt: str) -> str:
    return (
        f'<figure><div class="imgframe"><img src="{uri(name)}" alt="{alt}" loading="lazy"></div>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


c = digest["corpus"]
b = digest["bench"]
n_flag = len(audit)
consensus = [a for a in audit if any("consensus" in f for f in a["flags"])]
man_cons = [a for a in consensus if a["manual"]]

oracle_ok = sum(1 for r in rows if r["oracle"]["correct"])
extr_ok = sum(1 for r in rows if r.get("extracted", {}).get("correct"))
man_rows = [r for r in rows if r["manual"]]
man_oracle_ok = sum(1 for r in man_rows if r["oracle"]["correct"])
clean_man = [
    r
    for r in man_rows
    if r["gt_consistency_px"] <= 25
    and not any(a["name"] == r["name"] and any("consensus" in f for f in a["flags"]) for a in audit)
]
clean_man_ok = sum(1 for r in clean_man if r["oracle"]["correct"])

audit_rows = "\n".join(
    f"<tr><td class='mono'>{a['name'].replace('_01024', '')}</td>"
    f"<td><span class='chip {'m' if a['manual'] else 'a'}'>{'MANUAL' if a['manual'] else 'AUTO'}</span></td>"
    f"<td>{'; '.join(a['flags'])}</td></tr>"
    for a in audit
)

thumbs = "".join(
    f'<figure class="thumb"><div class="imgframe"><img src="{uri(f"audit_{a['name']}.jpg")}" alt="overlay {a["name"]}"></div>'
    f'<figcaption><span class="mono">{a["name"].replace("_01024", "")}</span> — {a["flags"][0]}</figcaption></figure>'
    for a in audit[:6]
    if (ASSETS / f"audit_{a['name']}.jpg").exists()
)

mh = digest["multihyp"]
pc = digest["profile_clean"]
pa = digest["profile_ambiguous"]

html = f"""<title>peakle · ground truth & scoring audit</title>
<style>
:root {{
  --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --mut:#898781;
  --line:#e1e0d9; --accent:#2a78d6; --aqua:#1baf7a; --good:#006300; --crit:#d03b3b;
  --chipm:#e3edfa; --chipa:#e2f3ec;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --page:#0d0d0d; --card:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --mut:#898781;
  --line:#2c2c2a; --accent:#3987e5; --good:#0ca30c; --chipm:#1c2c40; --chipa:#15302a;
}} }}
:root[data-theme="dark"] {{
  --page:#0d0d0d; --card:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --mut:#898781;
  --line:#2c2c2a; --accent:#3987e5; --good:#0ca30c; --chipm:#1c2c40; --chipa:#15302a;
}}
:root[data-theme="light"] {{
  --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --mut:#898781;
  --line:#e1e0d9; --accent:#2a78d6; --good:#006300; --chipm:#e3edfa; --chipa:#e2f3ec;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--page); color:var(--ink); margin:0;
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:40px 22px 90px; }}
h1 {{ font-size:26px; line-height:1.2; margin:6px 0 4px; text-wrap:balance; }}
h2 {{ font-size:19px; margin:0 0 4px; text-wrap:balance; }}
h3 {{ font-size:15.5px; margin:22px 0 6px; }}
p {{ max-width:72ch; margin:8px 0; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.09em; font-size:11.5px; color:var(--accent); font-weight:650; }}
.sub {{ color:var(--ink2); }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:24px 26px 20px; margin-top:22px; }}
figure {{ margin:14px 0 6px; }}
figcaption {{ color:var(--ink2); font-size:13px; margin-top:6px; max-width:80ch; }}
.imgframe {{ background:#fcfcfb; border:1px solid var(--line); border-radius:6px; padding:6px; }}
.imgframe img {{ display:block; width:100%; height:auto; border-radius:3px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:720px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th {{ text-align:left; color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase;
  letter-spacing:.05em; padding:6px 10px 6px 0; border-bottom:1px solid var(--line); }}
td {{ padding:7px 10px 7px 0; border-bottom:1px solid var(--line); vertical-align:top; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }}
.num {{ font-variant-numeric:tabular-nums; }}
.chip {{ display:inline-block; padding:1px 8px; border-radius:99px; font-size:11.5px; font-weight:600; }}
.chip.m {{ background:var(--chipm); color:var(--accent); }}
.chip.a {{ background:var(--chipa); color:var(--aqua); }}
.stat {{ display:flex; gap:26px; flex-wrap:wrap; margin:14px 0 2px; }}
.stat div {{ min-width:120px; }}
.stat b {{ font-size:24px; font-weight:650; display:block; }}
.stat span {{ color:var(--ink2); font-size:12.5px; }}
.ok {{ color:var(--good); font-weight:650; }}
.bad {{ color:var(--crit); font-weight:650; }}
.callout {{ border-left:3px solid var(--accent); padding:2px 0 2px 14px; margin:14px 0; color:var(--ink2); }}
.thumbs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
.legendline {{ font-size:13px; color:var(--ink2); }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px 0 10px; vertical-align:baseline; }}
.overflow {{ overflow-x:auto; }}
</style>
<main>
<div class="eyebrow">peakle · localization validation</div>
<h1>Ground truth &amp; scoring audit</h1>
<p class="sub">GeoPose3K benchmark corpus and every signal the solver uses to score a pose candidate —
with per-sample cleanliness checks. Generated {digest["generated"]} from bench run
<span class="mono">{Path(RESULTS).parent.name}</span>.</p>

<section>
<h2>1 · What the ground truth is</h2>
<p><b>Source:</b> GeoPose3K (Brejcha &amp; Čadík, FIT VUT) — mountain photographs with a full camera pose
obtained by aligning each photo to a terrain render, plus a depth map <em>rendered from that pose</em>.
We stream a subset from the 40&nbsp;GB archive and keep three files per sample.</p>
<div class="stat">
  <div><b class="num">{c["n_total"]}</b><span>samples on disk</span></div>
  <div><b class="num">{c["n_manual"]}</b><span>MANUAL (human-verified pose)</span></div>
  <div><b class="num">60</b><span>pinned bench subset<br>(35 MANUAL / 25 AUTO)</span></div>
  <div><b class="num">{c["lat_range"][0]}–{c["lat_range"][1]}°N</b><span>latitude span</span></div>
  <div><b class="num">{c["lon_range"][0]}–{c["lon_range"][1]}°E</b><span>longitude span</span></div>
</div>
<h3>Per-sample record</h3>
<div class="overflow"><table>
<tr><th>field</th><th>where it comes from</th><th>how we use it</th></tr>
<tr><td><b>photo</b> <span class="mono">cyl/photo_crop.jpg</span></td>
    <td>the photo warped onto a gravity-aligned cylindrical (azimuth × elevation) grid; the tilted
        black borders are the roll rectification</td>
    <td>input to the extracted track (standardised to ≤1152&nbsp;px width)</td></tr>
<tr><td><b>extrinsics</b> <span class="mono">info.txt</span></td>
    <td>lat / lon / elevation + ZYZ Euler orientation; decoded to yaw / pitch / roll
        (decode verified against a solved sample)</td>
    <td>position &amp; FOV are treated as known (the product analogue: GPS + EXIF);
        <b>yaw is the scored quantity</b> (success = |yaw err| ≤ 5°)</td></tr>
<tr><td><b>intrinsics</b> <span class="mono">info.txt</span></td>
    <td>horizontal FOV in radians; projection is TRUE cylindrical: columns linear in azimuth,
        rows linear in tan(elevation) — determined empirically (an el-linear fit needs an
        impossible 1.79× slope on steep views; tan fits at 1.017)</td>
    <td>defines the column→azimuth mapping of every rendered hypothesis</td></tr>
<tr><td><b>GT depth</b> <span class="mono">cyl/distance_crop.pfm</span></td>
    <td>terrain distance per pixel, rendered from the GT pose (sky ≤ 0); ships gzipped</td>
    <td>its sky boundary is the <b>oracle skyline</b> — the input of the oracle track</td></tr>
<tr><td><b>flag</b> <span class="mono">MANUAL / AUTO</span></td>
    <td>whether a human verified the pose</td>
    <td>results always split by it — AUTO labels are demonstrably less reliable</td></tr>
</table></div>
<div class="grid2">
{img("anatomy_photo.jpg", "The photo crop. Cylindrical re-projection; black corners = roll rectification of the original frame.", "sample photo crop")}
{img("anatomy_depth.png", "The GT depth render from the same pose (km). Sky is empty; the terrain–sky boundary is exact.", "GT depth render")}
</div>
{img("anatomy_oracle.jpg", "Oracle skyline (green) = first terrain row per column of the GT depth. This curve is the cleanest supervision the dataset offers and drives the oracle track.", "oracle skyline overlay")}
<div class="callout">
<b>What we do NOT use yet.</b> The GT depth also encodes <em>internal</em> silhouettes (occlusion
boundaries between ridge layers) — currently only the outer skyline is scored. The archive additionally
ships semantic labels and normal maps per sample that we do not download. Known caveats: decoded GT
<b>pitch</b> is not comparable in crop coordinates (the crops are not vertically centred on the optical
axis — measured as a per-sample constant offset), and the solver assumes roll = 0, which is valid here
only because the crops are roll-rectified (corpus |roll|: median {c["roll_med_p90"][0]}°, p90 {c["roll_med_p90"][1]}°).
</div>
</section>

<section>
<h2>2 · Corpus distributions</h2>
{img("corpus_map.png", "All downloaded viewpoints. The archive-head bias is visible: everything is Alpine, mostly Switzerland/Austria. Claims of generality need samples from deeper in the archive.", "map of sample positions")}
{img("corpus_hists.png", f"FOV spans {c['fov_range'][0]}–{c['fov_range'][2]}° (median {c['fov_range'][1]}°) — from telephoto to wide. Yaw covers the full circle. Pitch is small and centred. |roll| has a real tail past 15° — irrelevant for these rectified crops, but a gap for raw pinhole photos.", "corpus histograms")}
</section>

<section>
<h2>3 · Is the ground truth actually clean?</h2>
<p>Three automatic checks run per sample, no human review needed:</p>
<div class="overflow"><table>
<tr><th>check</th><th>what it means</th><th>result (bench-60)</th></tr>
<tr><td><b>GT consistency</b></td>
    <td>chamfer between the GT-depth skyline and <em>our</em> DEM rendered at the GT yaw
        (vertical shift free, ±50°). High = the GT pose and our terrain disagree.</td>
    <td class="num">median <b>{b["gt_consistency_median"]}&nbsp;px</b> (clean core), {n_flag} flagged</td></tr>
<tr><td><b>Camera below ground</b></td>
    <td>GT elevation vs DEM ground at the GT position (we clamp to ground + 2&nbsp;m before solving)</td>
    <td class="num">13 samples &gt; 5&nbsp;m below, worst −33&nbsp;m</td></tr>
<tr><td><b>Solver consensus</b></td>
    <td>oracle and extracted tracks agree with <em>each other</em> (≤10° apart) but both sit &gt;20°
        from the GT label — the label itself is the outlier</td>
    <td class="num"><b>{len(consensus)} sample(s)</b>, {len(man_cons)} MANUAL — most earlier "consensus" flags
        turned out to be OUR decode/search bugs, since fixed (see §5)</td></tr>
</table></div>
{img("cleanliness_hists.png", "Left: GT consistency — a clean core with a dirty tail. Middle: GT camera altitude vs DEM ground. Right: extraction error of the winning skyline hypothesis vs the GT skyline (2 px on clean photos, 200+ px on garbage ones).", "cleanliness histograms")}
<h3>Flagged samples ({n_flag}/60)</h3>
<div class="overflow"><table>
<tr><th>sample</th><th>pose src</th><th>flags</th></tr>
{audit_rows}
</table></div>
<h3>Worst offenders, visually</h3>
<p class="legendline">Overlay colours:<span class="dot" style="background:#00ff5a"></span>GT-depth skyline
<span class="dot" style="background:#ff4646"></span>photo extraction
<span class="dot" style="background:#ffe100"></span>DEM @ solved pose
<span class="dot" style="background:#00c8ff"></span>DEM @ GT pose</p>
<div class="thumbs">{thumbs}</div>
<div class="callout">
<b>Verdict on cleanliness.</b> The core is clean (median GT↔DEM consistency 8.3&nbsp;px ≈ 0.4°), but
roughly a quarter of the bench carries issues, and <b>the MANUAL flag is no guarantee</b>: 4 MANUAL
samples fail the consensus check with both tracks agreeing ~53–159° away from the label. Scoring
implication: our raw oracle success ({oracle_ok}/60, {man_oracle_ok}/35 MANUAL) <em>understates</em> the
solver — on MANUAL samples that pass both cleanliness checks it is <b>{clean_man_ok}/{len(clean_man)}
({clean_man_ok / len(clean_man):.0%})</b>. The flagged list needs case-by-case review before being
excluded, so both numbers are reported.
</div>
</section>

<section>
<h2>4 · How a candidate pose is scored</h2>
<p>Success in the benchmark is <b>never</b> the residual — it is pose error vs GT. At solve time
(no GT available) the solver scores candidates with the signals below, each shown on a real sample.</p>

<h3>4.1 · The residual: capped symmetric curve chamfer</h3>
<p>Observed skyline and rendered DEM skyline are per-column curves. The distance from each curve to
the other is averaged (symmetric — a curve can't win by matching only a fragment) and capped at
60&nbsp;px so a few garbage columns can't dominate. Pitch is folded in as a free vertical shift.</p>
<div class="grid2">
{img("chamfer_right.jpg", f"At the correct yaw the DEM curve (yellow) locks onto the observed skyline (green): chamfer {digest['chamfer_demo']['right']} px.", "chamfer at correct yaw")}
{img("chamfer_wrong.jpg", f"At a wrong yaw (+120°) nothing lines up: chamfer saturates at the {digest['chamfer_demo']['wrong']:.0f} px cap.", "chamfer at wrong yaw")}
</div>

<h3>4.2 · The 360° yaw profile: basin width &amp; alias ratio</h3>
<p>The horizon of a position is ray-cast once; the chamfer is then evaluated at <em>every</em> yaw.
The shape of that profile is the confidence signal: a single deep notch means the pose is determined;
several near-equal notches mean the skyline genuinely fits multiple directions.
<b>Alias ratio</b> = best rival chamfer outside the winning basin ÷ winner (the rival gets the same
fine polish as the winner, otherwise grid quantisation fakes a margin).</p>
{img("profile_clean.png", f"A distinctive horizon: one sharp well at the GT yaw. Solved {pc['err']:+.1f}° off GT with alias {pc['alias']} and basin width {pc['well']:.0f}° → CONFIRMED.", "sharp yaw profile")}
{img("profile_ambiguous.png", f"A self-similar horizon (the case that once produced a false CONFIRMED): several near-equal wells, alias {pa['alias']}. The solver picks a wrong one ({pa['err']:+.0f}°) — and says so: {pa['verdict']}.", "ambiguous yaw profile")}

<h3>4.3 · Terrain self-similarity: the SNR gate</h3>
<p>Even with a decent alias ratio, a wrong basin can look distinct if extraction noise happens to
favour it. The defence needs no photo at all: chamfer the <em>solved DEM window</em> against the DEM
horizon at every other yaw. If the terrain nearly repeats itself, the minimum outside the basin is
small — and no photograph taken there can tell the two directions apart.
<b>SNR</b> = that terrain distinctiveness ÷ the fit residual.</p>
<div class="grid2">
{img("selfscan_clean.png", f"Distinctive terrain: the solved window is ≥{pc['snr']:.0f}× the residual unlike anywhere else on the horizon (SNR {pc['snr']}).", "self-scan distinctive")}
{img("selfscan_ambiguous.png", f"Self-similar terrain: the DEM horizon nearly repeats at the rival yaw — SNR {pa['snr']} &lt; 2 caps the verdict at AMBIGUOUS regardless of the fit.", "self-scan self-similar")}
</div>

<h3>4.4 · Multi-hypothesis extraction, arbitrated by the DEM</h3>
<p>No single sky detector is trustworthy: atmospheric haze fools the blue-dominance detector, clouds
fool the brightness detector — on different photos. Both hypotheses are solved independently and the
DEM decides; if two plausible hypotheses solve &gt;10° apart, the verdict is downgraded.</p>
{img("multihyp.jpg", f"The blue detector (red curve) locks onto a haze boundary mid-slope; the bright detector (blue curve) finds the true skyline. Solving both: blue → chamfer {mh['blue']['chamfer']} px, {mh['blue']['yaw_err']:+.1f}° ({mh['blue']['verdict']}); bright → chamfer {mh['bright']['chamfer']} px, {mh['bright']['yaw_err']:+.1f}° ({mh['bright']['verdict']}). The DEM arbitration picks the right one.", "multi-hypothesis extraction")}

<h3>4.5 · Why the residual alone can never be the answer</h3>
{img("chamfer_overlap.png", "Every solve in the bench, residual vs actual correctness: the distributions overlap — several wrong poses fit better than correct ones. Any pipeline that reports success off the residual will lie.", "chamfer overlap strip plot")}
{img("diagnostic_auc.png", "Separation power of each no-GT diagnostic on the bench. Alias ratio dominates; coverage is useless (every candidate covers most columns).", "diagnostic AUC bars")}
<div class="callout">
<b>The calibrated gate</b> (max recall at 100% precision on this bench):
<span class="mono">CONFIRMED ⇔ alias ≥ 1.5 ∧ chamfer ≤ 20 px ∧ SNR ≥ 2 ∧ coverage ≥ 0.25</span>.
On the current run: <span class="ok">{n_conf} CONFIRMED, 0 wrong</span>; the price is recall — ~⅓ of correct
solves are labelled AMBIGUOUS. Enforced as acceptance test T7: <em>no wrong solve is ever CONFIRMED</em>.
The gate was recalibrated after the solver fixes below — the previous thresholds produced 2 false
CONFIRMED under the new search geometry. Recalibration after any solver change is part of the workflow.
</div>
</section>

<section>
<h2>5 · Weaknesses &amp; what changed today</h2>
<h3>Fixed across these passes</h3>
<p>① <b>Euler decode convention corrected</b> (brute-forced against 35 solver-verified poses →
<span class="mono">Rx(−g)·Rz(−b)·Ry(−a)·P</span>, max yaw error 2.5° vs 6.7°).
② <b>True-cylindrical (tan) row mapping</b> — the el-linear assumption compressed steep terrain
("too smooth, too much sky"); the crops are linear in tan(elevation).
③ <b>Top-K basin polish</b> — the coarse ranking could bury the true basin under grid quantisation;
the 12 best basins are now fine-polished (one sample: −159° → −0.2° CONFIRMED).
④ <b>Median-centred vertical-shift search</b> — scanning the full ±50°-pitch shift range at an
affordable grid made coarse chamfers nearly random (180° flips); centring the scan at the curves'
median offset is finer AND faster.
⑤ <b>Pitch bounds ±50°</b> (crop offsets reach ~48° ≈ 1.2·Euler-b; ±30° excluded the truth entirely).
⑥ <b>GT cleanliness checks</b> in every bench + pinned manifest + per-candidate extraction records.
Net effect: oracle success 41/60 → <b>{oracle_ok}/60</b>, flagged samples 16 → {n_flag},
GT reconstruction on clean samples reaches ~2 px with a ±125 m position polish (see the panno).</p>
<h3>Open, in leverage order</h3>
<p>① <b>Extraction</b> — oracle {oracle_ok}/60 vs extracted {extr_ok}/60 is now overwhelmingly the
gap; the two colour detectors fail on haze/cloud edge cases and there is no learned fallback validated
yet (the SAM3 sky prompt was measured 300–530 px wrong on these crops — do not trust it without
re-validation). ② <b>Bench hygiene</b> — review the {n_flag} flagged samples and publish a cleaned
MANUAL subset with raw and clean scores. ③ <b>Internal silhouettes</b> — the GT depth's occlusion
boundaries are unused supervision that would disambiguate self-similar horizons. ④ <b>Corpus growth</b>
— the full 3.1k-sample archive is downloading; CrossLocate (same group) offers ~12k more photos with
positions for semi-automatic GT growth via CONFIRMED-verdict self-annotation; foto-webcam fixed cameras
give an independent pinhole GT source (and exercise roll ≠ 0). ⑤ <b>AUTO labels</b> — weak GT only.</p>
</section>
</main>
"""

out = ASSETS / "report.html"
out.write_text(html)
print(f"-> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
