"""Stream-extract a GeoPose3K benchmark subset (no need to download the full 40 GB archive).

The tarball is gzip-compressed, so random access is impossible — we stream from byte 0 and keep
only the files the benchmark needs (info.txt, cylindrical photo crop, GT depth crop) until the
byte budget runs out.  Samples therefore come from the archive head: a known bias (one region,
alphabetical order) that is still vastly better than the previous N=1 validation.  Incomplete
trailing samples are pruned.

Usage: python scripts/fetch_geopose.py [--budget-gb 4] [--dest local/data/geopose]
"""

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

URL = "http://merlin.fit.vutbr.cz/elevation/geoPose3K_final_publish.tar.gz"
NEEDED = ["info.txt", "cyl/photo_crop.jpg", "cyl/distance_crop.pfm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--dest", default="local/data/geopose")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    budget = int(args.budget_gb * 1024**3)
    cmd = (
        f"curl -s {URL} | head -c {budget} | "
        f"tar -xz -C {dest} --strip-components=1 "
        f"--wildcards '*/info.txt' '*/cyl/photo_crop.jpg' '*/cyl/distance_crop.pfm.gz'"
    )
    print(f"streaming {args.budget_gb:g} GB from {URL}\n  -> {dest}", flush=True)
    # tar exits non-zero at the truncation point; extracted files up to there are fine
    subprocess.run(cmd, shell=True, check=False)

    kept, pruned = [], 0
    for d in sorted(p for p in dest.iterdir() if p.is_dir()):
        gz = d / "cyl/distance_crop.pfm.gz"
        if gz.exists():  # the archive ships the depth gzipped
            try:
                with gzip.open(gz, "rb") as fin, open(gz.with_suffix(""), "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            except (EOFError, OSError):  # truncated at the byte-budget cut — incomplete sample
                gz.with_suffix("").unlink(missing_ok=True)
            gz.unlink()
        if all((d / n).exists() for n in NEEDED):
            kept.append(d)
        else:
            shutil.rmtree(d)
            pruned += 1
    manual = sum(1 for d in kept if (d / "info.txt").read_text().splitlines()[0].strip().upper().startswith("MANUAL"))
    print(f"kept {len(kept)} complete samples ({manual} MANUAL, {len(kept) - manual} AUTO); pruned {pruned} incomplete")
    if not kept:
        sys.exit("no samples extracted — check the URL / budget")


if __name__ == "__main__":
    main()
