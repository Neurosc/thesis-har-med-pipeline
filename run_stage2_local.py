#!/usr/bin/env python3
"""
Stage 2a/2b unattended orchestrator — LOCAL Windows side.

Waits for the server's Stage 1 chain to finish and push, pulls, then computes
the metrics and builds the exclusion-threshold table. Start it before bed and
the results are on disk in the morning.

    waits for STAGE1_COMPLETE.txt on origin/master
      -> git pull
      -> ACW  spheres  (Julia)   PIPELINES=maximal_nocensor
      -> SampEn spheres (Python) PIPELINES=maximal_nocensor
      -> ACW  parcels  (Julia)   PIPELINES=maximal_nocensor
      -> SampEn parcels (Python) PIPELINES=maximal_nocensor
      -> Stage 2b exclusion-threshold table

STOPS AT 2b BY DESIGN. Stage 2c needs an exclusion threshold, and that is a
judgement call about motion -- not something to automate overnight, and
explicitly not something to pick by which cutoff flatters the result.

FAILURE HANDLING. If the server chain fails it pushes STAGE1_FAILED.txt; this
script detects that and exits immediately with the failure text, rather than
waiting out its timeout. Any metric stage that fails aborts the chain -- no
partial metric set is left looking complete.

DOES NOT PUSH. Everything is committed locally... in fact nothing is even
committed: outputs are left in the working tree for review in the morning.

Usage (from the repo root, in its own terminal):
    python run_stage2_local.py

Options:
    --timeout-hours N   give up waiting for the server (default 12)
    --poll-minutes  N   how often to check origin (default 5)
    --skip-wait         server outputs are already pulled; run the metrics now
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOGDIR = REPO / "_stage2_local_run"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "stage2_master.txt"

SENTINEL_OK = "_stage1_nocensor_run/STAGE1_COMPLETE.txt"
SENTINEL_FAIL = "_stage1_nocensor_run/STAGE1_FAILED.txt"

RSCRIPT = r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe"   # not on PATH; unused at 2b


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    with MASTER.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def git(*args, check=True):
    """Run a git command in the repo, return (rc, stdout)."""
    p = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{p.stderr.strip()}")
    return p.returncode, p.stdout


def remote_has(path):
    """True if `path` exists in origin/master."""
    rc, _ = git("cat-file", "-e", f"origin/master:{path}", check=False)
    return rc == 0


def remote_show(path):
    _, out = git("show", f"origin/master:{path}", check=False)
    return out


def wait_for_server(timeout_hours, poll_minutes):
    deadline = time.time() + timeout_hours * 3600
    log(f"waiting for {SENTINEL_OK} on origin/master "
        f"(timeout {timeout_hours}h, poll {poll_minutes}min)")
    while True:
        git("fetch", "-q", "origin", check=False)
        if remote_has(SENTINEL_FAIL):
            log("!!! server chain reported FAILURE:")
            for ln in remote_show(SENTINEL_FAIL).splitlines():
                log("    " + ln)
            sys.exit(1)
        if remote_has(SENTINEL_OK):
            log("server chain complete:")
            for ln in remote_show(SENTINEL_OK).splitlines():
                log("    " + ln)
            return
        if time.time() > deadline:
            log(f"!!! timed out after {timeout_hours}h with no sentinel on origin/master.")
            log("    The server job may still be running -- check "
                "_stage1_nocensor_run/stage1_master.txt there.")
            sys.exit(1)
        time.sleep(poll_minutes * 60)


def run_stage(name, cmd, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    logfile = LOGDIR / f"{name}.txt"
    log(f"START  {name}")
    with logfile.open("w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                           stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        log(f"!!! FAILED {name}  (rc={p.returncode})  see {logfile.name}")
        tail = logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        for ln in tail:
            log("    " + ln)
        sys.exit(1)
    log(f"DONE   {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-hours", type=float, default=12)
    ap.add_argument("--poll-minutes", type=float, default=5)
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()

    log("=" * 66)
    log("STAGE 2 LOCAL ORCHESTRATOR — maximal_nocensor")
    log("=" * 66)
    log(f"repo: {REPO}")

    if not args.skip_wait:
        wait_for_server(args.timeout_hours, args.poll_minutes)
        log("pulling")
        git("pull", "--ff-only", "origin", "master")
        log("pull done")
    else:
        log("--skip-wait: assuming server outputs are already present")

    ts_root = REPO / "02_timeseries_extraction" / "results"
    for atlas, expect in (("qinspheres", 468), ("qinparcels", 468)):
        d = ts_root / atlas / "maximal_nocensor"
        n = len(list(d.rglob("*_timeseries.csv"))) if d.is_dir() else 0
        log(f"input check: {atlas}/maximal_nocensor -> {n} CSV(s) (expect {expect})")
        if n == 0:
            log(f"!!! no {atlas} timeseries for maximal_nocensor. Aborting rather "
                f"than computing metrics on nothing.")
            sys.exit(1)

    env_nc = {"PIPELINES": "maximal_nocensor"}

    run_stage("01_acw_spheres",
              ["julia", "03_intrinsic_neural_metrics/scripts/01_compute_acw.jl"],
              env_nc)
    run_stage("02_sampen_spheres",
              [sys.executable, "03_intrinsic_neural_metrics/scripts/02_compute_sampen.py"],
              env_nc)
    run_stage("03_acw_parcels",
              ["julia", "03_intrinsic_neural_metrics/scripts/qinparcels/01_compute_acw_parcels.jl"],
              env_nc)
    run_stage("04_sampen_parcels",
              [sys.executable, "03_intrinsic_neural_metrics/scripts/qinparcels/02_compute_sampen_parcels.py"],
              env_nc)
    run_stage("05_exclusion_table",
              [sys.executable, "99_QC/01_motion_qc/scripts/exclusion_threshold_table.py"])

    log("=" * 66)
    log("STAGE 2a + 2b COMPLETE")
    log("=" * 66)
    log("Outputs are in the working tree, uncommitted, for review:")
    log("  03_intrinsic_neural_metrics/results/acw/maximal_nocensor/")
    log("  03_intrinsic_neural_metrics/results/sampen/maximal_nocensor/")
    log("  03_intrinsic_neural_metrics/results/acw_parcels/maximal_nocensor/")
    log("  03_intrinsic_neural_metrics/results/sampen_parcels/maximal_nocensor/")
    log("  99_QC/01_motion_qc/results/exclusion_threshold_table/")
    log("")
    log("STOPPED before Stage 2c: it needs an exclusion threshold, which is")
    log("yours to choose on motion grounds. Read the 2b table first.")
    log(f"Logs: {LOGDIR}")


if __name__ == "__main__":
    main()
