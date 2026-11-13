"""
FastSurfer segmentation (--seg_only) of the IXI BIDS images, then two 2 mm inputs for training.

FastSurfer seg_only: conform (LIA, 1 mm, 256^3, as FreeSurfer), whole-brain segmentation (FastSurferVINN),
brain mask from that segmentation, N4 bias field correction. No registration: the brain stays in native space.

Two inputs per subject, both resampled to 2 mm (linear, float32) like scripts/conform_preprocess.py:
  desc-nu     orig_nu.mgz              conform + N4                      (--data fastsurfer_nu)
  desc-brain  orig_nu.mgz x mask.mgz   conform + N4 + skull stripping    (--data fastsurfer_brain)

Resumable: a subject is skipped when both of its outputs exist; each output is written to a temporary
file and renamed, so an interrupted run never leaves a partial image. FastSurfer's own folder (~20-30 MB
per subject) is deleted after each subject; only the 2 mm images, the volume stats and a log are kept.

Input : data/IXI/bids/sub-*/ses-*/anat/*_T1w.nii.gz
Output: <out-dir>/subjects/sub-*/ses-*/fastsurfer/sub-*_ses-*_desc-{nu,brain}_res-2x2x2_T1w.nii.gz
        <out-dir>/stats/sub-*_ses-*_aseg+DKT.stats       FastSurfer volumes (kept, a few kB)
        <out-dir>/fastsurfer_log[_reverse].tsv            time, status and brain mask volume per subject
        + dataset_description.json and participants_sessions_age.tsv

Two sessions at once (e.g. two Colab runtimes on the same Drive folder): run one normally and one with
--reverse; they start from opposite ends and skip subjects the other one has finished.

Run:  python scripts/fastsurfer_preprocess.py --fastsurfer-home /content/FastSurfer [--out-dir DIR] [--max-subjects N] [--reverse]
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from conform_preprocess import SRC_DIR, to_2mm

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE_DIR / "data" / "IXI" / "caps_fastsurfer_2mm"
VARIANTS = ["nu", "brain"]
LOG_COLUMNS = ["participant_id", "session_id", "status", "seconds", "mask_ml", "message"]


def output_path(out_dir: Path, sub: str, ses: str, desc: str) -> Path:
    return out_dir / "subjects" / sub / ses / "fastsurfer" / f"{sub}_{ses}_desc-{desc}_res-2x2x2_T1w.nii.gz"


def run_fastsurfer(src: Path, sid: str, work_dir: Path, args: argparse.Namespace) -> None:
    home = args.fastsurfer_home
    python = home / ".venv" / "bin" / "python"
    env = os.environ.copy()
    env["PATH"] = f"{python.parent}:{env['PATH']}"
    env["PYTHONPATH"] = f"{home}:{env.get('PYTHONPATH', '')}"
    cmd = [
        str(home / "run_fastsurfer.sh"),
        "--t1", str(src.resolve()), "--sid", sid, "--sd", str(work_dir),
        "--seg_only", "--no_cereb", "--no_hypothal", "--no_cc",  # only aseg+DKT, mask and N4 are needed
        "--vox_size", "1",  # IXI voxels are 0.94 mm: the default ("min") would not give FreeSurfer's 1 mm
        "--device", args.device, "--threads", str(args.threads), "--py", str(python),
    ]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        cmd.append("--allow_root")  # Colab runs as root
    log = work_dir / f"{sid}.log"
    with open(log, "w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=args.timeout * 60)


def save_atomic(img: nib.Nifti1Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("tmp_" + dst.name)
    nib.save(img, tmp)
    os.replace(tmp, dst)


def make_inputs(subject_dir: Path, out_dir: Path, sub: str, ses: str) -> float:
    """Writes the two 2 mm inputs; returns the brain mask volume in ml (quick QC)."""
    nu = nib.load(subject_dir / "mri" / "orig_nu.mgz")
    mask = np.asanyarray(nib.load(subject_dir / "mri" / "mask.mgz").dataobj) > 0
    zooms = nu.header.get_zooms()[:3]
    if nu.shape[:3] != (256, 256, 256) or not np.allclose(zooms, 1.0):
        raise ValueError(f"unexpected conformed image: shape {nu.shape}, voxel {zooms}")
    data = np.asanyarray(nu.dataobj).astype(np.float32)
    images = {"nu": data, "brain": data * mask}
    for desc in VARIANTS:
        save_atomic(to_2mm(nib.Nifti1Image(images[desc], nu.affine)), output_path(out_dir, sub, ses, desc))
    return float(mask.sum()) / 1000.0  # 1 mm^3 voxels


def append_log(log: Path, row: dict) -> None:
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(log, sep="\t", index=False, mode="a", header=not log.exists())


def write_metadata(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_DIR / "participants_sessions_age.tsv", out_dir)
    description = {
        "Name": "IXI FastSurfer seg_only 2 mm",
        "BIDSVersion": "1.9.0",
        "CAPSVersion": "1.0.0",
        "DatasetType": "derivative",
        "Processing": [{
            "Name": "FastSurfer --seg_only --vox_size 1 --no_cereb --no_hypothal --no_cc",
            "Steps": "conform, N4 (orig_nu.mgz), brain mask (mask.mgz); desc-nu = orig_nu, "
                     "desc-brain = orig_nu x mask; then linear resampling to 2 mm",
        }],
    }
    with open(out_dir / "dataset_description.json", "w") as f:
        json.dump(description, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastsurfer-home", type=Path, required=True, help="FastSurfer folder with its .venv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/fastsurfer_work"),
                        help="FastSurfer's temporary subject folders (deleted after each subject)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--timeout", type=int, default=30, help="minutes per subject")
    parser.add_argument("--max-subjects", type=int, default=None, help="stop after N subjects (timing test)")
    parser.add_argument("--reverse", action="store_true",
                        help="start from the last subject: a second session can run at the same time")
    args = parser.parse_args()
    if not (args.fastsurfer_home / "run_fastsurfer.sh").exists():
        parser.error(f"run_fastsurfer.sh not found in {args.fastsurfer_home}")
    if not (SRC_DIR / "participants_sessions_age.tsv").exists():
        parser.error(f"{SRC_DIR} not found: run scripts/ixi_to_bids.py first")

    write_metadata(args.out_dir)
    images = sorted(SRC_DIR.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))
    todo = [src for src in images
            if not all(output_path(args.out_dir, src.parts[-4], src.parts[-3], d).exists() for d in VARIANTS)]
    print(f"{len(images)} images, {len(images) - len(todo)} already done, {len(todo)} to process", flush=True)
    todo = (todo[::-1] if args.reverse else todo)[:args.max_subjects]
    # one log per session: two Colab sessions appending to the same Drive file could lose lines
    log_file = args.out_dir / ("fastsurfer_log_reverse.tsv" if args.reverse else "fastsurfer_log.tsv")

    def is_done(src: Path) -> bool:
        return all(output_path(args.out_dir, src.parts[-4], src.parts[-3], d).exists() for d in VARIANTS)

    times, failed = [], []
    for i, src in enumerate(todo, 1):
        sub, ses = src.parts[-4], src.parts[-3]
        sid = f"{sub}_{ses}"
        if is_done(src):  # done in the meantime by the other session
            continue
        shutil.rmtree(args.work_dir / sid, ignore_errors=True)  # leftover of an interrupted run
        args.work_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        try:
            run_fastsurfer(src, sid, args.work_dir, args)
            mask_ml = make_inputs(args.work_dir / sid, args.out_dir, sub, ses)
            stats = args.work_dir / sid / "stats" / "aseg+DKT.stats"
            if stats.exists():
                (args.out_dir / "stats").mkdir(exist_ok=True)
                shutil.copy2(stats, args.out_dir / "stats" / f"{sid}_aseg+DKT.stats")
            row = dict(status="ok", mask_ml=round(mask_ml), message="")
        except Exception as e:  # keep going; the subject is retried at the next launch
            log = args.work_dir / f"{sid}.log"
            if log.exists():
                (args.out_dir / "failed").mkdir(exist_ok=True)
                shutil.copy2(log, args.out_dir / "failed" / f"{sid}.log")
                print(f"--- end of the FastSurfer log of {sid} ---")
                print("".join(log.read_text(errors="replace").splitlines(keepends=True)[-15:]), flush=True)
            row = dict(status="failed", mask_ml=None, message=str(e)[:200])
            failed.append(sid)
        seconds = time.time() - start
        append_log(log_file, dict(participant_id=sub, session_id=ses, seconds=round(seconds), **row))
        shutil.rmtree(args.work_dir / sid, ignore_errors=True)
        (args.work_dir / f"{sid}.log").unlink(missing_ok=True)

        times.append(seconds)
        remaining = sum(not is_done(s) for s in todo[i:])
        left = remaining * np.mean(times) / 3600
        print(f"[{i}/{len(todo)}] {sid}: {row['status']} in {seconds / 60:.1f} min "
              f"(mask {row['mask_ml']} ml) | {remaining} subjects left, ~{left:.1f} h for one session", flush=True)

    n = len(list(args.out_dir.glob("subjects/*/*/fastsurfer/*_desc-brain_res-2x2x2_T1w.nii.gz")))
    print(f"Done: {n} / {len(images)} subjects in {args.out_dir}")
    if failed:
        print(f"Failed ({len(failed)}), logs in {args.out_dir / 'failed'}: {failed}")


if __name__ == "__main__":
    main()
