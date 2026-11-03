"""
FreeSurfer "conform" of the IXI BIDS images, then a 2 mm copy for training (--data conform).

Conform, as FreeSurfer `mri_convert --conform`: LIA orientation, 1 mm isotropic voxels, 256^3 volume
centred on the centre of the input volume, trilinear interpolation, intensities rescaled to 0-255
(uint8; lower limit = minimum, upper limit = 99.9th percentile of the non-zero voxels).
No registration, no bias field correction, no skull stripping.

Python implementation following FastSurfer's conform.py (getscale / scalecrop / map_image):
https://github.com/Deep-MI/FastSurfer/blob/stable/FastSurferCNN/data_loader/conform.py
With --use-freesurfer, `mri_convert --conform` is used instead (FreeSurfer must be installed).

The conformed image is then resampled to 2 mm (linear, float32), like scripts/t1linear_to_2mm.py.

Input : data/IXI/bids/sub-*/ses-*/anat/*_T1w.nii.gz
Output: data/IXI/caps_conform_2mm/subjects/sub-*/ses-*/conform/sub-*_ses-*_desc-conform_res-2x2x2_T1w.nii.gz
        + dataset_description.json and participants_sessions_age.tsv
Images already converted are skipped.

Run from the project root:  python scripts/conform_preprocess.py [--workers N] [--use-freesurfer]
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from scipy.ndimage import affine_transform
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "data" / "IXI" / "bids"
DST_DIR = BASE_DIR / "data" / "IXI" / "caps_conform_2mm"

CONFORMED_SIZE = 256
VOXEL_SIZE = 1.0
# columns: voxel axes i, j, k in RAS -> Left, Inferior, Anterior
LIA = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def conformed_affine(img: nib.Nifti1Image) -> np.ndarray:
    """LIA, 1 mm, 256^3; same RAS centre as the input (voxel dims / 2, MGH convention)."""
    centre = img.affine @ np.r_[np.array(img.shape[:3]) / 2.0, 1.0]
    affine = np.eye(4)
    affine[:3, :3] = LIA * VOXEL_SIZE
    affine[:3, 3] = centre[:3] - affine[:3, :3] @ np.full(3, CONFORMED_SIZE / 2.0)
    return affine


def getscale(data: np.ndarray, dst_min: float, dst_max: float, f_low: float = 0.0, f_high: float = 0.999):
    """Robust intensity range from a 1000-bin histogram (FastSurfer getscale)."""
    src_min, src_max = float(data.min()), float(data.max())
    nz = int((np.abs(data) >= 1e-15).sum())
    voxnum = data.size
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, _ = np.histogram(data, histosize)
    cs = np.concatenate(([0], np.cumsum(hist)))

    idx = np.where(cs < int(f_low * voxnum))[0]
    idx = idx[-1] + 1 if len(idx) > 0 else 0
    src_min = idx * bin_size + src_min

    idx = np.where(cs >= voxnum - int((1.0 - f_high) * nz))[0]
    if len(idx) == 0:
        raise ValueError("rescale upper bound not found")
    src_max = (idx[0] - 2) * bin_size + src_min

    scale = 1.0 if src_min == src_max else (dst_max - dst_min) / (src_max - src_min)
    return src_min, scale


def scalecrop(data: np.ndarray, dst_min: float, dst_max: float, src_min: float, scale: float) -> np.ndarray:
    return np.clip(dst_min + scale * (data - src_min), dst_min, dst_max)


def conform(img: nib.Nifti1Image) -> nib.Nifti1Image:
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    out_affine = conformed_affine(img)
    src_min, scale = getscale(data, 0, 255)
    # output voxel -> RAS -> input voxel
    vox2vox = np.linalg.inv(img.affine) @ out_affine
    mapped = affine_transform(data, vox2vox, output_shape=(CONFORMED_SIZE,) * 3, order=1)
    if img.get_data_dtype() != np.uint8:
        mapped = scalecrop(mapped, 0, 255, src_min, scale)
    out = nib.Nifti1Image(np.rint(mapped).astype(np.uint8), out_affine)
    out.header.set_data_dtype(np.uint8)
    return out


def conform_freesurfer(src: Path) -> nib.Nifti1Image:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "conformed.mgz"
        subprocess.run(["mri_convert", "--conform", str(src), str(out)], check=True, capture_output=True)
        mgz = nib.load(out)
        return nib.Nifti1Image(np.asanyarray(mgz.dataobj), mgz.affine)


def to_2mm(conformed: nib.Nifti1Image) -> nib.Nifti1Image:
    as_float = nib.Nifti1Image(np.asanyarray(conformed.dataobj).astype(np.float32), conformed.affine)
    out = resample_to_output(as_float, voxel_sizes=(2.0, 2.0, 2.0), order=1)
    img = nib.Nifti1Image(out.get_fdata(dtype=np.float32), out.affine)
    img.header.set_data_dtype(np.float32)
    return img


def output_path(src: Path) -> Path:
    sub, ses = src.parts[-4], src.parts[-3]
    return DST_DIR / "subjects" / sub / ses / "conform" / f"{sub}_{ses}_desc-conform_res-2x2x2_T1w.nii.gz"


def _job(args: tuple[Path, bool]) -> str:
    src, use_freesurfer = args
    dst = output_path(src)
    conformed = conform_freesurfer(src) if use_freesurfer else conform(nib.load(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("tmp_" + dst.name)
    nib.save(to_2mm(conformed), tmp)
    os.replace(tmp, dst)
    return src.name


def write_metadata(use_freesurfer: bool) -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_DIR / "participants_sessions_age.tsv", DST_DIR)
    description = {
        "Name": "IXI conform 2 mm",
        "BIDSVersion": "1.9.0",
        "CAPSVersion": "1.0.0",
        "DatasetType": "derivative",
        "Processing": [{
            "Name": "conform",
            "Implementation": "mri_convert --conform" if use_freesurfer else "FastSurfer conform.py (reimplemented)",
            "Steps": "LIA, 1 mm, 256^3, intensities 0-255; then linear resampling to 2 mm",
        }],
    }
    with open(DST_DIR / "dataset_description.json", "w") as f:
        json.dump(description, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--use-freesurfer", action="store_true", help="use FreeSurfer's mri_convert --conform")
    args = parser.parse_args()
    if args.use_freesurfer and shutil.which("mri_convert") is None:
        parser.error("--use-freesurfer: mri_convert not found in PATH")
    if not (SRC_DIR / "participants_sessions_age.tsv").exists():
        parser.error(f"{SRC_DIR} not found: run scripts/ixi_to_bids.py first")

    write_metadata(args.use_freesurfer)
    images = sorted(SRC_DIR.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))
    todo = [(src, args.use_freesurfer) for src in images if not output_path(src).exists()]
    print(f"{len(images)} images, {len(images) - len(todo)} already done, {len(todo)} to process")
    with Pool(processes=args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(_job, todo), total=len(todo), unit="img"):
            pass
    n = len(list(DST_DIR.glob("subjects/*/*/conform/*_T1w.nii.gz")))
    print(f"Done: {n} images in {DST_DIR}")


if __name__ == "__main__":
    main()
