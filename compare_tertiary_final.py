"""compare_tertiary_final.py

Run Stage-4 final_analysis on TWO tertiary exports together:
  * Python : cappedpython_Tertiary_Export.npz   (already npz)
  * MATLAB : 100mL-20deg-35cpm_Tertiary_Export.mat (v7.3 — converted to npz)

The MATLAB tertiary stores the same fields but as HDF5 with reversed axis
order (frames, rows, cols); we transpose to the Python (rows, cols, frames)
layout and re-save as a matching *_Tertiary_Export.npz so final_analysis can
glob both. Output: a single results_rock.csv + comparison plots.
"""

import sys
import shutil
import warnings
from pathlib import Path

import numpy as np
import h5py

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from piv_pipeline.fn_final_analysis import final_analysis

PY_TERT = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\PipelineTest_vcap07interp\Tertiary Exports\firstsave_vcap.7interp_Tertiary_Export.npz"
MAT_TERT = r"Z:\EXPERIMENTAL TECHNIQUES\PIV\Rock - Reflective Flakes\100mL-20deg-35cpm\Matlab analysis\Primary Export\Tertiary Exports\100mL-20deg-35cpm_Tertiary_Export.mat"
COMPARE_ROOT = Path(r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\FinalAnalysis_compare")


def _decode_condition(f, key="condition"):
    try:
        arr = np.asarray(f[key][()]).ravel()
        return "".join(chr(int(c)) for c in arr if int(c) != 0)
    except Exception:
        return "MATLAB"


def convert_matlab_tertiary(mat_path: str, out_npz: Path, label: str):
    """Load a v7.3 MATLAB tertiary and save a Python-format *_Tertiary_Export.npz."""
    with h5py.File(mat_path, "r") as f:
        # (frames, rows, cols) -> (rows, cols, frames)
        def grab(name):
            a = np.asarray(f[name][()])
            return np.transpose(a, (1, 2, 0)) if a.ndim == 3 else a
        masked_shear = grab("masked_shear").astype(np.float32)
        masked_velmag = grab("masked_velmag").astype(np.float32) if "masked_velmag" in f else masked_shear * 0
        t = np.asarray(f["t"][()]).ravel().astype(np.float64)
        X = np.asarray(f["X"][()]); Y = np.asarray(f["Y"][()])
        AA = int(np.asarray(f["AA"][()]).ravel()[0]) if "AA" in f else 10

    print(f"  MATLAB masked_shear: shape {masked_shear.shape}  "
          f"p50={np.nanpercentile(masked_shear,50):.4g} "
          f"p99={np.nanpercentile(masked_shear,99):.4g}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        masked_shear=masked_shear,
        masked_velmag=masked_velmag,
        X=X.astype(np.float64), Y=Y.astype(np.float64),
        t=t, AA=np.int64(AA),
        condition=np.array(label),
    )
    print(f"  Saved: {out_npz}")


def main():
    tert_dir = COMPARE_ROOT / "Tertiary Exports"
    summary_dir = COMPARE_ROOT / "Summary"
    tert_dir.mkdir(parents=True, exist_ok=True)

    # Both are the SAME condition (100mL-20deg-35cpm) → rpm 35. Use names that
    # _parse_rpm recognises (the `_<n>-0-0_` token) so both plot at rpm=35.
    print("Converting MATLAB tertiary → npz ...")
    convert_matlab_tertiary(
        MAT_TERT, tert_dir / "MATLAB_Tertiary_Export.npz",
        label="MATLAB_35-0-0_20deg")

    print("\nCopying Python capped tertiary (relabel) ...")
    py_dst = tert_dir / "Pythoncapped_Tertiary_Export.npz"
    with np.load(PY_TERT, allow_pickle=False) as d:
        fields = {k: d[k] for k in d.files}
    fields["condition"] = np.array("Pythoncap_35-0-0_20deg")  # parses to rpm 35
    np.savez_compressed(py_dst, **fields)
    ps = fields["masked_shear"]
    print(f"  Python masked_shear: shape {ps.shape}  "
          f"p50={np.nanpercentile(ps,50):.4g} p99={np.nanpercentile(ps,99):.4g}")

    print("\nRunning final_analysis over both ...")
    csv_path = final_analysis(tert_dir, summary_dir, span=50)
    print(f"\nresults_rock.csv:\n{Path(csv_path).read_text()}")
    print(f"All comparison outputs in: {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
