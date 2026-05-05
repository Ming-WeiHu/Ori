"""
Bioreactor Particle Analyzer - GUI Launcher
Run this file to open the graphical interface.

USAGE
-----
    python gui.py

PLUG-AND-PLAY (sharing this code)
---------------------------------
To run on a different machine you need:
  1. Python 3.8+ with the packages below installed.
  2. Both `gui.py` AND `particle_testing.py` in the SAME directory.
     (gui.py launches particle_testing.py as a subprocess.)
  3. Tkinter (bundled with most Python installs; on Linux: `sudo apt install python3-tk`).

Required packages (one-liner):
    pip install opencv-python numpy pandas matplotlib pillow

That's it. Icon + logo are baked in as base64 PNGs below, so no asset files
need to ship alongside. Input/output folders are picked at runtime via the
GUI; nothing is hardcoded to the original developer's machine.

HSV THRESHOLDS
--------------
The GUI exposes optional per-volume-tier HSV tuning (Normal / Mid / High / CPM).
If you don't tune anything, the built-in defaults from particle_testing.py
are used.

To tune:
  1. Click "Tune <tier>...".
  2. Pick a sample VIDEO at that volume — best practice is to choose one whose
     last frame is at an INTERMEDIATE mixing state (partially concentrated,
     partially suspended). Fully concentrated / fully suspended frames don't
     reveal threshold sensitivity — intermediate ones do.
  3. The last frame is auto-extracted. The tuner ALSO looks for the matching
     background video in the same folder ("Background video.MP4" or, for CPM,
     "base_before_adding_particles.mp4"). If found, the live preview shows the
     COMBINED mask (HSV ∩ bg-subtraction) — same operation particle_testing.py
     performs at run time. If no bg video is found, the preview falls back to
     HSV-only.
  4. Drag the H/S/V sliders until the mask isolates particles cleanly. The
     "BG diff" slider (only shown when a bg video was found) is for live
     visualization — particle_testing.py uses its own static defaults at run
     time (15 normal/mid/high, 30 CPM).
  5. Press Enter or 'q' to confirm. Esc cancels.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import sys
import threading
import base64
import io
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np

# ───────────────────────── EMBEDDED ASSETS ─────────────────────────
# Icon (128x128 PNG) and logo (400x200 PNG) baked in as base64.
# Edit/regenerate by running:
#   python -c "import base64; print(base64.b64encode(open('icon.jpg','rb').read()).decode())"

ICON_B64 = """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"""

LOGO_B64 = """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"""


def _decode_image(b64_str):
    """Decode an embedded base64 PNG into a PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


# ───────────────────────── ON-SCREEN HELP CONTENT ─────────────────────────
# Displayed by the "Help" button. Designed to give a brand-new user enough
# context to operate the GUI without an external SOP. Plain text, ~one screen.
HELP_TEXT = """\
QUICK START
═══════════════════════════════════════════════════════════════════════════
  1. Click "Browse…" next to "Input Folder" — pick the folder with your videos
     and the matching "Background video.MP4".
  2. Click "Browse…" next to "Output Folder" — where results land. Auto-created.
  3. (Optional) Check "CPM mode" if all sample files are compression videos.
  4. (Optional) Click "Tune…" on a tier to dial HSV thresholds for problem data.
  5. Click "▶ Run Analysis". Watch the log. Done = green text says "complete".


WHAT THE INPUT FOLDER NEEDS
═══════════════════════════════════════════════════════════════════════════
  • One or more SAMPLE videos (.MP4) — your experiments
  • "Background video.MP4" — same vessel WITHOUT particles
        (or "base_before_adding_particles.mp4" for CPM data)
  Filename convention is parameter-rich, e.g.:
        50mL_15deg_25rpm_r1.MP4
        compression-750mL-25.0-...-_r1.MP4   (CPM files)


WHAT YOU GET IN THE OUTPUT FOLDER
═══════════════════════════════════════════════════════════════════════════
  • results.json — main metrics file (video_end_pct, blob counts, status, etc.)
  • Per-experiment overlay PNGs and detection visualizations
  • Optional charts comparing concentration vs. suspended states


CPM MODE CHECKBOX
═══════════════════════════════════════════════════════════════════════════
  Check this ONLY if every file in the folder is a compression-mode video
  (filename contains "cpm" or "compression"). Skip otherwise — the analyzer
  auto-detects per-file based on the filename anyway.


HSV THRESHOLDS — WHEN TO TUNE
═══════════════════════════════════════════════════════════════════════════
  Defaults work for typical-quality videos. Tune ONLY if particles aren't
  being detected well. Each volume regime has its own tier:
        Normal (50/60mL)   — small volumes
        Mid    (100mL)     — wider HSV range
        High   (200mL+)    — widest HSV (deeper water dims particles)
        CPM    (compress)  — fainter, more purple particles

  Workflow:
        1. Click "Tune <tier>…"
        2. Pick a sample VIDEO at that volume — best to choose one whose last
           frame is at an INTERMEDIATE state (partly mixed). Fully concentrated
           or fully dispersed frames hide threshold sensitivity.
        3. The last frame is auto-extracted. The tuner also looks for the
           matching background video in the same folder.
        4. Drag the H, S, V min/max sliders until the mask isolates particles.
        5. Press Enter or 'q' to confirm. Esc to cancel.
        6. Click "Reset" to revert that tier to particle_testing.py defaults.


THE TUNER WINDOW SLIDERS
═══════════════════════════════════════════════════════════════════════════
  ★ SAVED to the run (used by particle_testing.py at run time):
        H min / H max     hue range (color). 0-179 (OpenCV scale, NOT 0-360)
        S min / S max     saturation. Higher = more vivid color
        V min / V max     value (brightness)
        Morph             noise-removal kernel size. 0 = OFF (preserves single
                          pixel particles). 3 = default (kills 1-2 pixel blobs).
                          Use 0 if real particles disappear in the mask.

  ✗ TUNER-ONLY (NOT carried to the run, just for live preview/measurement):
        BG diff                  bg-subtraction threshold. Runtime uses tier
                                 defaults: 15 normal/mid/high, 30 CPM. The
                                 slider lets you SEE how bg-diff shapes the
                                 mask but does not change it.
        Region of Interest %     ("ROI radius %" on the slider) — radius of
                                 the green circle the coverage % is measured
                                 against in the tuner. The actual run uses a
                                 manually-drawn circle (you click+drag on the
                                 first frame of each experiment when prompted
                                 by particle_testing.py).
        Inner Region %           ("Inner ROI %" on the slider) — optional
                                 inner cut-out for the tuner readout (e.g. to
                                 exclude the CPM hub). Also NOT carried to
                                 the run.

  Top banner shows live coverage % and status label:
        < 10%  CONCENTRATED (clumped, small detected area)
        10-30% INTERMEDIATE
        > 30%  MOSTLY SUSPENDED (dispersed, large detected area)

  Why the split?  The tuner shows you what HSV + bg subtraction WOULD produce
  on a sample frame, so you can tune HSV by eye. But the Region of Interest
  (ROI) for the actual run has to be the real vessel boundary, which only you
  can draw on the specific frame at run time. So we ask you to draw it then.

  Glossary
  ────────
  ROI  =  Region of Interest. The area of the image we actually care about
          (the inside of the vessel where particles can be). Pixels outside
          the ROI are ignored when measuring coverage.


COMMON ISSUES
═══════════════════════════════════════════════════════════════════════════
  Particles missing in output → Tune Normal tier, set Morph slider to 0.
  Coverage shows 50%+ on default → HSV not being constrained; check that
                                   the right tier defaults are loaded.
  "particle_testing.py not found" → ensure both files are in the SAME folder.
  Hangs on "Loading…"             → first run on a big video is slow (~30s).
                                    First-time cache build on the powerlifting
                                    side project is unrelated.
  Cv2 / numpy ImportError         → run with the ORI conda env's Python:
                                    C:\\Users\\JackHu\\anaconda3\\envs\\ORI\\python.exe gui.py


KEYBOARD IN THE TUNER WINDOW
═══════════════════════════════════════════════════════════════════════════
  Enter or q     confirm thresholds and close window
  Esc            cancel without saving
  X (window)     also cancels
"""


# ───────────────────────── HSV THRESHOLD TUNER ─────────────────────────
# Opens an OpenCV trackbar window so the user can dial in HSV thresholds
# on a sample frame before running the analysis. Mirrors the slider iteration
# from moretest.py / former test versions but wired to particle_testing.py's
# blue_lower_video / blue_upper_video defaults.

def _extract_last_frame(video_path):
    """
    Read the last decoded frame of a video. Returns a BGR numpy array, or None on failure.

    Tries seeking to (frame_count - 1) first because that's instant on well-formed files.
    Falls back to reading sequentially if the seek-decode returns None — some MP4 / MOV
    containers report a frame_count that's larger than the actual number of decodable
    frames (B-frame ordering, broken indexes, etc.), and seeking past the real end gives
    you a black or empty frame. The sequential walk is slow (~real-time playback speed
    for the file) but always correct.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 1:
            # Try the fast path: seek directly to the last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
            # Seek failed — rewind and fall through to sequential read
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Fallback: read frame-by-frame, keep the most recent successful one
        last = None
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            last = frame
        return last
    finally:
        cap.release()


def _extract_first_frame(video_path):
    """Read the first decoded frame of a video (used for the bg in the tuner preview)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def _find_bg_video(sample_path, tier):
    """
    Locate the background video that particle_testing.py would use for this tier,
    by searching the same folder as the sample. Returns a path string, or None if
    no candidate exists. Mirrors the lookup at the bottom of particle_testing.py:
      * CPM tier  -> base_before_adding_particles.mp4 preferred
      * everything else -> Background video.MP4 / .mp4 / Background_video.* etc.
    """
    folder = Path(sample_path).parent
    candidates = []
    if tier == "cpm":
        candidates.extend(['base_before_adding_particles.mp4', 'base_before_adding_particles.MP4'])
    candidates.extend([
        'Background video.MP4', 'Background video.mp4',
        'Background_video.MP4', 'Background_video.mp4',
        'Background video no water.MP4', 'Background video no water.mp4',
    ])
    for name in candidates:
        p = folder / name
        if p.exists():
            return str(p)
    return None


def tune_thresholds_window(source, default_lower, default_upper, bg_frame=None, bg_thresh=15, morph_kernel=3, min_blob=0, max_blob=0):
    """
    Show an interactive HSV slider window on `source`.

    `source` may be a file path (str/Path to an image) OR a BGR numpy array (e.g. a frame
    already extracted via _extract_last_frame).

    If `bg_frame` is provided (BGR numpy array), the live preview combines the HSV mask
    with a background-subtraction mask the same way particle_testing.py does in detection:
        diff_gray = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY)
        bg_mask   = (diff_gray > bg_thresh)
        final     = hsv_mask AND bg_mask
    plus a slider so the user can also dial in `bg_thresh` interactively. Without
    bg_frame, the preview is HSV-only as before.

    `morph_kernel` is the starting size of the MORPH_OPEN ellipse kernel applied to the
    final mask. 0 = no morphology (preserves single-pixel particles). The slider goes
    0-9. This value IS plumbed through to particle_testing.py at run time (unlike the
    bg_thresh slider, which is visualization-only).

    Sliders: H/S/V min/max (always) + "BG diff" (only when bg_frame given) + "Morph" (always).
    The "BG diff" slider is for VISUALIZATION ONLY — particle_testing.py uses its own static
    per-tier defaults (15 for normal/mid/high, 30 for CPM) at run time. The "Morph" slider
    is saved and used at run time.
    Returns: (lower [H,S,V], upper [H,S,V], morph_kernel_int) on confirm, None on cancel.
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
    else:
        img = cv2.imread(str(source))
    if img is None:
        return None

    # IMPORTANT: do NOT resize img here. We compute HSV/bg-diff/masks at full resolution so
    # the tuner exactly matches what particle_testing.py will produce at run time. The display
    # gets downsized only at the very end (the combined 3-up image), so the cv2 window stays
    # screen-sized while the underlying math runs on the actual frame.
    # (Earlier versions resized here, which bilinear-blended pixels and weakened bg-diff —
    # making the BG diff slider feel less effective in the tuner than at runtime.)
    DISP_W_PER_PANEL = 700 # display width per panel in the 3-up combined view

    # Resize bg to match img if provided
    bg_resized = None
    diff_gray = None
    if bg_frame is not None:
        if bg_frame.shape[:2] != img.shape[:2]:
            bg_resized = cv2.resize(bg_frame, (img.shape[1], img.shape[0]))
        else:
            bg_resized = bg_frame.copy()
        # Pre-compute the absdiff once — only the threshold changes per frame so we don't
        # need to redo the subtraction every loop iteration.
        diff_gray = cv2.cvtColor(cv2.absdiff(img, bg_resized), cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Centered circular ROI — used to compute the coverage % and STATUS readout, AND to
    # mask the displayed mask so the LED ring noise around the vessel doesn't dominate
    # the percentage. Default radius = 35% of the smaller frame dimension; user can tune
    # via the "ROI radius" slider below (0-49% of frame).
    ih, iw = img.shape[:2]
    roi_cx, roi_cy = iw // 2, ih // 2
    _max_radius_px = min(ih, iw) // 2 - 1 # avoid going off-frame
    _default_roi_pct = 35

    win = "HSV Tuner  --  Enter/q to confirm, Esc to cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H min", win, int(default_lower[0]), 179, lambda x: None) # OpenCV hue is 0-179, not 0-359
    cv2.createTrackbar("H max", win, int(default_upper[0]), 179, lambda x: None)
    cv2.createTrackbar("S min", win, int(default_lower[1]), 255, lambda x: None)
    cv2.createTrackbar("S max", win, int(default_upper[1]), 255, lambda x: None)
    cv2.createTrackbar("V min", win, int(default_lower[2]), 255, lambda x: None)
    cv2.createTrackbar("V max", win, int(default_upper[2]), 255, lambda x: None)
    if diff_gray is not None:
        cv2.createTrackbar("BG diff", win, int(bg_thresh), 100, lambda x: None) # particle_testing.py uses 15 (normal) / 30 (CPM)
    # Morph open kernel size (0 = skip morphology, 3 = current default in particle_testing.py).
    # Single-pixel particles get wiped by a 3x3 ellipse open — drop to 0 if you see particles disappearing.
    cv2.createTrackbar("Morph", win, int(morph_kernel), 9, lambda x: None)
    # ROI radius as % of min(height, width). 35% is a reasonable default for centered vessels.
    cv2.createTrackbar("ROI radius %", win, _default_roi_pct, 49, lambda x: None)
    # Inner ROI carve-out — excludes a centered circle (the CPM hub / compression bar / dead zone)
    # from the percentage and the displayed mask. 0 = no inner exclusion (full circle ROI).
    cv2.createTrackbar("Inner ROI %", win, 0, 30, lambda x: None)

    confirmed = None
    last_bg_thresh = bg_thresh
    last_morph = morph_kernel
    while True:
        h_lo = cv2.getTrackbarPos("H min", win)
        h_hi = cv2.getTrackbarPos("H max", win)
        s_lo = cv2.getTrackbarPos("S min", win)
        s_hi = cv2.getTrackbarPos("S max", win)
        v_lo = cv2.getTrackbarPos("V min", win)
        v_hi = cv2.getTrackbarPos("V max", win)

        lower = np.array([h_lo, s_lo, v_lo])
        upper = np.array([h_hi, s_hi, v_hi])
        hsv_mask = cv2.inRange(hsv, lower, upper)

        # Combine with bg-subtraction mask if available — mirrors particle_testing.py's pipeline.
        # Special case: bg slider at 0 = "bg filter off" (skip the AND entirely). Otherwise
        # cv2.threshold's strictly-greater semantic would still drop pixels with exactly 0 diff,
        # creating a misleading ~0.2% gap from 100% even when the user means "no filtering".
        if diff_gray is not None:
            last_bg_thresh = cv2.getTrackbarPos("BG diff", win)
            if last_bg_thresh > 0:
                _, bg_mask = cv2.threshold(diff_gray, last_bg_thresh, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_and(hsv_mask, bg_mask)
            else:
                mask = hsv_mask # bg filter explicitly disabled
        else:
            mask = hsv_mask

        # Morph open — same operation particle_testing.py applies in video_mode.
        # Skip if kernel size is < 2 (a 1x1 kernel is a no-op anyway).
        last_morph = cv2.getTrackbarPos("Morph", win)
        if last_morph >= 2:
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (last_morph, last_morph))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        # Build ROI from the slider — annulus = outer circle minus inner circle.
        # Outer radius excludes the LED ring; inner radius (if set) excludes the central hub
        # / CPM compression bar where most of the radial spoke noise originates anyway.
        # Coverage % is reported relative to the annulus area.
        roi_pct = max(1, cv2.getTrackbarPos("ROI radius %", win))
        roi_radius = int(min(ih, iw) * (roi_pct / 100.0))
        roi_radius = max(1, min(roi_radius, _max_radius_px))
        inner_pct = cv2.getTrackbarPos("Inner ROI %", win)
        inner_radius = int(min(ih, iw) * (inner_pct / 100.0))
        # Make sure inner < outer so we always have positive annulus area
        inner_radius = max(0, min(inner_radius, roi_radius - 1))
        roi_mask = np.zeros((ih, iw), dtype=np.uint8)
        cv2.circle(roi_mask, (roi_cx, roi_cy), roi_radius, 255, -1)
        if inner_radius > 0:
            cv2.circle(roi_mask, (roi_cx, roi_cy), inner_radius, 0, -1) # carve out the hub
        roi_area = max(int(np.count_nonzero(roi_mask)), 1)

        # Apply ROI to the detection mask — only what's inside the circle counts
        mask = cv2.bitwise_and(mask, roi_mask)
        detected_pixels = int(np.count_nonzero(mask))
        coverage_pct = 100.0 * detected_pixels / roi_area

        # Status categorization — flipped to match physical reality:
        # Concentrated particles cluster in a small area => few activated pixels in ROI => LOW %.
        # Mostly suspended particles spread throughout the ROI => many activated pixels => HIGH %.
        # Cutoffs are rough; user can recalibrate by eye.
        if coverage_pct < 10.0:
            status = "CONCENTRATED"
            status_color = (255, 200, 100) # blue-ish
        elif coverage_pct < 30.0:
            status = "INTERMEDIATE"
            status_color = (60, 200, 60)   # green
        else:
            status = "MOSTLY SUSPENDED"
            status_color = (60, 200, 200)  # yellow

        # Build the 3-up display AT FULL RES first (for accurate visualization), then downsize
        # the entire combined image for the cv2 window. The green circles are drawn at full res.
        img_left = img.copy()
        cv2.circle(img_left, (roi_cx, roi_cy), roi_radius, (0, 255, 0), max(2, ih // 400))
        if inner_radius > 0:
            cv2.circle(img_left, (roi_cx, roi_cy), inner_radius, (0, 255, 0), max(2, ih // 400))
        overlay = cv2.bitwise_and(img, img, mask=mask)
        combined_full = np.hstack([img_left, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay])

        # Downsize the combined 3-up to fit the screen. Use INTER_NEAREST on the mask region's
        # samples is ideal but for a 3-up we use INTER_AREA which gives clean visual downsampling.
        combined_full_w = combined_full.shape[1]
        target_w = DISP_W_PER_PANEL * 3  # 700 * 3 = 2100 px wide window
        if combined_full_w > target_w:
            disp_scale = target_w / combined_full_w
            disp_h = int(combined_full.shape[0] * disp_scale)
            combined = cv2.resize(combined_full, (target_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            combined = combined_full

        # Text overlay — coverage % and status. Drawn AFTER downsize so font size is screen-correct.
        bar_text = f"Coverage: {coverage_pct:5.2f}% of ROI    Status: {status}"
        text_size, _ = cv2.getTextSize(bar_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(combined, (0, 0), (text_size[0] + 16, text_size[1] + 16), (0, 0, 0), -1)
        cv2.putText(combined, bar_text, (8, text_size[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        # Second-line caveat: Region of Interest circle and BG-diff slider are tuner-only — not
        # carried to the run. Only HSV ranges + Morph kernel are saved and applied at run time.
        caveat = "Region of Interest (ROI) and BG diff sliders = tuner-only (not saved). Saved: H/S/V + Morph."
        ct_size, _ = cv2.getTextSize(caveat, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(combined,
                      (0, text_size[1] + 16),
                      (ct_size[0] + 16, text_size[1] + 16 + ct_size[1] + 10),
                      (0, 0, 0), -1)
        cv2.putText(combined, caveat, (8, text_size[1] + 16 + ct_size[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(win, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 13: # Enter
            confirmed = (lower.tolist(), upper.tolist(), int(last_morph)) # bg_thresh discarded (viz-only); morph saved
            break
        if key == 27: # Esc
            confirmed = None
            break
        # If user closed the window via the X button, getWindowProperty returns < 1
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            confirmed = None
            break

    cv2.destroyAllWindows()
    return confirmed


class AnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bioreactor Particle Analyzer")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)
        self.root.configure(bg="white")

        self.style = ttk.Style(self.root)
        self.style.configure("White.TFrame", background="white")
        self.style.configure("White.TLabelframe", background="white")
        self.style.configure("White.TLabelframe.Label", background="white")
        self.style.configure("White.TLabel", background="white")

        self._running = False
        self._script = Path(__file__).parent / "particle_testing.py"
        self._script_mtime = self._script.stat().st_mtime if self._script.exists() else None

        # HSV threshold overrides — one set per volume tier, mirroring the mode logic in
        # particle_testing.py (see ParticleAnalyzer._get_hsv_range and the volume detection at
        # the bottom of particle_testing.py). Each tier keeps its own override; None means
        # "use the matching built-in default in particle_testing.py".
        #   normal: 50/60mL  (blue_*_video)
        #   mid:    100mL    (midvol_*_video)
        #   high:   200mL+   (highvol_*_video)
        #   cpm:    compression mixing — selected by filename containing 'cpm' or 'compression'
        self.hsv_overrides = {
            # morph_kernel: size of the MORPH_OPEN ellipse kernel applied to the final mask
            # in video_mode. 0 = skip morphology (preserves single-pixel particles).
            # Default in particle_testing.py is 3, which can wipe pixel-sized particles —
            # the GUI slider lets you see and override per tier.
            "normal": {"lower": None, "upper": None, "morph_kernel": None},
            "mid":    {"lower": None, "upper": None, "morph_kernel": None},
            "high":   {"lower": None, "upper": None, "morph_kernel": None},
            "cpm":    {"lower": None, "upper": None, "morph_kernel": None},
        }
        # Defaults to seed the slider window with — must match particle_testing.py exactly.
        self._default_hsv = {
            "normal": ([95, 35, 35], [130, 255, 255]),  # blue_lower_video / blue_upper_video
            "mid":    ([90, 28, 28], [135, 255, 255]),  # midvol_lower_video / midvol_upper_video
            "high":   ([85, 20, 20], [140, 255, 255]),  # highvol_lower_video / highvol_upper_video
            "cpm":    ([95, 40, 30], [150, 255, 255]),  # cpm_lower_video / cpm_upper_video
        }
        # Status labels populated in _build_ui — one StringVar per tier
        self.threshold_status = {}

        self._build_ui()

    #UI Layout 
    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # Use embedded base64 images so this file can be shared standalone
        self.img = ImageTk.PhotoImage(_decode_image(ICON_B64).resize((64, 64), Image.LANCZOS))
        self.root.iconphoto(False, self.img)

        self.logo_img = ImageTk.PhotoImage(_decode_image(LOGO_B64).resize((200, 100), Image.LANCZOS))
        ttk.Label(self.root, image=self.logo_img).pack(pady=(10, 0))

        # Quick-start banner — visible up front so a new user knows the workflow at a glance.
        # Full reference is in the "?  Help" button on the run row below.
        ttk.Label(
            self.root,
            text="Quick start:  1. Pick Input folder (videos + Background video.MP4)   "
                 "2. Pick Output folder   3. (Optional) Tune HSV   4. Click ▶ Run.   "
                 "Click ?  Help for details.",
            foreground="#444", style="White.TLabel", wraplength=900, justify="left",
        ).pack(anchor="w", padx=12, pady=(2, 4))

        # ── Input folder ──
        frm_in = ttk.LabelFrame(self.root, text="Input Folder  (must contain sample videos + Background video.MP4)", style="White.TLabelframe")
        frm_in.pack(fill="x", **pad)

        self.input_var = tk.StringVar()
        ttk.Entry(frm_in, textvariable=self.input_var, width=60).pack(side="left", fill="x", expand=True, padx=(5, 0), pady=5)
        ttk.Button(frm_in, text="Browse…", command=self._browse_input).pack(side="left", padx=5, pady=5)

        #Output folder
        frm_out = ttk.LabelFrame(self.root, text="Output Folder  (auto-created, holds results.json + overlay PNGs + charts)", style="White.TLabelframe")
        frm_out.pack(fill="x", **pad)

        self.output_var = tk.StringVar(value=str(Path(__file__).parent / "output"))
        ttk.Entry(frm_out, textvariable=self.output_var, width=60).pack(side="left", fill="x", expand=True, padx=(5, 0), pady=5)
        ttk.Button(frm_out, text="Browse…", command=self._browse_output).pack(side="left", padx=5, pady=5)

        #Options
        frm_opts = ttk.Frame(self.root, style="White.TFrame")
        frm_opts.pack(fill="x", padx=10, pady=2)

        self.cpm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_opts,
            text="CPM-only mode  (skip non-compression files; usually leave unchecked — analyzer auto-detects per filename)",
            variable=self.cpm_var,
        ).pack(side="left")

        # Threshold tuning — one row per volume tier, matching particle_testing.py's mode split.
        # Tune on a representative frame for each tier (last frame of a video at that volume).
        frm_thr = ttk.LabelFrame(self.root, text="HSV thresholds  (optional — tune per volume tier)", style="White.TLabelframe")
        frm_thr.pack(fill="x", padx=10, pady=4)

        # (tier_key, button_label, default_status_label) for each row
        tier_rows = [
            ("normal", "Tune Normal (50/60mL)…",      "Normal (50/60mL) default  H 95-130, S 35-255, V 35-255"),
            ("mid",    "Tune Mid Volume (100mL)…",    "Mid (100mL) default  H 90-135, S 28-255, V 28-255"),
            ("high",   "Tune High Volume (200mL+)…",  "High (200mL+) default  H 85-140, S 20-255, V 20-255"),
            ("cpm",    "Tune Compression (CPM)…",     "CPM default  H 95-150, S 40-255, V 30-255"),
        ]
        for tier, btn_label, default_label in tier_rows:
            row = ttk.Frame(frm_thr, style="White.TFrame")
            row.pack(fill="x", padx=5, pady=1)
            # `t=tier` captures the loop var by value (avoids the late-binding lambda closure trap)
            ttk.Button(row, text=btn_label, command=lambda t=tier: self._tune_thresholds(t), width=30).pack(side="left", padx=2)
            ttk.Button(row, text="Reset", command=lambda t=tier: self._reset_thresholds(t), width=8).pack(side="left", padx=2)
            self.threshold_status[tier] = tk.StringVar(value=default_label)
            ttk.Label(row, textvariable=self.threshold_status[tier], foreground="gray", style="White.TLabel").pack(side="left", padx=8)

        # Tip: tune on intermediate (mid-mixing) frames — they're the most informative for picking thresholds.
        # Fully concentrated and fully suspended frames make almost any threshold "look right" so they hide
        # how loose / strict your settings really are. Intermediate frames force you to see the boundary.
        tip_text = (
            "Tip: tune on a video at an INTERMEDIATE mixing state (partially concentrated, partially suspended). "
            "Fully-concentrated or fully-suspended frames don't reveal threshold sensitivity — intermediate ones do."
        )
        ttk.Label(frm_thr, text=tip_text, foreground="#555", style="White.TLabel", wraplength=860, justify="left").pack(anchor="w", padx=8, pady=(4, 0))

        # IMPORTANT clarification: the green circle in the tuner is a tuner-only visualization.
        # particle_testing.py uses its OWN manually-drawn circle at run time (you'll be prompted
        # to click+drag on the first frame of each experiment), NOT the tuner's circle.
        roi_note = (
            "Note:  the green Region of Interest (ROI) circle in the tuner is for VISUALIZATION ONLY. "
            "It only shapes the coverage % readout in the tuner — it is NOT carried into the analysis "
            "run. particle_testing.py uses its own manually-drawn circle (you draw it on the first "
            "frame of each experiment when prompted). Same for the BG diff slider (also viz-only). "
            "Only the H/S/V values and the Morph kernel size are saved and applied at run time."
        )
        ttk.Label(frm_thr, text=roi_note, foreground="#a04020", style="White.TLabel", wraplength=860, justify="left").pack(anchor="w", padx=8, pady=(2, 6))

        #Run button
        frm_run = ttk.Frame(self.root, style="White.TFrame")
        frm_run.pack(fill="x", padx=10, pady=8)

        self.run_btn = ttk.Button(frm_run, text="▶  Run Analysis", command=self._toggle_run, width=20)
        self.run_btn.pack(side="left")

        ttk.Button(frm_run, text="⟳ Refresh backend", command=self._refresh_script).pack(side="left", padx=(5, 0))

        # Help button — opens a popup with the full reference (HELP_TEXT). Designed so a
        # new user can run the GUI without an external SOP. Pinned right side for visibility.
        ttk.Button(frm_run, text="?  Help", command=self._show_help, width=10).pack(side="right", padx=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frm_run, textvariable=self.status_var, foreground="gray", style="White.TLabel").pack(side="left", padx=15)

        #Log output
        ttk.Label(self.root, text="Output log:", style="White.TLabel").pack(anchor="w", padx=10)
        self.log = scrolledtext.ScrolledText(self.root, state="disabled", wrap="word", font=("Courier", 9), bg="white", fg="black")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Colour tags
        self.log.tag_config("err", foreground="red")
        self.log.tag_config("ok",  foreground="green")

    #Folder Pickers 
    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_var.set(folder)

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_var.set(folder)

    # ── Run / Stop 
    def _toggle_run(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        input_dir = self.input_var.get().strip()
        if not input_dir:
            self._log("Please select an input folder first.\n", tag="err")
            return
        if not Path(input_dir).exists():
            self._log(f"Input folder not found: {input_dir}\n", tag="err")
            return

        self._running = True
        self.run_btn.config(text="■  Stop")
        self.status_var.set("Running…")
        self._log(f"Starting analysis on: {input_dir}\n", tag="ok")

        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _stop(self):
        if hasattr(self, "_proc") and self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("Stopped by user.\n", tag="err")
        self._finish()

    def _run_analysis(self):
        self._check_script_update()

        cmd = [
            sys.executable, str(self._script),
            "--input",  self.input_var.get().strip(),
            "--output", self.output_var.get().strip(),
        ]
        if self.cpm_var.get():
            cmd.append("--cpm")
        # Optional HSV + morph overrides — pass each tuned tier independently. Untuned tiers
        # fall back to particle_testing.py's built-in defaults (no flag = no override).
        # Flag suffix per tier: normal -> "", mid -> "-mid", high -> "-high", cpm -> "-cpm".
        for tier, suffix in [("normal", ""), ("mid", "-mid"), ("high", "-high"), ("cpm", "-cpm")]:
            ov = self.hsv_overrides[tier]
            if ov["lower"] is not None and ov["upper"] is not None:
                cmd.extend([
                    f"--hsv-lower{suffix}", ",".join(str(int(v)) for v in ov["lower"]),
                    f"--hsv-upper{suffix}", ",".join(str(int(v)) for v in ov["upper"]),
                ])
            if ov["morph_kernel"] is not None:
                cmd.extend([f"--morph-kernel{suffix}", str(int(ov["morph_kernel"]))])

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._proc.stdout:
                self._log(line)
            self._proc.wait()
            if self._proc.returncode == 0:
                self._log("\nAnalysis complete.\n", tag="ok")
            elif self._running:  # not stopped manually
                self._log(f"\nProcess exited with code {self._proc.returncode}.\n", tag="err")
        except Exception as exc:
            self._log(f"\nError: {exc}\n", tag="err")
        finally:
            self.root.after(0, self._finish)

    def _finish(self):
        self._running = False
        self.run_btn.config(text="▶  Run Analysis")
        self.status_var.set("Done" if not self._running else "Stopped")

    def _refresh_script(self):
        if not self._script.exists():
            self._log(f"Backend script not found: {self._script}\n", tag="err")
            return

        self._script_mtime = self._script.stat().st_mtime
        self._log(f"Backend refreshed: {self._script.name}\n", tag="ok")

    def _check_script_update(self):
        if not self._script.exists():
            self._log(f"Backend script not found: {self._script}\n", tag="err")
            return

        current_mtime = self._script.stat().st_mtime
        if self._script_mtime is None or current_mtime != self._script_mtime:
            self._script_mtime = current_mtime
            self._log(f"Detected updated backend script: {self._script.name}\n", tag="ok")
        else:
            self._log(f"Backend script is up to date.\n")

    # ── Threshold tuning ──────────────────────────────────────
    # Volume-tier labels reused across tune/reset/log
    _TIER_LABELS = {
        "normal": "Normal (50/60mL)",
        "mid":    "Mid (100mL)",
        "high":   "High (200mL+)",
        "cpm":    "Compression (CPM)",
    }
    _TIER_DEFAULT_STATUS = {
        "normal": "Normal (50/60mL) default  H 95-130, S 35-255, V 35-255",
        "mid":    "Mid (100mL) default  H 90-135, S 28-255, V 28-255",
        "high":   "High (200mL+) default  H 85-140, S 20-255, V 20-255",
        "cpm":    "CPM default  H 95-150, S 40-255, V 30-255",
    }

    # Recognized extensions for the picker — videos open as last-frame, images load directly.
    _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.mts', '.m4v', '.webm', '.wmv'}
    _IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    # Default bg_diff threshold per tier — must match particle_testing.py's static values
    # (used as the starting position of the BG diff slider; see tune_thresholds_window).
    _DEFAULT_BG_THRESH = {"normal": 15, "mid": 15, "high": 15, "cpm": 30}
    # Default morph-open kernel size per tier — must match particle_testing.py's defaults.
    # Set to 0 to skip the opening (preserves single-pixel particles).
    _DEFAULT_MORPH_KERNEL = {"normal": 3, "mid": 3, "high": 3, "cpm": 3}

    def _tune_thresholds(self, tier):
        """
        Open file picker (videos preferred), extract the last frame if the user picked a video
        (falls back to direct load if they picked an image), look in the same folder for the
        background video so we can show realistic bg-subtracted detection in the live preview,
        then run the HSV slider window and save the chosen thresholds for `tier`.
        tier in {"normal", "mid", "high", "cpm"} — must match the keys in self.hsv_overrides.

        Best practice: tune on a frame from an INTERMEDIATE mixing state. Frames that are
        fully concentrated or fully suspended make any threshold "look right" — the
        intermediate state is where loose vs strict thresholds visibly diverge, so that's
        where you actually pick the operating point. (See README / GUI hint label.)
        """
        if tier not in self.hsv_overrides:
            self._log(f"Internal error: unknown threshold tier '{tier}'.\n", tag="err")
            return

        label = self._TIER_LABELS[tier]
        # Videos listed first; the picker title nudges toward an intermediate-state video.
        file_path = filedialog.askopenfilename(
            title=f"Pick {label} VIDEO at an INTERMEDIATE mixing state  (last frame auto-extracted)",
            filetypes=[
                ("Videos", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.mts *.MTS *.m4v *.webm *.wmv"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        # Decide whether to extract a frame (video) or use as-is (image)
        ext = Path(file_path).suffix.lower()
        if ext in self._VIDEO_EXTS:
            self._log(f"Extracting last frame from {Path(file_path).name}…\n")
            self.root.update_idletasks() # force the log to repaint before the (potentially slow) extraction
            try:
                frame = _extract_last_frame(file_path)
            except Exception as exc:
                messagebox.showerror("Could not read video", f"OpenCV failed on:\n{file_path}\n\n{exc}")
                return
            if frame is None:
                messagebox.showerror("Could not read video", f"Failed to extract last frame from:\n{file_path}")
                return
            self._log(f"Last frame extracted: {frame.shape[1]}x{frame.shape[0]}\n", tag="ok")
            source = frame # numpy array — tune_thresholds_window handles it
        else:
            # Treat anything non-video as an image path; tune_thresholds_window calls cv2.imread
            source = file_path

        # Try to find a matching background video in the same folder so the live preview
        # mirrors particle_testing.py's actual detection (HSV mask AND bg-diff mask). If
        # none is found, the tuner gracefully degrades to HSV-only.
        bg_frame = None
        bg_video_path = _find_bg_video(file_path, tier)
        if bg_video_path:
            try:
                bg_frame = _extract_first_frame(bg_video_path)
            except Exception as exc:
                self._log(f"Could not read bg video '{Path(bg_video_path).name}' ({exc}). Falling back to HSV-only preview.\n", tag="err")
                bg_frame = None
        if bg_frame is not None:
            self._log(f"Using background: {Path(bg_video_path).name}  (preview shows HSV ∩ bg-diff)\n", tag="ok")
        else:
            self._log("No background video found in folder — tuning preview is HSV-only.\n")

        # Start the HSV sliders from the current override (if any) or the tier's default
        cur_lo = self.hsv_overrides[tier]["lower"]
        cur_hi = self.hsv_overrides[tier]["upper"]
        if cur_lo is None or cur_hi is None:
            cur_lo, cur_hi = self._default_hsv[tier]
        # Same for morph kernel — start from override if set, else tier default
        cur_morph = self.hsv_overrides[tier]["morph_kernel"]
        if cur_morph is None:
            cur_morph = self._DEFAULT_MORPH_KERNEL[tier]

        try:
            result = tune_thresholds_window(
                source, cur_lo, cur_hi,
                bg_frame=bg_frame,
                bg_thresh=self._DEFAULT_BG_THRESH[tier],
                morph_kernel=cur_morph,
            )
        except Exception as exc:
            messagebox.showerror("Threshold tuner failed", f"{exc}")
            return

        if result is None:
            self._log(f"Threshold tuning ({label}) cancelled — keeping previous values.\n")
            return

        lower, upper, morph = result
        self.hsv_overrides[tier]["lower"] = lower
        self.hsv_overrides[tier]["upper"] = upper
        self.hsv_overrides[tier]["morph_kernel"] = morph
        morph_note = f"morph={morph}" if morph >= 2 else "morph=off"
        self.threshold_status[tier].set(
            f"{label} tuned:  H {lower[0]}-{upper[0]}, S {lower[1]}-{upper[1]}, V {lower[2]}-{upper[2]}, {morph_note}"
        )
        self._log(f"Tuned {label}: lower={lower} upper={upper} morph_kernel={morph}\n", tag="ok")

    def _show_help(self):
        """Pop up a scrollable Help window with the full HELP_TEXT reference.
        Designed so a new user can operate the GUI without an external SOP."""
        win = tk.Toplevel(self.root)
        win.title("Bioreactor Particle Analyzer — Help")
        win.geometry("820x650")
        win.configure(bg="white")
        # Try to inherit the icon from the main window
        try:
            win.iconphoto(False, self.img)
        except Exception:
            pass

        txt = scrolledtext.ScrolledText(win, wrap="word", font=("Consolas", 10),
                                        bg="white", fg="black", padx=10, pady=10)
        txt.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        txt.insert("1.0", HELP_TEXT)
        txt.config(state="disabled") # read-only

        ttk.Button(win, text="Close", command=win.destroy, width=12).pack(pady=10)

    def _reset_thresholds(self, tier):
        if tier not in self.hsv_overrides:
            return
        self.hsv_overrides[tier]["lower"] = None
        self.hsv_overrides[tier]["upper"] = None
        self.hsv_overrides[tier]["morph_kernel"] = None
        self.threshold_status[tier].set(self._TIER_DEFAULT_STATUS[tier])
        self._log(f"{self._TIER_LABELS[tier]} thresholds reset to particle_testing.py defaults.\n")

    #Log Helper
    def _log(self, text, tag=None):
        def _append():
            self.log.config(state="normal")
            if tag:
                self.log.insert("end", text, tag)
            else:
                self.log.insert("end", text)
            self.log.see("end")
            self.log.config(state="disabled")
        self.root.after(0, _append)


#Entry Point 
if __name__ == "__main__":
    root = tk.Tk()
    app = AnalyzerGUI(root)
    root.mainloop()
