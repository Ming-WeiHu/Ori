# Bioreactor Particle Analyzer

Analysis pipeline for bioreactor mixing videos. Detects suspended microcarrier
particles per frame, summarises each run, and produces overlay PNGs / a
`results.json` you can plot from.

The headline metric is **`video_end_pct`** — the fraction of the vessel's region
of interest (ROI) covered by detected particles in the last frame of each video.

---

## What's in the repository

| File                          | Purpose                                                               |
| ----------------------------- | --------------------------------------------------------------------- |
| `gui.py`                      | Tkinter front-end. Folder pickers, per-volume HSV tuner, Run button.  |
| `particle_testing.py`         | The actual analyzer. Detection, ROI prompting, charts, JSON writeout. |

 **`gui.py`** and **`particle_testing.py`** must live in the same folder — `gui.py`
shells out to `particle_testing.py` by relative path.

---

## Setup

Python 3, with these libraries:

```
opencv-python
numpy
pandas
matplotlib
pillow
```

## Input folder layout

A run folder must contain:

- One or more **sample videos** (`.MP4`) — your experiments.
  Filenames carry parameters and are parsed by the analyzer, e.g.
  `50mL_15deg_25rpm_r1.MP4` or
  `compression-750mL-25.0-...-_r1.MP4` (CPM files).
- **`Background video.MP4`** — same vessel WITHOUT particles, used for
  background subtraction.
  - For CPM data, `base_before_adding_particles.mp4` is preferred.
  - Other accepted names: `Background_video.MP4`, `Background video no water.MP4`.

That's it. Drop the videos in, point the GUI at the folder.

---

## Running the GUI

1. **Browse…** next to *Input Folder* — pick the folder above.
2. **Browse…** next to *Output Folder* — auto-created if it doesn't exist.
3. *(Optional)* Tick **CPM mode** if every file in the folder is compression.
   Otherwise leave it unchecked — the analyzer auto-detects per-file from the
   filename anyway.
4. *(Optional)* Click **Tune…** on a volume tier if defaults aren't picking up
   particles well. See *Threshold tuning* below.
5. Click **▶ Run Analysis**. Watch the log pane. When it goes green and says
   *"complete"*, you're done.

There's a **?  Help** button on the run row that opens a full reference popup
(everything in `HELP_TEXT` near the top of `gui.py`).

---

## Threshold tuning

Detection is HSV-thresholded blue, intersected with a background-subtraction
mask, intersected with the ROI you draw at run time. There are four tiers,
selected automatically per file based on volume:

| Tier   | Volume            | Notes                                  |
| ------ | ----------------- | -------------------------------------- |
| Normal | 50 / 60 mL        | Default `[95,35,35]`–`[130,255,255]`.  |
| Mid    | 100 mL            | Wider range — more dilute particles.   |
| High   | 200 mL+           | Widest — deeper water dims particles.  |
| CPM    | Compression files | Fainter, more purple particles.        |

### When to tune

Defaults work for typical-quality videos. Tune only if particles aren't being
detected. Workflow per tier:

1. Click **Tune <tier>…** on the GUI.
2. Pick a **sample video** at that volume. The tuner extracts the last frame
   automatically and looks for the matching background video in the same folder.
   - Best results: pick a video whose last frame is at an **intermediate state**
     (partly mixed). Fully concentrated or fully dispersed frames hide
     threshold sensitivity.
3. Drag the H / S / V min/max sliders until the centre and right panels show
   particles cleanly with minimal junk.
4. Press **Enter** or **q** to confirm. **Esc** cancels.
5. **Reset** reverts that tier to `particle_testing.py`'s defaults.

### What the tuner sliders do

**Saved to the run** (used by `particle_testing.py` at run time):

- `H min / H max` — hue (0–179, OpenCV scale).
- `S min / S max` — saturation.
- `V min / V max` — value (brightness).
- `Morph` — `MORPH_OPEN` kernel size on the final mask. **0 = OFF**, which
  preserves single-pixel particles. Default 3 kills 1–2 pixel blobs (good for
  clean runs, bad when you're hunting individual carriers).

**Tuner-only** (live preview / measurement, NOT carried to the run):

- `BG diff` — background-subtraction threshold the tuner previews with. The
  run uses tier defaults (15 normal/mid/high, 30 CPM).
- `Region of Interest %` — radius of the green circle in the tuner used to
  compute the live coverage %. The actual run uses the ROI you draw with the
  mouse on the first frame at run time.
- `Inner Region %` — optional inner cut-out for the tuner readout (e.g. to
  exclude the CPM hub from coverage measurement).

The top banner reads:

```
Coverage: NN.NN%   Status: <label>
```

with `< 10% CONCENTRATED  /  10–30% INTERMEDIATE  /  > 30% MOSTLY SUSPENDED`.

Tuner computes at full resolution and only downsizes the combined three-up image
for display — what you see in the tuner is what the runtime mask will look like.

---

## Output

In the output folder:

- `results.json` — main metrics file. Per-experiment entries with at least:
  - `video_end_pct` — last-frame coverage %, **the headline metric**.
  - filename-parsed parameters (volume, rpm, deg, replicate, etc.).
  - status counts, blob counts.
- Per-experiment overlay PNGs and detection visualisations.
- Optional charts comparing concentrated vs suspended states across runs.

---

## Running `particle_testing.py` directly

The GUI just builds and runs a CLI command. You can call it yourself:

```
python particle_testing.py --input "<folder>" --output "<folder>"
```

Useful overrides (all optional — match the GUI tuner controls):

```
--cpm                              process CPM files only
--hsv-lower      H,S,V             override normal-tier HSV lower bound
--hsv-upper      H,S,V
--hsv-lower-mid  H,S,V             ditto for mid (100mL) tier
--hsv-upper-mid  H,S,V
--hsv-lower-high H,S,V             ditto for high (200mL+) tier
--hsv-upper-high H,S,V
--hsv-lower-cpm  H,S,V             ditto for CPM tier
--hsv-upper-cpm  H,S,V
--morph-kernel       N             MORPH_OPEN kernel for normal tier (0 = skip)
--morph-kernel-mid   N
--morph-kernel-high  N
--morph-kernel-cpm   N
```

`H,S,V` is a literal three-int comma string, e.g. `--hsv-lower 95,35,35`.

---

## Common issues

| Symptom                                         | Fix                                                                |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| Particles missing in output                     | Tune the relevant tier; set Morph slider to 0.                     |
| Coverage shows 50%+ on default settings         | Wrong tier defaults loaded. Confirm the volume parses correctly.   |
| `particle_testing.py not found`                 | Both files must be in the SAME folder.                             |
| Hangs on "Loading…"                             | First decode of a big video is ~30s. Give it a moment.             |
| `cv2` / `numpy` ImportError                     | Run with the ORI conda env's Python (see *Setup*).                 |
| Spoke / hub noise in rocking-mode rerun videos  | The background video must match the rocking phase. If the bg is a static frame the rocking vessel won't subtract cleanly. Re-record bg or accept the noise floor. |

---

## Glossary

- **ROI** — Region of Interest. The area inside the vessel where particles can
  be. Pixels outside the ROI are ignored when measuring coverage.
- **CPM** — compression mixing mode (one of three platform modes:
  static / rocking / compression).
- **`video_end_pct`** — the last-frame coverage % inside the ROI. The metric
  the project actually cares about.
- **Tier** — one of four HSV / morph profiles (normal / mid / high / cpm),
  selected automatically per file.
