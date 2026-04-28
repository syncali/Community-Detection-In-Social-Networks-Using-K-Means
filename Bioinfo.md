# ADNI Neuroimaging Preprocessing Pipeline — Full Explanation

---

## Cell 1 — Project Overview (Markdown)

**What it is:** A documentation cell introducing the project.

**What it says:**

- The dataset is **ADNI** (Alzheimer's Disease Neuroimaging Initiative) — a major, publicly-funded multi-site study that collects brain scans, cognitive tests, and biomarkers from ~2,175 participants to study Alzheimer's progression.
- The raw data is **33 GB** of **DICOM files** (`.dcm`) — the hospital/scanner standard format for medical images.
- The folder structure is deeply nested: `Subject ID → Modality → Date → Image ID → individual DICOM slices`.
- Since each scan is stored as many 2D slices (one `.dcm` per slice), the first job is reassembling them into a **single 3D brain volume**.
- The final output format is **NIfTI** (`.nii.gz`) — the neuroscience-standard compressed 3D file format used by essentially all analysis tools.
- Only **200 of 2,175 subjects** are processed due to Google Colab's time/memory limits.

**Key terminology:**
- **DICOM** — Digital Imaging and Communications in Medicine. Every MRI scanner outputs this format. One `.dcm` file = one 2D image slice.
- **NIfTI** — Neuroimaging Informatics Technology Initiative. A 3D array file with a header describing voxel size, orientation, etc. This is what analysis software (FSL, FreeSurfer, SPM, etc.) expects.
- **MPRAGE / Sag_IR** — Magnetization Prepared Rapid Gradient Echo. A specific MRI pulse sequence optimized for high-contrast 3D structural brain images. This is the gold standard T1-weighted scan for brain anatomy studies.
- **ANTsPy / antspynet** — Python wrappers around the Advanced Normalization Tools (ANTs) library, used for medical image registration and preprocessing.
- **FSL-BET** — Brain Extraction Tool from the FMRIB Software Library. The classical skull-stripping tool, but requires a full Linux installation — not feasible on Colab. `antspynet` replaces it here.
- **MNI152** — A standard brain template space (Montreal Neurological Institute, average of 152 subjects). All brains get warped to this common coordinate space so you can compare voxels across subjects.

---

## Cell 2 — Phase 1: Environment Setup (Markdown)

**What it says:** Explains that Google Drive must be mounted first so Colab can read the raw ADNI data and write outputs persistently — if Drive is not mounted, everything is lost when the Colab session ends.

---

## Cell 3 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**What it does:** Mounts your Google Drive at the path `/content/drive` inside the Colab virtual machine.

**Why:** All 33 GB of ADNI data lives on Drive, and all preprocessed outputs need to be saved there permanently. Colab's local disk (`/content/`) is wiped when the runtime disconnects. Drive is the only persistent storage available.

**How it helps:** After this cell, any path starting with `/content/drive/MyDrive/` points to your actual Drive files.

---

## Cell 4 — Install Dependencies (Markdown + Code)

```python
!apt-get install -y dcm2niix -qq
!pip install antspyx antspynet nilearn pydicom tqdm -q
```

**What it does:** Installs all required software packages.

**Each package:**

| Package | Purpose |
|---|---|
| `dcm2niix` | System-level tool that compiles 2D DICOM slices → one 3D NIfTI volume. Handles complex scanner-specific DICOM quirks. |
| `antspyx` | Python binding for ANTs. Provides N4 bias field correction, image registration, masking operations. |
| `antspynet` | Deep-learning models built on top of ANTs. Provides the `brain_extraction()` function (a U-Net that removes skull/non-brain tissue). |
| `nilearn` | Neuroimaging plotting library. Used to render orthographic brain slice views. |
| `pydicom` | Python library to read `.dcm` files and extract metadata (patient ID, slice dimensions, pixel arrays, etc.). |
| `tqdm` | Progress bar library. Shows live progress through the batch loop. |

**Why the `!` prefix:** The `!` executes shell commands (not Python). `apt-get` and `dcm2niix` are system-level, not Python packages.

**Why `-qq` and `-q`:** Silent/quiet install flags — suppress verbose output so the notebook stays readable.

---

## Cell 5 — Imports & Configuration (Markdown + Code)

This is the most important setup cell. Let's go through it in sections:

### Imports

```python
import os, gc, glob, subprocess, csv, hashlib
from datetime import datetime
import numpy as np
import ants
import pydicom
import matplotlib.pyplot as plt
from nilearn import plotting
from tqdm.notebook import tqdm
```

- `os` — file system operations (paths, directory creation, listing files)
- `gc` — Python's garbage collector. Called after each subject to free RAM.
- `glob` — pattern-based file search (e.g., find all `*.nii.gz` files)
- `subprocess` — run external shell programs (like `dcm2niix`) from Python
- `csv` — write/read the processing log
- `hashlib` — used for deterministic worker sharding (MD5 hash)
- `datetime` — timestamp each log entry
- `numpy` — array math for Z-score normalization
- `ants` — the main ANTs image processing library
- `pydicom` — DICOM file reading
- `matplotlib` — plotting
- `nilearn.plotting` — brain-specific visualization
- `tqdm.notebook` — Jupyter-friendly progress bar

### ANTsPyNet conditional import

```python
try:
    from antspynet.utilities import brain_extraction
    USE_ANTSPYNET = True
except ImportError:
    USE_ANTSPYNET = False
```

**Why:** `antspynet` downloads model weights on first use and may fail to install in some environments. This graceful fallback means the pipeline still runs using `ants.get_mask()` (a simpler threshold-based skull stripping) if the deep-learning approach is unavailable.

### Path constants

```python
ADNI_ROOT = '/content/drive/MyDrive/BioInfo/ADNI'
OUTPUT_BASE = '/content/drive/MyDrive/BioInfo/Processed_NIfTI'
CONVERTED_DIR  = ...'/converted'    # Raw NIfTI after DICOM conversion
PREPROCESSED_DIR = ...'/preprocessed' # Final preprocessed volumes
QC_DIR = ...'/qc'                   # Quality control images
```

**Why constants instead of hardcoded strings:** Centralizing paths means you only change one place if you move the data. It also makes the code self-documenting.

### Modality keywords

```python
MODALITY_KEYWORDS = ['MPRAGE', 'MP-RAGE', 'Sag_IR', 'SAG_IR', 'IR-FSPGR', 'IR_FSPGR']
```

**Why:** Different ADNI sites named the MPRAGE sequence differently. Case-insensitive matching on these synonyms ensures the folder-search function finds the right scan across all sites.

### Critical constants

```python
MIN_SLICES_FOR_3D = 20       # Any volume with a dimension ≤ 20 is rejected as a 2D scout
TRANSFORM_TYPE = 'antsRegistrationSyNQuick[s]'  # Registration algorithm
MAX_SUBJECTS = 200           # Processing cap per session
RETRY_PREVIOUS_SKIPS = False # Don't re-attempt subjects already flagged as unfindable
```

**Why `MIN_SLICES_FOR_3D = 20`:** ADNI also contains 2D localizer/scout images (small ~192×192×1 volumes). If these accidentally entered the 3D registration step, it would crash. This threshold gates them out.

**Why `SyNQuick` instead of `SyN`:** Full `SyN` (Symmetric Normalization, a non-linear diffeomorphic registration) takes ~3 minutes per subject. `SyNQuick` is a fast approximation that takes ~30–45 seconds with comparable accuracy — critical for batching 200 subjects within Colab's session time limit.

### Team sharding

```python
TEAM_WORKERS = 3
WORKER_INDEX = 0
```

**Why:** Multiple team members were running the pipeline simultaneously across different Colab accounts to split the 2,175-subject workload. Each worker gets a deterministic non-overlapping subset so no subject is processed twice and no two workers write to the same log file.

### Directory creation

```python
for directory in [CONVERTED_DIR, PREPROCESSED_DIR, QC_DIR]:
    os.makedirs(directory, exist_ok=True)
```

**Why `exist_ok=True`:** If these directories already exist (e.g., from a previous Colab session), this does not raise an error — it silently continues.

---

## Cell 6 — Phase 2: Dataset Exploration (Markdown)

Explains the purpose of the next two cells: inspect the folder hierarchy, count subjects, identify modalities, and visually verify the DICOM data before building the full pipeline.

---

## Cell 7 — Discover Subjects & Modalities

```python
all_subject_ids = sorted([
    d for d in os.listdir(ADNI_ROOT)
    if SUBJECT_ID_MARKER in d and os.path.isdir(os.path.join(ADNI_ROOT, d))
])
```

**What it does:** Lists every folder under `ADNI_ROOT` that contains `_S_` in its name (the ADNI subject ID pattern, e.g., `135_S_4356`) and confirms it is a directory.

**Why filter by `_S_`:** ADNI subject IDs all follow the format `XXX_S_XXXX`. This ensures we only collect real subject folders and skip any metadata/readme files or unrelated subdirectories.

**Then:** It picks the first subject, lists its subdirectory names, and prints them — so you can see what modality folders exist (e.g., `MPRAGE`, `Accelerated_Sagittal_MPRAGE`, `2-Plane_Localizer`).

**Why this matters:** You need to know what folder names are present before writing the keyword-based search logic. This is exploratory — you wouldn't know your modality filter is correct without seeing the actual folder names first.

---

## Cell 8 — Inspect DICOM Metadata & Render a Slice

This is the most complex exploration cell. It:

1. **Walks the folder tree** of the first subject looking for an MPRAGE DICOM file (using `MODALITY_KEYWORDS`). Falls back to any `.dcm` file if no MPRAGE folder is found.
2. **Reads the DICOM file** using `pydicom.dcmread()`.
3. **Extracts metadata fields:** `PatientID`, `Modality`, `StudyDate`, `Rows`, `Columns`.
4. **Renders the raw pixel array** as a grayscale image using `matplotlib`.

```python
dcm = pydicom.dcmread(sample_dcm_path, force=True)
patient_id = getattr(dcm, 'PatientID', 'UNKNOWN')
...
plt.imshow(dcm.pixel_array, cmap='gray')
```

**Why `force=True`:** Some ADNI DICOM files have incomplete or non-standard headers. `force=True` tells pydicom to attempt reading even if the header is malformed.

**Why `getattr(..., 'UNKNOWN')`:** Defensive coding — not every DICOM file guarantees every metadata field exists. Using `getattr` with a default prevents `AttributeError` crashes.

**Why visualize a single slice:** Confirms that the pixel data is valid, the scan is brain anatomy (not noise or artifact), and that the grayscale intensities look correct before investing time in the full pipeline.

**What a DICOM slice looks like:** A single 2D cross-section (usually axial, sagittal, or coronal plane) of a brain — grayscale with bright white for bone/CSF, gray for brain tissue.

---

## Cell 9 — Single-Subject DICOM→NIfTI Test

```python
result = subprocess.run(
    ['dcm2niix', '-z', 'y', '-f', sample_subject, '-o', sample_nifti_output, sample_dicom_dir],
    capture_output=True, text=True
)
```

**What it does:** Runs `dcm2niix` on one subject's MPRAGE DICOM folder.

**Flags explained:**
- `-z y` — compress the output with gzip (produces `.nii.gz` instead of `.nii`)
- `-f sample_subject` — name the output file after the subject ID
- `-o` — output directory
- Final argument — input DICOM directory (one folder = one scan series)

**What `dcm2niix` does internally:** It reads all `.dcm` files in the folder, sorts them by slice position, reads the voxel spacing from DICOM headers, and stacks them into a properly oriented 3D array. It also writes a `.json` sidecar with scan parameters (echo time, repetition time, etc.).

**3D validation:**
```python
is_valid_3d = len(volume_shape) == 3 and all(dim > MIN_SLICES_FOR_3D for dim in volume_shape)
```
Checks that the output has exactly 3 dimensions and each dimension is > 20 voxels. A typical MPRAGE is ~256×256×176. A scout image would be something like 192×192×1 and would fail this check.

---

## Cell 10 — Phase 3: Pipeline Definition (Markdown)

Explains the 5 reusable functions defined in the next code cell and their roles.

---

## Cell 11 — Pipeline Functions (Core Cell)

This defines the entire preprocessing pipeline as 5 functions:

### `find_mprage_dicom_dir(subject_path)`

```python
for root, dirs, files in os.walk(subject_path):
    root_lower = root.lower()
    if any(token in root_lower for token in keyword_tokens):
        if any(f.lower().endswith('.dcm') for f in files):
            candidate_dirs.append(root)
```

**What it does:** Recursively walks the subject's folder tree. For each folder that contains `.dcm` files AND whose path contains an MPRAGE keyword, it records it as a candidate.

**Why case-insensitive:** Different scanner sites used different capitalizations (`SAG_IR`, `Sag_IR`, `MPRAGE`, etc.).

**Why sort by earliest date:**
```python
candidate_dirs.sort(key=lambda path: (os.path.basename(os.path.dirname(path)), path))
```
Some subjects have multiple time-point scans (longitudinal study). Taking the earliest visit minimizes selection bias — all subjects are measured at their earliest available scan, making cross-subject comparison fairer.

---

### `convert_dicom_to_nifti(dicom_dir, subject_id)`

Calls `dcm2niix` via `subprocess` (same as the test cell) and returns the path to the first `.nii.gz` file produced. The output is named by `subject_id` for clear provenance tracking.

---

### `validate_3d_volume(nifti_path)`

Loads the NIfTI with `ants.image_read()` and checks `len(shape)==3` and all dimensions `> MIN_SLICES_FOR_3D`. Returns `True`/`False`.

**Why this function exists as a separate step:** After `dcm2niix` runs, you don't know if the output is a proper 3D brain volume or a 2D scout until you inspect its shape. This gate prevents scout images from crashing the registration step.

---

### `preprocess_volume(nifti_path, mni_template)` — The Core Function

This applies all four major preprocessing transformations:

#### Step 1 — N4 Bias Field Correction

```python
bias_corrected = ants.n4_bias_field_correction(raw_image)
```

**What it is:** MRI scanners produce images with a low-frequency, smooth intensity gradient across the field of view — brighter in some regions, darker in others, due to radio-frequency (RF) coil non-uniformity. This is called **bias field** or **intensity inhomogeneity**.

**Why it's a problem:** If uncorrected, tissue segmentation and registration algorithms see the same tissue type (e.g., white matter) with different intensity values depending on its position in the image. This breaks any intensity-based analysis.

**What N4 does:** N4 (N4ITK) is an iterative algorithm that estimates the smooth bias field using a B-spline model and divides it out, producing a corrected image where the same tissue type has consistent intensity throughout.

**Why do it first:** Skull stripping and registration are more accurate on bias-corrected images because the intensity gradients don't confuse the algorithms.

---

#### Step 2 — Brain Extraction (Skull Stripping)

```python
if USE_ANTSPYNET:
    brain_probability_mask = brain_extraction(bias_corrected, modality='t1')
    binary_brain_mask = ants.threshold_image(brain_probability_mask, 0.5, 1.0)
else:
    binary_brain_mask = ants.get_mask(bias_corrected)

skull_stripped = bias_corrected * binary_brain_mask
```

**What it does:** Produces a binary mask (1 = brain, 0 = non-brain). Multiplying the image by the mask zeroes out the skull, eyes, neck, and scalp.

**Why remove the skull:** 
- Registration to MNI152 is dramatically more accurate when only brain tissue is aligned — the skull's shape varies a lot between people and would mislead the registration algorithm.
- Downstream analyses (cortical thickness, voxel-based morphometry) only care about brain tissue.

**`antspynet.brain_extraction()` details:** This is a trained deep-learning model (a U-Net convolutional neural network) that learned to segment brain vs. non-brain from thousands of labeled MRI scans. It outputs a **probability map** — each voxel gets a number from 0 to 1 representing how likely it is to be brain tissue. The `threshold_image(0.5, 1.0)` call binarizes this: any voxel with probability > 0.5 is declared "brain."

**Fallback `ants.get_mask()`:** Uses Otsu thresholding (a statistical method that finds the optimal intensity cutoff to separate two populations — brain vs. background). Less accurate than the deep-learning approach but always works.

---

#### Step 3 — Spatial Registration to MNI152

```python
registration_result = ants.registration(
    fixed=mni_template,
    moving=skull_stripped,
    type_of_transform=TRANSFORM_TYPE  # 'antsRegistrationSyNQuick[s]'
)
registered_image = registration_result['warpedmovout']
```

**What it does:** Finds a spatial transformation that warps the subject's brain to match the MNI152 template brain.

**Why this is necessary:** Every person's brain has a different size, shape, and orientation. If you want to compare voxel at coordinate (x=50, y=60, z=70) across 200 subjects, that coordinate needs to correspond to the same anatomical region in every brain. Without registration, a given coordinate might be in the motor cortex for one subject and the occipital lobe for another.

**What `SyNQuick[s]` means:**
- `SyN` = Symmetric Normalization — a **diffeomorphic** (topology-preserving, invertible, smooth) non-linear registration. This is not just a rigid rotation/translation; it warps every voxel independently while preserving anatomical topology (no tearing or folding).
- `Quick` = uses a multi-resolution strategy with fewer iterations than full `SyN`.
- `[s]` = uses the **SyN** deformation at the final stage (non-linear, as opposed to `[r]` which is rigid only).

**What `warpedmovout` is:** The subject's skull-stripped brain after being warped to match the MNI152 template space. Every voxel is now aligned to its anatomical equivalent in the template.

**The `[s]` flag fallback:**
```python
except ValueError as error:
    if 'does not exist' in str(error):
        registration_result = ants.registration(..., type_of_transform='SyN')
```
In some ANTs versions, the string `'antsRegistrationSyNQuick[s]'` may not be recognized. The fallback uses plain `'SyN'`.

---

#### Step 4 — Z-Score Intensity Normalization

```python
voxel_data = registered_image.numpy()
brain_voxels = voxel_data[voxel_data > 0]
normalized_data = (voxel_data - np.mean(brain_voxels)) / np.std(brain_voxels)
normalized_data[voxel_data == 0] = 0
```

**What it does:** Converts raw MRI intensity values to Z-scores: subtracts the mean brain intensity and divides by its standard deviation.

**Why:** Even after N4 correction, different MRI scanners produce images with completely different absolute intensity scales. A "200" in one scanner might correspond to "600" in another. Z-score normalization re-expresses every voxel's intensity as "how many standard deviations above or below the mean brain intensity" — making intensities comparable across subjects and across scanners.

**Why only use `brain_voxels > 0`:** The skull-stripping zeroed out non-brain voxels. If you included those zeros in the mean/std calculation, the statistics would be dominated by background noise rather than actual brain tissue. We compute the normalization statistics only on brain tissue and then force background voxels back to 0 at the end.

**Why the `std == 0` check:**
```python
if brain_voxels.size == 0 or np.std(brain_voxels) == 0:
    raise ValueError("Brain mask is empty or has zero variance after registration")
```
Division by zero would produce `NaN` values throughout the volume, silently corrupting the data. This catches a failed skull-strip or a degenerate registration result.

---

### `save_qc_figure(stage_images, mni_template, output_path)`

```python
fig, axes = plt.subplots(4, 1, figsize=(10, 14))
plotting.plot_anat(temp_files['raw'], ...)
plotting.plot_anat(temp_files['skull_stripped'], ...)
plotting.plot_anat(temp_files['normalized'], ...)
plotting.plot_anat(temp_files['template'], ...)
```

**What it does:** Creates a 4-panel figure showing the brain at each stage side by side: raw, skull-stripped, registered+normalized, and the MNI template.

**Why orthographic view:** `nilearn`'s `plot_anat` with `display_mode='ortho'` shows three simultaneous cross-sections through the center of the brain: axial (top-down), coronal (front-back), and sagittal (left-right). This is the standard neuroimaging QC view.

**Why four panels:**
- **Raw** confirms the DICOM-to-NIfTI conversion worked and the anatomy is intact.
- **Skull-stripped** confirms brain extraction removed the skull without eating into brain tissue.
- **Registered+normalized** confirms the warp to MNI space aligned the anatomy correctly.
- **MNI template** provides the ground truth to visually compare against.

**Why write to temp files first:** `nilearn.plotting.plot_anat()` takes a **file path** (not an in-memory array). The function saves ANTs images to temporary `.nii.gz` files, passes the paths to nilearn, then cleans up with `os.remove()` in a `finally` block (guaranteed cleanup even if an exception occurs).

---

## Cell 12 — Single-Subject Validation Run

```python
validation_subject = '135_S_4356'
mni_template = ants.image_read(ants.get_ants_data('mni'))
dicom_dir = find_mprage_dicom_dir(validation_subject_path)
nifti_path = convert_dicom_to_nifti(dicom_dir, validation_subject)
final_image, stage_images = preprocess_volume(nifti_path, mni_template)
ants.image_write(final_image, output_nifti)
save_qc_figure(stage_images, mni_template, qc_path)
display(Image(filename=qc_path))
```

**What it does:** Runs the entire pipeline end-to-end on exactly one subject and displays the QC figure inline.

**Why before the batch loop:** This is a **smoke test** / **dry run**. Running 200 subjects takes hours. If any function has a bug, a path error, or a logic flaw, you want to catch it in 1 subject (~2–3 minutes) rather than discovering it 5 hours into the batch.

**`ants.get_ants_data('mni')`:** ANTsPy ships with a bundled MNI152 template. This call returns the file path to that built-in template, so you don't need to download it separately.

**`gc.collect()`:** Explicitly triggers Python's garbage collector to release memory from all the large image arrays used during preprocessing. On Colab's ~12 GB RAM limit, this prevents out-of-memory errors.

---

## Cell 13 — Phase 4: Batch Processing (Markdown)

Explains all the engineering decisions that make the batch loop robust for Colab:
- Subject cap to fit within session time
- Resume capability (skip already-done subjects)
- Skip retry control
- Team sharding (non-overlapping work split)
- Per-worker log files
- Error isolation (try/except)
- Memory management

---

## Cell 14 — Build Subject Queue

```python
def worker_for_subject(subject_id, total_workers):
    digest = hashlib.md5(subject_id.encode('utf-8')).hexdigest()
    return int(digest, 16) % total_workers
```

**What it does:** Takes an MD5 hash of the subject ID string, converts it to an integer, and takes `% total_workers`. This deterministically maps every subject ID to exactly one worker index (0, 1, or 2).

**Why MD5 hash instead of alphabetical split:** Alphabetical splitting could accidentally cluster all "easy" or "hard" subjects on one worker if subject IDs correlate with scan quality (they sometimes do, e.g., by site). Hashing distributes subjects pseudo-randomly and evenly.

**Why it's deterministic:** The same subject ID always hashes to the same worker, no matter how many times you rerun the cell. This prevents overlap.

```python
already_processed = {
    os.path.splitext(os.path.splitext(f)[0])[0]
    for f in os.listdir(PREPROCESSED_DIR)
    if f.endswith('.nii.gz')
}
```

**What it does:** Builds a set of subject IDs that already have a `.nii.gz` output file. `os.path.splitext` twice strips both `.nii` and `.gz` from `subject_id.nii.gz`.

**Why:** Resume capability — if Colab disconnects mid-batch (very common), subjects already saved to Drive are automatically skipped on the next run.

```python
previously_skipped = {
    sid for sid, status in latest_status_by_subject.items()
    if status in {'skipped_no_mprage', 'skipped_not_3d'}
}
```

**Why:** Some subjects genuinely don't have an MPRAGE scan or only have 2D scout images. Re-processing them every time wastes time since they'll be skipped again. The `RETRY_PREVIOUS_SKIPS` flag gives you a manual override if you change the modality-detection logic and want to retry these.

The final queue:
```python
eligible_subjects → filter by worker hash → take first MAX_SUBJECTS
```

---

## Cell 15 — Load MNI Template & Open Log File

```python
mni_template = ants.image_read(ants.get_ants_data('mni'))
log_file = open(PROCESSING_LOG_PATH, 'a', newline='')
log_writer = csv.writer(log_file)
if not log_file_exists:
    log_writer.writerow(['subject_id', 'status', 'error', 'timestamp'])
```

**Why load MNI once before the loop:** Loading a NIfTI file from disk has I/O overhead. Loading it 200 times inside the loop would waste ~200 × 2 seconds = several minutes unnecessarily.

**Why append mode `'a'`:** Allows resuming — if the script stopped halfway, previous log entries are preserved and new ones are appended rather than overwriting the file.

**Why write CSV headers only if the file is new:** If the file already has headers from a previous run, appending headers again would corrupt the CSV.

---

## Cell 16 — The Batch Processing Loop

```python
for subject_id in progress_bar:
    dicom_dir = find_mprage_dicom_dir(subject_path)
    if dicom_dir is None:
        log_writer.writerow([subject_id, 'skipped_no_mprage', '', timestamp])
        continue
    
    nifti_path = convert_dicom_to_nifti(dicom_dir, subject_id)
    
    if not validate_3d_volume(nifti_path):
        log_writer.writerow([subject_id, 'skipped_not_3d', ...])
        continue
    
    final_image, stage_images = preprocess_volume(nifti_path, mni_template)
    ants.image_write(final_image, output_nifti)
    save_qc_figure(stage_images, mni_template, qc_path)
    
    log_writer.writerow([subject_id, 'success', '', timestamp])
    
    log_file.flush()
    gc.collect()
```

**What it does:** Iterates over every subject in the queue and applies the full pipeline to each one.

**The flow per subject:**
1. Find the MPRAGE DICOM directory → skip if not found
2. Convert DICOMs to NIfTI
3. Validate it's a genuine 3D volume → skip if not
4. Run `preprocess_volume()` (bias correction → skull strip → register → normalize)
5. Save the output `.nii.gz` to Drive
6. Save the QC figure PNG to Drive
7. Log the result

**`log_file.flush()`:** Forces the CSV write buffer to disk immediately after each subject, rather than buffering in memory. If Colab crashes, you don't lose the log entries for already-completed subjects.

**`gc.collect()`:** Frees all the large NumPy arrays and ANTs image objects from the current subject before the next one loads. Without this, memory usage accumulates and Colab crashes with an OOM error.

**`try/except Exception`:** If any step throws any error (corrupt DICOM, registration failure, empty brain mask), the error message is logged, and the loop continues with the next subject. This prevents one bad subject from halting the entire batch.

**`progress_bar.set_postfix(ok=..., fail=..., skip=...)`:** Displays live success/failure/skip counts in the progress bar so you can monitor quality at a glance without waiting for the loop to finish.

---

## Cell 17 — Phase 5: Quality Control Review (Markdown)

Explains that after batch processing, the log file is reviewed for success rates and a random sample of QC images is displayed.

---

## Cell 18 — Processing Log Summary

```python
processing_log = pd.read_csv(PROCESSING_LOG_PATH)
print(processing_log['status'].value_counts().to_string())

failed_subjects = processing_log[processing_log['status'] == 'failed']
print(failed_subjects[['subject_id', 'error']])
```

**What it does:** Loads the CSV log into a pandas DataFrame and counts how many subjects fell into each status category: `success`, `failed`, `skipped_no_mprage`, `skipped_not_3d`.

**Why review failures:** Failures may indicate systemic issues — e.g., if 40 subjects all failed with "brain mask is empty," that suggests the skull-stripping model is producing poor masks, and the pipeline needs to be adjusted.

---

## Cell 19 — Random QC Image Sample

```python
sampled_subjects = random.sample(successful_subjects, sample_size)
for subject_id in sampled_subjects:
    display(Image(filename=qc_path))
```

**What it does:** Randomly selects 5 successfully processed subjects and displays their 4-panel QC figures inline.

**Why random sampling:** Manually reviewing all 200 QC images is impractical. A random sample of 5 gives a statistically representative view of preprocessing quality. If all 5 look correct, the pipeline is likely working properly across the board.

**What to look for in the QC images:**
- **Raw:** Brain and skull visible, correct orientation.
- **Skull stripped:** Only brain tissue remains — no residual skull fragments, and no brain tissue accidentally removed.
- **Registered:** Brain shape matches the MNI template outline closely.
- **MNI template:** The reference — your registered brain should resemble this.

---

## Overall Pipeline Summary

The entire notebook implements this flow:

$$\text{Raw DICOM slices} \xrightarrow{\text{dcm2niix}} \text{3D NIfTI} \xrightarrow{\text{N4}} \text{Bias Corrected} \xrightarrow{\text{antspynet}} \text{Skull Stripped} \xrightarrow{\text{SyNQuick}} \text{MNI Space} \xrightarrow{\text{Z-score}} \text{Analysis Ready}$$

Each step solves a specific problem:
| Step | Problem Solved |
|---|---|
| DICOM → NIfTI | Convert scanner format to analysis format |
| N4 Bias Correction | Remove scanner-induced intensity gradients |
| Skull Stripping | Remove non-brain tissue that confounds alignment |
| Registration to MNI152 | Make voxel coordinates anatomically comparable across subjects |
| Z-Score Normalization | Make intensity values comparable across scanners |
| QC Visualization | Human verification that no step introduced distortions |
