# CMB Pipeline Demo - Annotated Notebook Overview

This README provides a structured overview of the **cmb_pipeline_demo** notebook and its companion module `cmb_modular_pipeline.py`. It explains the purpose of each section and how the notebook ties together the reusable pipeline functions.

---

## 1. Imports & Environment
- Import core libraries: NumPy, Torch, Matplotlib, SciPy.
- Import custom pipeline functions from `cmb_modular_pipeline.py`.
- Configure device (CPU/GPU) and seeds for reproducibility.

---

## 2. Data Loading & Geometry
- Use `load_tod_ctx` to load Time-Ordered Data (TOD) with preprocessing (demean, detrend, calibration).
- Select uncut detectors for a frequency band (e.g., f090).
- Build focal-plane geometry (`FocalPlane`, `FocalPlaneFlex`) with triangulation for TOD <-> image mapping.

---

## 3. Preprocessing
- Convert TOD (DxT) -> focal-plane video (TxHxW) with `tod_to_video`.
- Apply high-pass filtering (`fc`, `order`) if specified.
- Reduce video slices to 2D images with `slices_to_images` (max/sum/mean reduction + normalization).

---

## 4. Model Definition
- Define **TinyUNet**, a lightweight U-Net variant for segmentation.
- Architecture: encoder -> bottleneck -> decoder with skip connections.
- Outputs logits (B, C, H, W) where C = number of classes (CR, PS, EL).

---

## 5. Training Setup
- Configure loss function, optimizer, learning rate scheduler.
- Define `run_epoch` to handle both training and validation steps (returns loss, IoU, accuracy).
- Track per-class IoUs and accuracies.

---

## 6. Training Loop
- Iterate for N epochs:
  - Run training and validation via `run_epoch`.
  - Save checkpoints: `last` each epoch, `best` when validation IoU improves.
  - Early stopping when validation loss plateaus beyond `PATIENCE` epochs.

- Log metrics: train/val loss, per-class IoU, accuracy.

---

## 7. Visualization
- Plot training vs validation loss curves.
- Plot per-class IoU over epochs (CR, PS, EL).
- Plot accuracy curves for training vs validation.

---

## 8. Inference & Detection
- Use `run_model` to batch inference on images, returning per-class probability maps.
- Use `preds_to_det_hits` to back-project probabilities to per-detector hits (with thresholds).
- Use `detect_on_slices` for an end-to-end pipeline: TOD -> video -> images -> probs -> hits + metadata.

---

## 9. Export & Analysis
- Convert structured results (`SliceResult[]`) into a tidy pandas DataFrame with `results_to_dataframe` for further analysis or CSV export.

---

## 10. Visualization Utilities
- `plot_filtered_images`: show input slices.
- `plot_prob_maps`: per-class probability heatmaps.
- `plot_composite`: RGB overlay of all classes.
- `plot_class_scatter` / `plot_class_scatter_hits`: scatter detectors on focal-plane with color-coded classes.
- `plot_timestreams`: visualize high-pass TOD slices for selected detectors.

---

## Usage Pattern
Typical workflow in the notebook:

```python
ctx  = load_tod_ctx(tod_id, depot=DEPOT_PATH, release=RELEASE_TAG, band="f090", s_rate_hz=S_RATE_HZ)
fp   = build_fplane(ctx, grid_res=32)
video = tod_to_video(ctx, fp, hp_filter=(2.0, 5))
images = slices_to_images(video, slices, reduce="max")

# Train or load model
model = TinyUNet(n_classes=3)
# ... training loop ...

# Detect on slices
res  = detect_on_slices(ctx, fp, model, DEVICE, slices, reduce="max", thr_per_class={"CR":0.5,"PS":0.5,"EL":0.5})
df   = results_to_dataframe(res)
```

---

## Key Concepts
- **TOD**: Raw time-ordered detector data (DxT).
- **Video**: Resampled TOD into focal-plane grid (TxHxW).
- **Images**: Reduced slices of video, fed into TinyUNet.
- **DetHits**: Per-detector classification results with probabilities & labels.
- **SliceResults**: Structured object holding hits + metadata per slice.

---

This README is meant to complement the heavily commented code and notebook, serving as a quick reference for understanding the workflow at a high level.
