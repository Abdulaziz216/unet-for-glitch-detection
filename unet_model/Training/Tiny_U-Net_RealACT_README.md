# Tiny U‑Net Pipeline on Real ACT Data — Two Notebooks, One Workflow

This README documents **two Jupyter notebooks** in this folder that train and visualize a tiny U‑Net for **multi‑label segmentation** on *real* ACT focal‑plane images:

- `Anas's_TinyUnet_realData_noELglitches.ipynb`
- `Anas's_TinyUnet_realData_withELglitches.ipynb`

They are **largely identical** in structure, hyper‑parameters, and training logic. The **main differences** are only in **the third class channel** (placeholder vs. true *Electronic Glitch*) and in **how the data splits are wired** (details below). Everything described here is taken directly from the notebooks’ code.

---

## 1) What both notebooks do

**Goal.** Train a compact U‑Net (2× down/2× up) for **multi‑label pixel‑wise segmentation** of three classes (**CR**, **PS**, **EL**) on **real 32×32 focal‑plane images** (images are loaded as single‑channel float arrays).

**End‑to‑end flow (same in both):**
1. **Imports & config**: PyTorch, TorchMetrics, NumPy, Matplotlib, tqdm.
2. **Sanity check**: Load one example `.png` and show the raw image.
3. **Optional normalization**: Percentile clip to **[0.5, 99.5]** then scale to **[0, 1]** via `pct_clip_norm(img)`.
4. **Dataset**: `TelescopeDataset` reads images and stacks three binary masks per sample (shape `(3, H, W)`).
5. **Model**: `TinyUNet` — encoder(1→32→64), bottleneck(64→128), decoder with transposed‑convs and skip connections, `Dropout2d(p=0.25)`, final `Conv2d(..., out_channels=3, kernel_size=1)`.
6. **Loss**: **0.5 × BCEWithLogits + 0.5 × DiceLoss**.
7. **Metrics**: `MultilabelJaccardIndex` (IoU) and `MultilabelAccuracy` with **threshold=0.5** (class‑wise and mean).
8. **Optimizer/Scheduler**: **Adam(lr=1e‑3, weight_decay=1e‑4)** + **ReduceLROnPlateau(mode="min", patience=3, factor=0.3)**.
9. **Training loop** with early‑stopping style stall counter (**patience=8**, **min_delta=1e‑5**), per‑epoch logging, and checkpointing.
10. **Plots**: Loss curves, per‑class IoU, and prediction visualizations (raw probability maps and composite overlays).

**Hyper‑parameters (identical):**
- `NUM_CLASSES = 3`  (CR, PS, EL)
- `BATCH_SIZE = 8`
- `EPOCHS = 10 * 12`  → **120 epochs**
- `LR = 1e-3`
- `NUM_WORKERS = 32`
- Device auto‑select: `"cuda" if torch.cuda.is_available() else "cpu"`

**Checkpoints & logs (same mechanism, different names):**
- Checkpoints: saved **every epoch** with tags `{last|best}`.
- Logs: a CSV row is appended at the end of a run (fields: batch size, num workers, dataset size, losses, IoU, pixel accuracy, GPU id, elapsed time).

**Visualization utilities (shared):**
- `composite_map(probs, thr=0.5)` builds a 3‑color overlay (CR, PS, EL) from predicted probability maps.

---

## 2) Expected data layout (both notebooks)

Both notebooks expect the **same folder structure** rooted at:

```
.../data/real_converted_images32/
├── images/
│   ├── cosmic_ray/
│   ├── point_source/
│   └── <third_class_folder>/
└── masks/
    ├── cosmic_ray/
    ├── point_source/
    └── <third_class_folder>/
```
Each image is a **grayscale float PNG**. Each mask is a **binary PNG** per class. The **third folder differs** between the notebooks:

- In the **with‑EL** notebook, the folder is **`electronic_glitch/`** and is **used as a real third class**.
- In the **no‑EL** notebook, the folder used in code is **`none/`** (a placeholder channel).

> The dataset class stacks **three channels** in both notebooks. In the **no‑EL** version this third channel comes from `masks/none/…` (so it will be all‑zero if no masks exist there), while all plotting code still labels channel 3 as “EL”.

---

## 3) Side‑by‑side: what’s identical vs what’s different

### Identical
- **Architecture**: same `TinyUNet` with 2 encoders, bottleneck, and 2 decoders, `Dropout2d(0.25)` and BatchNorm+ReLU blocks.
- **Loss/metrics/optim/scheduler**: shared configuration (see §1).
- **Normalization**: same `pct_clip_norm` (clip to [0.5,99.5] percentiles → scale to [0,1]).
- **Epoch function**: `run_epoch` computes the composite loss, updates IoU/Accuracy, and returns `(avg_loss, mean_iou, mean_acc, per_class_iou, per_class_acc)`.
- **Checkpoints**: saved every epoch with `save_ckpt`, and a `best` snapshot when validation IoU improves.
- **Plots**: Training/validation loss, per‑class IoU curves, and grid visualizations of inputs, per‑class probability maps, and composite overlays.

### Different
1. **Class list used to build samples & masks**
   - **with‑EL**: `['cosmic_ray','point_source','electronic_glitch']` (true 3rd class).
   - **no‑EL**:   `['cosmic_ray','point_source','none']` (3rd channel is a placeholder).
2. **Data splits wiring**
   - **no‑EL**: Builds **train/val/test** indices (70/15/15) and correctly constructs **`Subset` datasets** → **distinct** loaders.
   - **with‑EL**: Computes split indices **but constructs all three `DataLoader`s directly on the full dataset** (`ds`) — so **train/val/test loaders all see the same data** (i.e., splits are not applied in the loader construction).
3. **Checkpoint & log paths**
   - **no‑EL**: `checkpoints_multilabel_noEL/` and `run_log_multilabel_noEL.csv`.
   - **with‑EL**: `checkpoints_multilabel_withEL/` and `checkpoints_multilabel_withEL.csv`.
4. **Per‑class IoU logging fields**
   - **no‑EL**: Appends `iou_cosmic`, `iou_pointsrc` to CSV (no separate EL field).
   - **with‑EL**: Includes all three (`iou_cr`, `iou_ps`, `iou_el`) in the “fine‑tuning log” cell (see note below).
5. **Visualization threshold in one cell**
   - **no‑EL**: Uses default `thr=0.5` in `composite_map` calls.
   - **with‑EL**: One visualization cell calls `composite_map(pred, **thr=0.75**)` for a slightly stricter overlay.


---

## 4) Key code blocks (present in both)

- **Normalization**
  ```python
  def pct_clip_norm(img, lo_pct=0.5, hi_pct=99.5):
      lo, hi = np.percentile(img, [lo_pct, hi_pct])
      img2 = np.clip(img, lo, hi)
      return (img2 - lo) / (hi - lo + 1e-12)
  ```

- **Dataset** (class & mask stacking; third folder differs as described in §3)
  ```python
  class TelescopeDataset(Dataset):
      def __init__(self, img_dir, mask_dir_base, transform=None):
          self.transform = transform
          self.samples = []
          for cls in [<CR>, <PS>, <THIRD_CLASS>]:
              for p in (img_dir/cls).glob("*.png"):
                  self.samples.append((p, cls))
          self.mask_dir_base = mask_dir_base

      def __getitem__(self, idx):
          img_path, cls = self.samples[idx]
          img = np.array(Image.open(img_path).convert('F'), dtype=np.float32)
          if self.transform: img = self.transform(img)
          img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,H,W)

          masks = []
          for mcls in [<CR>, <PS>, <THIRD_CLASS>]:
              mpath = self.mask_dir_base / mcls / img_path.name
              if mpath.exists():
                  m = (np.array(Image.open(mpath).convert('L'), dtype=np.uint8) > 0).astype(np.uint8)
              else:
                  m = np.zeros_like(img, dtype=np.uint8)
              masks.append(m)
          m_t = torch.tensor(np.stack(masks, 0), dtype=torch.float32)  # (3,H,W)
          return img_t, m_t
  ```

- **Model**
  ```python
  class TinyUNet(nn.Module):
      def __init__(self, n_classes=3, base_c=32, p_drop=0.25):
          super().__init__()
          self.enc1 = nn.Sequential(ConvBNReLU(1, base_c),  ConvBNReLU(base_c, base_c))
          self.enc2 = nn.Sequential(nn.MaxPool2d(2),        ConvBNReLU(base_c, base_c*2))
          self.bott = nn.Sequential(nn.MaxPool2d(2),        ConvBNReLU(base_c*2, base_c*4))
          self.up1  = nn.ConvTranspose2d(base_c*4, base_c*2, 2, stride=2)
          self.dec1 = nn.Sequential(ConvBNReLU(base_c*4, base_c*2), nn.Dropout2d(p_drop))
          self.up2  = nn.ConvTranspose2d(base_c*2, base_c,   2, stride=2)
          self.dec2 = nn.Sequential(ConvBNReLU(base_c*2, base_c),   nn.Dropout2d(p_drop))
          self.outc = nn.Conv2d(base_c, n_classes, 1)
      def forward(self, x):
          e1 = self.enc1(x); e2 = self.enc2(e1); b = self.bott(e2)
          d1 = self.dec1(torch.cat([self.up1(b), e2], 1))
          d2 = self.dec2(torch.cat([self.up2(d1), e1], 1))
          return self.outc(d2)  # logits
  ```

- **Loss, metrics, optim, scheduler**
  ```python
  bce_loss = nn.BCEWithLogitsLoss()
  dice_loss = DiceLoss()
  metric_iou = MultilabelJaccardIndex(num_labels=3, threshold=0.5, average=None).to(DEVICE)
  metric_acc = MultilabelAccuracy(num_labels=3, threshold=0.5, average=None).to(DEVICE)
  opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.3)
  ```

- **Training loop** (early‑stopping style stall counter = 8, min delta = 1e‑5).

- **Checkpoints & logging**
  ```python
  save_ckpt(model, opt, epoch, tag)  # tag in {'last','best'}
  log_run(...fields...)              # appends one CSV row
  ```

- **Visualization**
  - Grid: Input, per‑class probability maps (`CR`, `PS`, `EL`), composite overlay.
  - Optional random single‑example grid (with per‑pixel probability scatter for each channel) in the **with‑EL** notebook.

---

## 5) How to run

1. **Install dependencies** (PyTorch, TorchVision, TorchMetrics, NumPy, Matplotlib, PIL/Pillow, tqdm).
2. **Prepare data folder** as in §2; ensure class subfolders and matching mask file names exist.
3. **Open either notebook** and run top‑to‑bottom.
   - GPU is auto‑used if available; otherwise CPU.
   - Checkpoint and CSV log files are written in the notebook working directory.
4. **Inspect outputs**:
   - Checkpoints: `checkpoints_multilabel_{noEL|withEL}/unet_multilabel_{noEL|withEL}_ep<epoch>_{last|best}.pt`
   - CSV log: `{run_log_multilabel_noEL.csv | checkpoints_multilabel_withEL.csv}`
   - Plots: loss/IoU/accuracy curves and qualitative predictions.

---

## 6) Notes & gotchas

- **Class‑3 label**:
  - **with‑EL** trains on real `electronic_glitch` masks.
  - **no‑EL** uses a placeholder `none` channel; plotting still labels it as “EL”, so expect that channel’s IoU to reflect the placeholder behavior (e.g., near‑zero if masks are empty).
- **Data splits**:
  - **no‑EL** uses **`Subset`** objects for train/val/test.
  - **with‑EL** currently constructs loaders on the **full dataset** for all three loaders; if you intend non‑overlapping splits, replace those loaders with `DataLoader(Subset(ds, train_i), ...)`, etc.
- **“Fine‑tuning log” cell (with‑EL only)**: It’s a **template** referencing variables not defined elsewhere. It won’t run unless you define those variables or adapt it to your current training run.

---

## 7) Quick reference

- **Hyper‑params**: `BATCH_SIZE=8`, `EPOCHS=120 (10×12)`, `LR=1e‑3`, `NUM_CLASSES=3`.
- **Loss**: `0.5*BCEWithLogits + 0.5*Dice`.
- **Metrics**: IoU & Pixel Accuracy at threshold 0.5.
- **Scheduler**: ReduceLROnPlateau(patience=3, factor=0.3).
- **Early‑stop style**: patience 8, min delta 1e‑5.
- **Checkpoints**: saved each epoch (`last` + `best`).

