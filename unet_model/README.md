# U-Net Results Snapshot (Read-Only)

This snapshot contains the **final U-Net results** we can share publicly.
Because the **real ACT data are NDA-protected**, the notebooks here are **view-only**: you can read all saved cell outputs (tables, figures, masks), but you **cannot re-run** them without the private data and environment.

> ⚠️ Scope: This snapshot highlights the outcomes we reached but **may not reflect every experiment or intermediate step** performed during the project.

---

## Folders

```
Tiny_U-net_pipeline/   # End-to-end pipeline notebook(s): preprocessing → inference → visualizations
Training/              # Training notebook(s), metrics, and saved cell outputs for key runs
```

* **Tiny\_U-net\_pipeline/**
  Walkthrough of the inference/visualization flow. All critical cells have their **outputs saved** so results (e.g., probability maps, segmentation masks, detector hits) are visible without execution.

* **Training/**
  Training notebook(s) with **saved outputs** (loss curves, IoU/precision/recall, sample predictions). Re-execution requires NDA data and the original cluster setup.

---

## Why notebooks are non-runnable

* **Data privacy:** Real ACT TODs and derived products are **not distributed** due to NDA.
* **Environment coupling:** Paths and dependencies target our internal cluster; reproducing runs requires the same data and setup.

---

## How to use this snapshot

* **Open notebooks and scroll** to view saved outputs (figures/metrics) for the final results.
