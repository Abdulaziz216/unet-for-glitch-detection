# Glitch Segmentation on ACT Time-Ordered Data

This repository builds on the SUDS2025 work and extends it with a full pipeline for simulating, processing, and segment glitches in Time-Ordered Data (TOD) from the Atacama Cosmology Telescope (ACT).

---

## 📑 Table of Contents
- [What’s included vs. excluded](#whats-included-vs-excluded)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start (no environment required)](#-quick-start-no-environment-required)
- [🔧 Environment](#-environment)
- [👥 Contributors (SUDS 2025)](#-contributors-suds-2025)
- [🙏 Acknowledgements](#-acknowledgements)

---

## What’s included vs. excluded

- ✅ **Included**: Simulated mini-dataset, simulation notebooks, U-Net training/inference notebooks with saved outputs.
- 🔒 **Excluded**: Real ACT/SO TODs and any derived products (NDA), runnable training/inference with real data.

---

## 📁 Project Structure
```
* `notebooks/`
  Development scripts and experiment notebooks
  ├── `generationScript.ipynb`: Main image generation notebook using simulation helpers
  ├── `tod_simulator_v2.ipynb`: Original TOD simulation script by Yilun Guan
  ├── `context.py`: Shared notebook config/context (e.g. paths)
  └─── `README.md`: Overview of notebook workflows

* `src/`
  Core helper functions for TOD simulation and image conversion

* `data/` Organized storage for both real and simulated datasets
├─ `data/real_data/`
├─ `data/simulated_data/output_dataset120`
└─ `README.md`

* `unet_model/`
├─ `Tiny_U-net_pipeline/` # Inference & visualization notebooks (saved outputs; view-only)
├─ `Training/` # Training notebooks, metrics, sample predictions (saved outputs; view-only)
└─ `README.md`

* `layer-cam_model/` *(placeholder)*
  Will contain Grad-CAM/Layer-CAM visualizations and analysis tools
```


---

## Quick Start (no environment required)

1. **Browse the simulated sample**: `data/simulated_data/output_dataset1200/`.  
2. **Open notebooks to review results** (no execution):
   - `Tiny_U-net_pipeline/` → end-to-end inference visuals (probability maps, masks, detector hits).
   - `Training/` → loss curves, IoU/precision/recall, qualitative predictions.
3. Need full reproducibility? It requires NDA data and our internal environment; contact project leads.

---


## 🔧 Environment

This project assumes a working `glitchenv` conda environment from helen cluster with packages like:

* `so3g`, `moby2`, `pixell`, `numpy`, `scipy`, `shapely`, `alphashape`, `matplotlib`

---


## 👥 Contributors (SUDS 2025)

This project was developed by members of the SUDS 2025 team:

* **Anas Alshehri** — U-Net model implementation, simulation evaluation, real data pipeline
* **Abdulaziz Alkharjy** — Simulation data generation, real data pipeline
* **Mehtab Cheema** — Layer-CAM model implementation, real data pipeline

---

## 🙏 Acknowledgements

We gratefully acknowledge the support and guidance from:

* **Adam Hincks**
* **Renée Hložek**
* **Yilun Guan**
* **KAUST Academy**
* **Data Science Institute**

Their contributions, mentorship, and resources were essential to the success of this project.
