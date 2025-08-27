## Simulation Helper Modules

This folder contains helper modules used across the simulation and training pipeline. These files are **not executable on their own** — they are designed to be **imported and reused** in other scripts.

- **`simulatedDataGenerationHelper.py`**  
  Handles the generation of synthetic Time-Ordered Data (TOD) with injected point sources and cosmic rays. Produces image and mask outputs for model training.

- **`simulationDataMappingHelper.py`**  
  Provides functions to map TOD to image space using focal plane geometry. Supports both grid-based (bilinear) and flexible (triangulated) interpolation methods.

These modules form the **backbone of all simulated data preparation** in this project.
