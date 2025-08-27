# CMB Modular Pipeline

This repository contains a modular Python pipeline for processing
**Time-Ordered Data (TOD)** from CMB telescopes, mapping it to the focal
plane, running a U-Net segmentation model, and projecting detections
back to individual detectors.

------------------------------------------------------------------------

## Features

-   **TinyUNet model**: Minimal encoder--decoder CNN for pixel-level
    segmentation.
-   **Flexible focal-plane resampling**: `FocalPlaneFlex` provides
    bidirectional mapping between detectors and a dense grid using
    Delaunay triangulation and barycentric weights.
-   **Preprocessing utilities**:
    -   Percentile clipping & normalization
    -   High-pass filtering with Butterworth filters
-   **End-to-end detection**:
    -   TOD → focal-plane video → slice images → U-Net model →
        per-detector hits
    -   Metadata and results structured as dataclasses
-   **Visualization utilities**:
    -   Filtered images
    -   Per-class probability maps
    -   Composite RGB overlays
    -   Detector scatters and timestream plots
-   **Export tools**:
    -   Results converted to tidy Pandas DataFrames for analysis.

------------------------------------------------------------------------

## File Structure

-   **cmb_modular_pipeline.py**: Core library containing models,
    utilities, and pipeline functions.
-   **cmb_pipeline_demo.ipynb**: Example Jupyter notebook demonstrating
    usage and plotting.

------------------------------------------------------------------------

## Core Classes and Data Structures

-   **`TinyUNet`**: Minimal U-Net-style CNN for segmentation.
-   **`FocalPlane`**: Represents detector coordinates in radians.
-   **`FocalPlaneFlex`**: Handles resampling between detectors and grid
    images.
-   **`TodContext`**: Container for TOD data and metadata.
-   **`FpContext`**: Bundles focal-plane geometry and resampling caches.
-   **`DetHit`**: Represents a per-detector detection with probabilities
    and labels.
-   **`SliceResult`**: Stores detection results for a specific time
    slice.

------------------------------------------------------------------------

## Typical Workflow

1.  **Load TOD and build contexts**:

    ``` python
    ctx  = load_tod_ctx(tod_id, depot=DEPOT_PATH, release=RELEASE_TAG, band="f090", s_rate_hz=S_RATE_HZ)
    fp   = build_fplane(ctx, grid_res=32)
    ```

2.  **Convert TOD to focal-plane video and slice images**:

    ``` python
    video  = tod_to_video(ctx, fp, hp_filter=(2.0, 5))
    images = slices_to_images(video, [(0,1000), (1000,2000)], reduce="max")
    ```

3.  **Run model and get per-detector hits**:

    ``` python
    results = detect_on_slices(ctx, fp, model, DEVICE, [(0,1000),(1000,2000)], thr_per_class={"CR":0.5,"PS":0.5,"EL":0.5})
    ```

4.  **Export results to DataFrame**:

    ``` python
    df = results_to_dataframe(results)
    ```

------------------------------------------------------------------------

## Visualization

-   **Filtered images**: `plot_filtered_images(images)`
-   **Probability maps**: `plot_prob_maps(probs[i])`
-   **Composite overlays**: `plot_composite(probs[i])`
-   **Detector scatter plots**: `plot_class_scatter_hits(img, fp, hits)`
-   **Timestreams**: `plot_timestreams(hp, (s0, s1), masks)`

------------------------------------------------------------------------

## Requirements

-   `numpy`
-   `torch`
-   `scipy`
-   `matplotlib`
-   `pandas` (optional, for export)
-   `cutslib` (for TOD access)

------------------------------------------------------------------------

## Notes

-   The module is designed to be **imported** into notebooks or scripts.
-   No code executes on import; you explicitly call functions in your
    pipeline.
-   Supports caching of high-pass filtered data and focal-plane videos
    for speed.
-   Modular design makes it easy to test each component separately.
