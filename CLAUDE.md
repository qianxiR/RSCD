# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSCD (遥感影像变化检测系统) V1.0 — a remote sensing change detection system using deep learning (X3D 3D-CNN). Identifies changes between temporal image/raster pairs for urban planning, environmental monitoring, and disaster assessment.

## Architecture

MVC three-tier architecture (single process, no HTTP layer):

```
Frontend (PySide6/Qt6)  ──in-process──▶  Controller (local task scheduling)  ──▶  Backend (X3D Model, PyTorch/CUDA)
Frontend/                                    Controller/                                  Backend/
```

**Data flow**: Frontend copies images to shared dirs (`t1/`, `t2/`) → calls `Controller.local_service.submit_detection()` → background thread runs X3D inference via Backend → results written to `output/` → frontend polls `get_task_status()` and displays.

**Unified processing flow**: single image / batch (no more image/raster split). Whether the georeference is preserved is auto-probed per image via `has_georeference()` (`Backend/processing/common/geo_probe.py`): georeferenced inputs → GeoTIFF mask + vectors (SHP/GeoJSON), non-georeferenced inputs → PNG mask (vectors skipped).

### Model Input Format

The X3D model treats change detection as a 3D video understanding task. Input is a 3-frame pseudo-video: `[pre-image, perception-map, post-image]`. `input_clip_length=3`, `depth_factor=5.0` (X3D-L variant). The Encoder uses X3D backbone with learnable tokens and temporal difference enhancement; the ChangeDecoder produces binary change masks.

## Key Modules

| Path | Purpose |
|---|---|
| `start_app.py` | Entry point — sets up shared dirs, launches PySide6 app |
| **Frontend/** | **View layer — PySide6 UI** |
| `Frontend/app.py` | `RemoteSensingApp`, `HomePage` (single entry, no image/raster split) |
| `Frontend/widgets.py` | `ZoomableLabel`, `NavigationFunctions` |
| `Frontend/theme.py` | `ThemeManager` (dark/light theme) |
| `Frontend/views/image_import.py` | `ImportBeforeImage`, `ImportAfterImage` |
| `Frontend/views/change_detection.py` | `ExecuteChangeDetectionTask` (calls `local_service.submit_detection`) |
| `Frontend/views/batch_dialog.py` | `BatchProcessingDialog`, `BatchProcessing` |
| `Frontend/views/grid_cropping.py` | `GridCropping` (fishnet segmentation) |
| `Frontend/views/clear_task.py` | `ClearTask` |
| `Frontend/views/common/` | Shared UI tools: `thread_pool` (singleton+atexit), `qt_logging` (ThreadSafeLogMixin), `styles` (QSS generators), `utils` (`parse_grid_size`) |
| `Frontend/views/training_dialog.py` | `TrainingDialog`, `TrainingModule`, `TrainingWorker` — model training UI with real-time logs |
| **Controller/** | **Controller layer — local task scheduling (no HTTP/FastAPI)** |
| `Controller/local_service.py` | Frontend entry point: `submit_detection` / `get_task_status` / `start_training` / `get_training_progress` |
| `Controller/detection_service.py` | `ChangeDetectionModel` — lazy-loads Backend `process_and_save`, routes by batch flag |
| `Controller/task_manager.py` | `detection_tasks` dict + `run_detection_task()` — synchronous status machine (pending→running→completed/failed) |
| `Controller/default_args.py` | `DEFAULT_INFERENCE_ARGS` + `build_default_args()` — centralized inference config |
| `Controller/training_task_manager.py` | `training_tasks` dict + `run_training_task()` — training task scheduling with `_training_lock` |
| **Backend/** | **Model layer — AI + processing** |
| `Backend/network/encoder.py` | `Encoder` (X3D backbone + feature enhancement), `Trainer` |
| `Backend/network/x3d.py` | X3D architecture (from Facebook PyTorchVideo) |
| `Backend/network/decoder.py` | `ChangeDecoder` |
| `Backend/network/network_utils.py` | Loss functions, metrics, weight init |
| `Backend/processing/single_image.py` | Single-image change detection — auto-probes georef: block+patch sliding (no coord) or geo-preserving sliding (with coord) |
| `Backend/processing/batch_image.py` | Batch processing — pairs `.png/.jpg/.jpeg/.tif/.tiff`, delegates each pair to `single_image.process_and_save`, merges vectors |
| `Backend/processing/common/` | Shared tools: `model_cache`, `geo_probe` (`has_georeference`), `transforms_helper`, `sliding_window`, `memory`, `visualization`, `io_utils` |
| `Backend/processing/raster/` | Shared raster tools (GDAL-dependent): `geotiff_io`, `geo_transform`, `vector_export` |
| `Backend/training/` | Training main loop: `train_loop.train_model()` — epoch loop with loss/optimizer/checkpoint |
| `Backend/data/dataset.py` | Dataset classes (BCD, SCD, BDA, Caption) |
| `Backend/data/transforms.py` | Data augmentation and normalization transforms |
| `Backend/evaluation/metrics.py` | `AverageMeter`, `ConfuseMatrixMeter`, metric functions |
| **utils/** | **Shared utilities** |
| `utils/paths.py` | `PROJECT_ROOT` via `.project_root` marker, `setup_module_paths()`, shared dirs (`T1_DIR`/`T2_DIR`/`OUTPUT_DIR`/`ensure_shared_dirs`) |

## Commands

### Run the App

```bash
pip install -r requirements.txt
python start_app.py
```

The app is a single-process desktop application. There is no separate server to start — `Controller` runs in-process and calls `Backend` directly.

## Tech Stack

- **AI**: PyTorch 2.8, CUDA 12.6, PyTorchVideo, einops, albumentations
- **GIS**: GDAL, Rasterio, GeoPandas, Fiona, Shapely
- **Frontend**: PySide6 (Qt6), OpenCV, NumPy, SciPy
- **Infra**: Conda environment, NVIDIA CUDA

## Codebase Conventions

- Primary language: Python 3.10+, code comments and UI in Chinese
- Conda for environment management
- No formal linting or CI/CD
- Model checkpoint at `checkpoint/X3D_L.pyth` (~50MB), `checkpoint/checkpoint.pth.tar`
- Shared data dirs (unified under `data/`): `data/t1/` (before images), `data/t2/` (after images), `data/output/` (results) — defined in `utils/paths.py` as `T1_DIR`/`T2_DIR`/`OUTPUT_DIR`, initialized via `ensure_shared_dirs()`
- `.project_root` marks repository root for path resolution

## Cross-Layer Import Rules

- Frontend → Controller: `from Controller.local_service import submit_detection, get_task_status, start_training, get_training_progress`
- Controller → Backend: `from Backend.processing.xxx import ...`, `from Backend.network.xxx import ...` (lazy-imported inside functions to speed up startup)
- Backend internal: relative imports `from .xxx import ...`
- All → utils: `from utils.paths import ...`
- Frontend must NOT import Backend directly

## Path Resolution

- `utils/paths.py` defines `PROJECT_ROOT` via `.project_root` marker, plus shared dirs (`T1_DIR`/`T2_DIR`/`OUTPUT_DIR` under `data/`) and `ensure_shared_dirs()` initializer
- `Controller/local_service.py` copies user-selected inputs into the shared dirs (session-id prefixed) and copies results back to the user output dir
- `start_app.py` calls `utils.paths.ensure_shared_dirs()` to initialize `data/t1`, `data/t2`, `data/output`
