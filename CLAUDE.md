# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSCD (遥感影像变化检测系统) V1.0 — a remote sensing change detection system using deep learning (X3D 3D-CNN). Identifies changes between temporal image/raster pairs for urban planning, environmental monitoring, and disaster assessment.

## Architecture

Three-tier client-server with containerized backend:

```
Frontend (PySide6/Qt6)  ──HTTP──▶  Backend API (FastAPI, Docker)  ──▶  X3D Model (PyTorch/CUDA)
zhuyaogongneng_docker/              change3d_api_docker/                change3d_docker/
```

**Data flow**: Frontend copies images to shared dirs (`t1/`, `t2/`) → POST to backend API → async task runs X3D inference → results written to `output/` → frontend polls and displays.

**Four processing modes**: `single_image`, `single_raster`, `batch_image`, `batch_raster`. Raster modes use GDAL/GeoPandas and export GeoTIFF + vector files.

## Key Modules

| Directory | Purpose |
|---|---|
| `zhuyaogongneng_docker/app.py` | Main PySide6 application (RemoteSensingApp, HomePage classes) |
| `zhuyaogongneng_docker/function/` | Feature modules: API client, image import, batch processing, raster handling |
| `change3d_api_docker/main.py` | FastAPI application with 4 detection endpoints + task management |
| `change3d_api_docker/change_detection_model.py` | Model wrapper (ChangeDetectionModel class) |
| `change3d_docker/model/x3d.py` | X3D architecture (modified from Facebook PyTorchVideo) |
| `change3d_docker/model/trainer.py` | Encoder-Decoder with X3D backbone |
| `change3d_docker/model/change_decoder.py` | Change detection decoder head |
| `change3d_docker/scripts_app/` | 4 processing scripts (image/raster × single/batch) |

## Commands

### Backend (Docker)

```bash
# Build and start the API service
cd change3d_api_docker
docker-compose -f docker-compose.optimized.yml build --no-cache
docker-compose -f docker-compose.optimized.yml up -d

# Stop
docker-compose -f docker-compose.optimized.yml down

# View logs
docker logs -f change3d-api-optimized
```

### Frontend

```bash
# Install dependencies (uses Conda environment)
pip install -r requirements.txt

# Launch the desktop application
python start_app.py
```

### Testing

```bash
# Run API integration tests (no pytest framework, custom test runner)
cd change3d_api_docker
python api_test_script.py
```

Test config: `change3d_api_docker/test_config.json` (API base URL, test data paths, which test cases to run).

### API

- Swagger docs: `http://localhost:8000/docs`
- Endpoints: `/health`, `/detect/single_image`, `/detect/single_raster`, `/detect/batch_image`, `/detect/batch_raster`, `/tasks/{task_id}`, `/tasks`

## Tech Stack

- **AI**: PyTorch 2.8, CUDA 12.6, PyTorchVideo, einops, albumentations
- **GIS**: GDAL, Rasterio, GeoPandas, Fiona, Shapely
- **Frontend**: PySide6 (Qt6), OpenCV, NumPy, SciPy
- **Backend**: FastAPI, Uvicorn, aiofiles
- **Infra**: Docker, Docker Compose, NVIDIA Container Toolkit

## Codebase Conventions

- Primary language: Python 3.10+, code comments and UI in Chinese
- Conda for environment management (not pip venv)
- No formal linting or CI/CD configured
- No unit test framework — testing is integration/manual via `api_test_script.py`
- Model checkpoint at `change3d_docker/checkpoint/X3D_L.pyth` (~50MB)
- Docker volumes mount shared data dirs: `t1/` (before images), `t2/` (after images), `output/` (results)
- `.project_root` file marks repository root for path resolution
