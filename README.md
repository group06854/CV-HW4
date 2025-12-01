# 3D Gaussian Splatting Exploration Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful pipeline for exploring 3D Gaussian Splatting scenes with automatic object detection, camera path planning, and video generation.

## Features

- 🎯 **3DGS Scene Rendering** - Render high-quality scenes from 3D Gaussian Splatting models
- 🚀 **Intelligent Camera Path Planning** - Generate smooth camera trajectories with multiple algorithms
- 🔍 **Real-time Object Detection** - YOLOv8 integration with confidence filtering (>0.65)
- 🎬 **Video Generation** - Create exploration videos with annotated object detection
- 📊 **Detection Reporting** - JSON reports with object statistics and tracking
- ⚡ **GPU Acceleration** - Optimized rendering with CUDA support
- 🔧 **Modular Architecture** - Clean, maintainable codebase with separation of concerns

## Project Structure
src/

├── explorer.py # Camera path generation and smoothing algorithms

├── detector.py # YOLO-based object detection and tracking

├── path_planner.py # Camera path planning utilities

├── renderer.py # 3DGS rendering and video generation

└── main.py # Main pipeline and data loading

└── inputs/ # 3DGS PLY files (gitignored - use LFS)

└── outputs/ # Generated videos and reports

└──common_file.py # if src file do not work please use this file

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/3d-gaussian-explorer.git
cd 3d-gaussian-explorer
```
### 2. Install Dependencies
```bash
docker-compose up --build
docker-compose run --rm terminal python3 /app/common_file.py
or
docker-compose run --rm terminal python3 /app/src/main.py
```