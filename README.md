# High-Performance CUDA YOLO Pre-Processing Pipeline

This project implements a high-throughput, GPU-accelerated image pre-processing pipeline designed specifically for YOLO object detection architectures.

In computer vision deployments, neural network inference is often bottlenecked by CPU-bound data ingestion and formatting. This project solves that bottleneck by leveraging **Asynchronous CUDA Streams** and **Custom CUDA Kernels** to prepare raw image data for deep learning models at scale.

---

## Features

* **Custom CUDA Kernel:** Performs simultaneous:

  * Color-space conversion (RGB standard → RGB planar)
  * Interleaved-to-Planar memory transposition (HWC → CHW)
  * Mathematical normalization (0–255 integers → 0.0–1.0 float32)

* **Multi-Stream Architecture:**

  * Overlaps CPU-to-GPU memory transfers (`cudaMemcpyAsync`) with GPU computation
  * Maximizes hardware utilization

* **Page-Locked Memory:**

  * Uses `cudaMallocHost` for pinned host memory
  * Enables maximum PCIe transfer bandwidth

* **Dependency-Free Image Loading:**

  * Uses lightweight `stb_image.h`
  * Avoids heavy dependencies like OpenCV

* **Auto-Scaling:**

  * Dynamically scans target directory
  * Processes images in optimized batch sizes

---

## Prerequisites

* NVIDIA GPU with CUDA support
* NVIDIA CUDA Toolkit (`nvcc` compiler)
* Standard C++14 compatible compiler (e.g., GCC)

---

## Download Dependency

To download the required `stb_image.h` library, run:

```bash
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
```

---

## How to Execute

Before running the pipeline, ensure your `.jpg` images are placed inside the `data/` folder.

### Method 1: Automated Bash Script (Recommended)

```bash
# Give execution permission (only once)
chmod +x run.sh

# Run the pipeline
./run.sh
```

---

### Method 2: Using Makefile

```bash
# Compile
make

# Run
./yolo_pipeline

# Optional cleanup
make clean
```

---

### Method 3: Direct NVCC Compilation

```bash
# Compile with optimization
nvcc -O3 -std=c++14 -o yolo_pipeline main.cu

# Run
./yolo_pipeline
```

---

## Understanding the Output

Standard images are stored in compressed formats and decoded into an **interleaved memory layout (HWC)** with integer values.

YOLO models require a tensor-based format. This pipeline converts images into:

### Output Specifications

* **Format:** Uncompressed 32-bit floating point (`float32`)
* **Memory Layout:** Planar / Channel-First (`CHW`)

  * All Red values
  * Followed by Green
  * Followed by Blue
* **Normalization:** Pixel values scaled to `[0.0, 1.0]`
* **Dimensions:** `3 × 640 × 640`
* **File Size:** `4,915,200 bytes` per file

---

## Final Output

The processed `.bin` files are saved in:

```text
data/output/
```

These tensors are ready for:

* Direct GPU loading
* TensorRT inference pipelines
* PyTorch dataloaders

This enables **zero-overhead, high-throughput inference workflows**.

---

## Directory Structure

Ensure your project directory looks exactly like this before compiling:

```text
~/project/
├── data/                  <-- Drop your raw .jpg/.jpeg images in here
│   └── output/            <-- (Auto-generated) Tensor .bin files are saved here
├── main.cu                <-- Main C++/CUDA source code
├── yolo_pipeline.h        <-- Pipeline configuration header
├── stb_image.h            <-- Single-file image loader dependency
├── Makefile               <-- Compilation instructions
├── run.sh                 <-- Execution script
└── README.md              <-- This documentation
```
