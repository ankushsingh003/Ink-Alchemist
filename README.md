# Ink-Alchemist: 3D Ink Detection & Virtual Unrolling

![Vesuvius](https://raw.githubusercontent.com/Ankushsingh003/Ink-Alchemist/main/image_for_eval/ir.png)

## 📜 Overview
Ink-Alchemist is an industry-level AI pipeline designed for the **Vesuvius Challenge**. It implements a non-destructive restoration workflow to detect ancient Greek ink on carbonized papyrus scrolls using high-resolution 3D X-ray CT scans.

Our goal is to turn prehistoric "ash" into digital "ink" through state-of-the-art computer vision and volumetric data engineering.

---

## 🛠️ Technical Architecture

### 1. Data Engineering (Phase 1)
- **Lazy Loading**: Utilizes memory-mapped NumPy arrays to handle massive 3D volumes without RAM bottlenecks.
- **Surface Extraction**: Targets the middle 16-32 slices where the carbon signal is mathematically strongest.
- **Dynamic Tiling**: On-the-fly generation of overlapping tiles for high-fidelity training.

### 2. Hybrid Model (Phase 2)
- **3D-ResNet Backbone**: Extracts textural and volumetric features from the Z-axis (depth).
- **Transformer Encoder**: Captures global semantic context to recognize character shapes and word patterns.
- **Head**: 2D Semantic Segmentation head for high-resolution mask prediction.

### 3. Integrated MLOps (Phase 3)
- **Experiment Tracking**: Integrated with **Weights & Biases (WandB)** for real-time monitoring.
- **Post-Processing**: Threshold sweeping and Morphological Denoising to maximize the F0.5/Dice score.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (CUDA recommended)
- `pip install -r requirements.txt` (or manually install `wandb`, `opencv-python`, `torch`, `tqdm`)

### Execution Flow
1. **Download & Organize**:
   ```powershell
   cd process_fragments
   ./download_fragment_1.ps1
   ./unzip_fragment_1.ps1
   ```
2. **Preprocess**:
   ```bash
   python data_preprocessing.py --fragment 1
   ```
3. **Train**:
   ```bash
   python train.py
   ```
4. **Evaluate**:
   ```bash
   python evaluation.py
   ```

---

## 🏗️ Project Structure
- `data_preprocessing.py`: Volume compression and tiling system.
- `model.py`: Hybrid 3D-CNN + Transformer architecture.
- `train.py`: Training loop with WandB integration.
- `evaluation.py`: Post-processing and threshold optimization.
- `process_fragments/`: Data acquisition scripts.
- `image_for_eval/`: Metadata visualization.

---

## 🌟 Industry Impact
This project replaces physical archaeology with **Virtual Unrolling**, enabling the reading of thousands of lost scrolls without risking physical damage. The techniques developed here are directly applicable to Medical Imaging (CT/MRI) and Material Science (Non-destructive testing).
