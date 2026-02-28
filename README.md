# InkSense: 3D Ink Detection & Virtual Unrolling

![Vesuvius](https://raw.githubusercontent.com/Ankushsingh003/Ink-Alchemist/main/image_for_eval/ir.png)

## 📜 Overview
InkSense is an industry-level AI pipeline designed for the **Vesuvius Challenge**. It implements a non-destructive restoration workflow to detect ancient Greek ink on carbonized papyrus scrolls using high-resolution 3D X-ray CT scans.

Our goal is to turn prehistoric "ash" into digital "ink" through state-of-the-art computer vision and volumetric data engineering.

---

## 🌐 Web Interface
A premium web-based ink detection analyzer is available in the `ink-alchemist-web/` directory. Upload any image and get a full ink coverage report, heatmap, and downloadable mask.

```bash
# Start the web server
python serve_app.py
# Open http://localhost:3000
```

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

## 🏆 Industrial Impact & "Problem-Solver" Traits
*This project is built to showcase technical resilience beyond academics:*

1.  **Resilience to Noisy Data**: Our hybrid architecture is specifically designed to bypass CT artifacts and physical "papyrus noise," focusing only on the chemical signature of the ink.
2.  **Efficient Resource Management**: InkSense handles 20GB+ datasets on consumer-grade hardware through memory mapping and high-performance tiling, demonstrating scalable engineering.
3.  **Domain Adaptation**: The "Virtual Unrolling" techniques implemented here are pivots for **Medical AI** (segmenting micro-tumors) and **Aerospace Inspection** (identifying structural cracks in 3D-scanned parts).

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
5. **Web UI**:
   ```bash
   python serve_app.py
   ```

---

## 🏗️ Project Structure
- `data_preprocessing.py`: Volume compression and tiling system.
- `model.py`: Hybrid 3D-CNN + Transformer architecture.
- `train.py`: Training loop with WandB integration.
- `evaluation.py`: Post-processing and threshold optimization.
- `serve_app.py`: Web server for the InkSense UI.
- `ink-alchemist-web/`: Frontend Vite web application.
- `process_fragments/`: Data acquisition scripts.
- `image_for_eval/`: Metadata visualization.

---

## 🌟 Industry Impact
This project replaces physical archaeology with **Virtual Unrolling**, enabling the reading of thousands of lost scrolls without risking physical damage. The techniques developed here are directly applicable to Medical Imaging (CT/MRI) and Material Science (Non-destructive testing).
