# Multi-View Optimal Pose Skeleton-Based Action Recognition

A comprehensive framework for skeleton-based human action recognition using multi-view fusion techniques. This project implements and compares multiple approaches for processing 3D skeleton data from different viewpoints to improve action classification accuracy.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)

## 🎯 Overview

This project focuses on **skeleton-based action recognition** using **multiview fusion** techniques. The core innovation is the optimal pose fusion method that intelligently combines information from multiple camera views to improve recognition accuracy.

### Key Innovations
- **Multi-view Fusion**: Optimal pose selection across different viewpoints
- **Kalman Filtering**: Noise reduction and temporal smoothing
- **Multiple ML Models**: HMM, HCN, ST-GCN
- **Comprehensive Preprocessing**: Multiple normalization and alignment strategies

## ✨ Features

### 🎬 Supported Datasets
- **NTU RGB+D**: 60 action classes, cross-view/cross-subject evaluation
- **PKU-MMD**: 51 action classes, cross-view/cross-subject evaluation

### 🤖 Machine Learning Models
- **Hidden Markov Models (HMM)**: Sequential pattern recognition
- **Hierarchical Co-occurrence Network (HCN)**: CNN-based approach with motion features
- **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**: Graph-based spatial-temporal modeling

### 🔧 Multi-view Fusion Methods
- **OPT_POSE**: Optimal pose selection across views
- **OPT_POSE_KALMAN**: Kalman-filtered optimal pose fusion
- **MID_VIEW_ONLY**: Simple middle view selection
- **NONE**: Separate view processing

### 📊 Data Processing
- **Multiple Normalization Strategies**: Bone unit vectors, skeleton reference, joint differences
- **Coordinate System Transformations**: Various rotation and alignment methods
- **Missing Data Handling**: Interpolation and noise reduction
- **Multi-view Synchronization**: Temporal alignment across cameras

## 🚀 Installation

### Prerequisites
- Python >= 3.8
- CUDA (optional, for GPU acceleration)

### Dependencies
```bash
pip install -r requirements.txt
```

### Core Dependencies
- **PyTorch** >= 1.9.0
- **NumPy** >= 1.21.0
- **SciPy** >= 1.7.0
- **scikit-learn** >= 1.0.0
- **pomegranate** >= 0.14.0
- **PyTables** >= 3.6.0
- **matplotlib** >= 3.4.0
- **pandas** >= 1.3.0
- **dtaidistance** >= 2.3.0

## 📁 Datasets

### NTU RGB+D Dataset
1. Download from [NTU RGB+D website](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp)
2. Extract to `datasets/ntu_rgb_d/`
3. The framework will automatically process the data on first run

### PKU-MMD Dataset
1. Download from [PKU-MMD website](http://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html)
2. Extract to `datasets/pku_mmd/`
3. The framework will automatically process the data on first run

## 💻 Usage

### Command Line Interface

The project provides a comprehensive command-line interface for running experiments:

```bash
python main.py [OPTIONS]
```

#### Basic Examples

**Visualization with NTU dataset:**
```bash
python main.py --dataset ntu --benchmark cs --fusion both --method-type visualization --draw-type gif_single
```

**HMM classification with PKU dataset:**
```bash
python main.py --dataset pku --benchmark cs --fusion both --method hmm --method-type classification
```

**HCN training with custom parameters:**
```bash
python main.py --dataset ntu --benchmark cv --fusion both --method hcn --method-type classification --epochs 50
```

**Custom normalization and fusion:**
```bash
python main.py --dataset ntu --benchmark cs --fusion both --method hmm --norm-type NORM_BONE_UNIT_VEC --mv-fuse-type OPT_POSE
```

#### Available Arguments

| Argument | Type | Choices | Default | Description |
|----------|------|---------|---------|-------------|
| `--dataset` | str | ntu, pku | **required** | Dataset to use |
| `--benchmark` | str | cv, cs, all | cv | Benchmark type |
| `--fusion` | str | none, train_only, both | none | Multi-view fusion |
| `--method` | str | hmm, hcn, stgcn | hmm | ML method |
| `--method-type` | str | classification, visualization, validation | classification | Processing type |
| `--classes` | str | single, all | single | Number of classes |
| `--norm-type` | str | NORM_SKEL_REF, NORM_BONE_UNIT_VEC, etc. | NORM_SKEL_REF | Normalization type |
| `--mv-fuse-type` | str | NONE, OPT_POSE, etc. | NONE | Fusion type |
| `--draw-type` | str | gif_single, mv_seq_eq, etc. | none | Visualization type |
| `--epochs` | int | - | 100 | Training epochs |
| `--batch-size` | int | - | 64 | Batch size |
| `--hmm-n-comp` | int list | - | [10] | HMM components |

### Programmatic Usage

```python
from processes import process_common
from utils.multi_view_util import FuseType, NormType

# Run HMM classification
process_common(
    dataset='ntu',
    bm_train_type='cs',
    mv_fuse_data='both',
    method='hmm',
    method_type='classification',
    params_d={
        'norm_type': NormType.NORM_BONE_UNIT_VEC,
        'mv_fuse_type': FuseType.OPT_POSE,
        'classes': 'single',
        'hmm_n_comp': [10, 15, 20]
    }
)
```

## 🧠 Methods

### Multi-view Fusion Algorithm


## 📈 Results

### Performance Comparison


### Multi-view Fusion Impact


### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/multiview-optimal-pose.git
cd multiview-optimal-pose

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-party Licenses

- **HCN Model**: Adapted from [HCN-pytorch](https://github.com/huguyuehuhu/HCN-pytorch) (MIT License)
- **ST-GCN Model**: Adapted from [ST-GCN](https://github.com/yysijie/st-gcn) (BSD-2-Clause License)

## 📚 Citations

### Related Papers

- **HCN**: Li, C., et al. "Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation." IJCAI 2018.
- **ST-GCN**: Yan, S., et al. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition." AAAI 2018.

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.
