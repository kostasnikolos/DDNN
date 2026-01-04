# Distributed Deep Neural Network (DDNN) with Intelligent Offloading

## Project Overview
This project implements a Distributed Deep Neural Network system for image classification with an intelligent offloading mechanism. The system consists of a local (edge) component and a cloud component, with an adaptive deep learning-based offloading mechanism to optimize computational resource usage and classification accuracy.

## Key Components

### Neural Network Architecture
- **Local Feature Extractor**: Performs initial feature extraction on the edge device
- **Local Classifier**: Makes preliminary classifications on the edge
- **Cloud CNN**: More complex model running on cloud infrastructure
- **Offload Mechanism**: Deep learning-based decision system for intelligent offloading

### Main Features
- **Multi-dataset Support**: CIFAR-10, CIFAR-100, CINIC-10, SVHN, GTSRB-32
- **Multiple Input Modes**: 'feat', 'logits', 'logits_plus', 'img'
- **Comprehensive Experiments**: Offload mechanism testing, timing analysis, overfitting detection, border/noisy sample analysis
- **Modular Architecture**: Clean separation of models, data loaders, training, and evaluation
- **Unified CLI**: Single script for all experiments with intuitive arguments

## File Structure
```
├── src/
│   ├── models/
│   │   ├── ddnn_models.py          # LocalFeatureExtractor, LocalClassifier, CloudCNN
│   │   └── offloading_model.py     # OffloadMechanism
│   ├── data/
│   │   └── data_loader.py          # Multi-dataset loaders
│   ├── utils/
│   │   └── utils.py                # Helper functions (BKS computation, model initialization)
│   ├── training.py                 # DDNN and offload mechanism training
│   └── evaluation.py               # 11 comprehensive test functions
├── scripts/
│   └── main.py                     # Unified experiment runner
├── models/                         # Saved model weights (.pth files)
├── plots/                          # Generated visualization outputs
├── data/                           # Dataset directory (auto-downloaded)
├── archive/                        # Legacy code (reference only)
└── README.md
```

## Requirements
```bash
# Core dependencies
Python >= 3.8
torch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.19.0
matplotlib >= 3.3.0
scipy >= 1.5.0

# Optional (for GPU acceleration)
CUDA >= 11.0
```

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib scipy
```

## Usage

### Quick Start
```bash
# Train DDNN and test offload mechanism on CIFAR-10
python scripts/main.py --mode train --dataset cifar10 --epochs_ddnn 50 --epochs_offload 30

# Run experiments with pretrained models (only if models already exist from same dataset/parameters)
python scripts/main.py --mode load --dataset cifar10 --testing_mode timing
```

### Command-Line Arguments

#### Main Arguments
- `--mode`: Execution mode (default: `train`)
  - `train`: Train DDNN from scratch, then run experiments
  - `load`: **Use only if** you have pretrained DDNN models (from a previous run with the **same dataset and parameters**, or downloaded from GitHub)
  
- `--dataset`: Dataset selection
  - `cifar10`: CIFAR-10 (10 classes, baseline)
  - `cifar100`: CIFAR-100 (100 classes, harder variant)
  - `cinic10`: CINIC-10 (CIFAR-10 + ImageNet mix)
  - `svhn`: Street View House Numbers
  - `gtsrb32`: German Traffic Sign Recognition (43 classes)

- `--testing_mode`: Type of experiment to run
  - `offload_mechanism` (default): Test offload decisions across multiple L0 values
  - `timing`: Inference time benchmarking
  - `border_noisy`: Misclassification analysis (border/noisy samples)
  - `overfitting`: Train/validation accuracy tracking

#### Training Hyperparameters
- `--epochs_ddnn`: DDNN training epochs (default: 50)
- `--epochs_offload`: Offload mechanism training epochs (default: 30)
- `--batch_size`: Batch size (default: 256)
- `--local_weight`: Local loss weight in DDNN training (default: 0.7)
- `--L0`: Target local percentage for single-L0 experiments (default: 0.54)

### Experiment Examples

#### 1. Offload Mechanism Performance (Multiple L0 Values)
```bash
python scripts/main.py \
  --mode train \
  --dataset cifar10 \
  --testing_mode offload_mechanism \
  --epochs_ddnn 50 \
  --epochs_offload 30 \
  --batch_size 128
```
**Output**: Plot showing DDNN accuracy vs local percentage for different methods

#### 2. Inference Timing Benchmark
```bash
python scripts/main.py \
  --mode train \
  --dataset cifar10 \
  --testing_mode timing \
  --epochs_ddnn 50 \
  --epochs_offload 20 \
  --batch_size 128 \
  --L0 0.54
```
**Output**: Timing comparison plot for each method

#### 3. Border/Noisy Sample Analysis
```bash
python scripts/main.py \
  --mode train \
  --dataset cifar10 \
  --testing_mode border_noisy \
  --epochs_ddnn 50 \
  --epochs_offload 50 \
  --batch_size 128 \
  --L0 0.54
```
**Output**: Test misclassification rates and training labeling quality analysis

#### 4. Overfitting Detection
```bash
python scripts/main.py \
  --mode train \
  --dataset cifar10 \
  --testing_mode overfitting \
  --epochs_ddnn 50 \
  --epochs_offload 50 \
  --batch_size 128 \
  --L0 0.54
```
**Output**: Train vs validation accuracy curves

#### 5. Multi-Dataset Training
```bash
# Train on CIFAR-100 (100 classes)
python scripts/main.py --mode train --dataset cifar100 --epochs_ddnn 50 --epochs_offload 30

# Train on GTSRB-32 (43 traffic sign classes)
python scripts/main.py --mode train --dataset gtsrb32 --epochs_ddnn 50 --epochs_offload 30
```

## Analysis Tools & Experiments

The system provides 4 comprehensive experiment modes:

### 1. **Offload Mechanism Testing**
- Tests multiple L0 values (0%, 10%, ..., 100% local processing)
- Compares different input representations (features, logits, logits+margin+entropy)
- Benchmarks against baselines (entropy, oracle, random)
- Generates accuracy vs local percentage plots

### 2. **Timing Analysis**
- Measures inference time for each method
- Runs multiple iterations for statistical reliability
- Outputs per-sample timing (milliseconds)
- Creates timing comparison visualizations

### 3. **Border/Noisy Sample Analysis**
- Identifies misclassified samples near decision boundaries
- Analyzes oracle labeling quality on training data
- Computes misclassification rates for different sample types
- Generates detailed analysis plots

### 4. **Overfitting Detection**
- Tracks train/validation accuracy across training epochs
- Compares different input modes (feat, logits)
- Helps identify optimal stopping points
- Visualizes learning curves

## Model Weights
Pre-trained models are saved in `models/` directory:
- `local_feature_extractor.pth` - Edge feature extraction
- `local_classifier.pth` - Edge classification head
- `cloud_cnn.pth` - Cloud classification network
- `offload_mechanism.pth` - Learned offload decisions
- `best_offload_mechanism.pth` - Best offload model (auto-saved during training)

## Results & Visualization
All plots are saved to `plots/` directory:
- `ddnn_overall_<dataset>_<modes>.png` - Overall accuracy plots
- `timing_benchmark_<dataset>_L0<value>.png` - Timing comparison
- `overfitting_<dataset>_L0<value>.png` - Train/val curves
- `test_misclassification_<dataset>_L0<value>.png` - Misclassification analysis
- `train_labeling_quality_<dataset>_L0<value>.png` - Labeling quality

## License
This project is part of a diploma thesis on Distributed Deep Neural Networks.

## Contact
For questions or collaborations, please open an issue on GitHub.
