# Distributed Deep Neural Network (DDNN) with Intelligent Offloading

## Project Overview
This project implements a Distributed Deep Neural Network system for CIFAR-10 image classification with an intelligent offloading mechanism. The system consists of a local (edge) component and a cloud component, with an adaptive decision mechanism to optimize computational resource usage and classification accuracy.

## Key Components

### Neural Network Architecture
- **Local Feature Extractor**: Performs initial feature extraction on the edge device
- **Local Classifier**: Makes preliminary classifications on the edge
- **Cloud CNN**: More complex model running on cloud infrastructure
- **Offload Mechanism**: Deep learning-based decision system for intelligent offloading

### Main Features
- Adaptive computation offloading
- Entropy-based decision making
- Multiple input modes support ('feat', 'logits', 'logits_plus')
- Comprehensive evaluation and analysis tools

## File Structure
- `latest_deep_offload.py`: Main implementation file
- `testing_architecture_entropy.py`: Architecture testing with entropy-based decisions
- `Cifar10_local_features.py`: Local feature extraction implementation
- Model weights:
  - `local_feature_extractor.pth`
  - `local_classifier.pth`
  - `cloud_cnn.pth`
  - `offload_mechanism.pth`

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- CUDA (optional, for GPU acceleration)

## Usage

### Training the DDNN
```python
# Initialize and train the DDNN
python latest_deep_offload.py --epochs_DDNN 30 --batch_size 256 --L0 0.54 --local_weight 0.7
```

### Parameters
- `epochs_DDNN`: Number of epochs for DDNN training
- `epochs_optimization`: Number of epochs for optimization training
- `batch_size`: Batch size for training
- `L0`: Target local computation percentage
- `local_weight`: Weight for local classification loss

## Results
The system includes various analysis tools and visualization capabilities:
- Performance metrics across different offloading strategies
- Entropy-based decision analysis
- Resource utilization statistics
- Accuracy comparisons between local and cloud processing

## Model Weights
Pre-trained model weights are included in the repository:
- Local feature extractor
- Local classifier
- Cloud CNN
- Offload mechanism

## Analysis Tools
- Correlation metrics analysis
- Decision boundary visualization
- Performance profiling
- Resource usage monitoring

## License
[Your chosen license]

## Acknowledgments
[Your acknowledgments]

## Contact
[Your contact information] 