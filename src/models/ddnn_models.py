"""
DDNN Model Architectures
========================

Neural network models for the Distributed Deep Neural Network (DDNN) system:
- LocalFeatureExtractor: Edge device feature extraction (Conv2D → BN → LeakyReLU → Pool)
- LocalClassifier: Edge device classifier (flattens features → FC layers)
- CloudCNN: Cloud-side deep CNN (additional Conv blocks + FC layers)

These models form the two-tier DDNN architecture where inference can be split
between local (edge) and cloud execution based on the offload decision mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFeatureExtractor(nn.Module):
    """
    Local CNN Feature Extractor for edge device.
    
    Creates local features from input images using a basic CNN block:
    Conv2D → Batch Normalization → Leaky ReLU → MaxPool → Dropout
    
    Architecture:
    - Input: (batch, 3, 32, 32) RGB images
    - Conv2D: 3 → 32 channels, 3x3 kernel
    - BatchNorm2d
    - LeakyReLU (negative_slope=0.01)
    - MaxPool2d: 2x2, stride=2
    - Dropout: p=0.2
    - Output: (batch, 32, 16, 16) feature maps
    
    The output shape after pooling is (32, 16, 16) for CIFAR-10/CIFAR-100 inputs.
    These features serve as input to both LocalClassifier (edge) and CloudCNN (cloud).
    """
    
    def __init__(self):
        super(LocalFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Forward pass through local feature extractor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 3, 32, 32)
        
        Returns
        -------
        torch.Tensor
            Feature maps of shape (batch, 32, 16, 16)
        """
        x = F.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.01)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class LocalClassifier(nn.Module):
    """
    Local Classifier for edge device.
    
    Acts as the local classifier on top of LocalFeatureExtractor output.
    Flattens the feature map and uses FC layers to classify the image.
    
    Architecture:
    - Input: (batch, 32, 16, 16) feature maps from LocalFeatureExtractor
    - Flatten: (batch, 32*16*16) = (batch, 8192)
    - FC1: 8192 → 64 with LeakyReLU
    - FC2: 64 → NUM_CLASSES (logits)
    - Output: (batch, NUM_CLASSES) unnormalized logits
    
    Note: NUM_CLASSES must be defined globally or imported before instantiation
    """
    
    def __init__(self,num_classes=10):
        super(LocalClassifier, self).__init__()
        # Fully connected layer to classify based on the local features
        self.fc1 = nn.Linear(32 * 16 * 16, 64)  # 32 channels, 32x32 feature maps, 10 output classes
        
        # Import NUM_CLASSES from global scope (set in main script)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Forward pass through local classifier.
        
        Parameters
        ----------
        x : torch.Tensor
            Feature maps of shape (batch, 32, 16, 16)
        
        Returns
        -------
        torch.Tensor
            Class logits of shape (batch, NUM_CLASSES)
        """
        x = x.view(-1, 32 * 16 * 16)  # Flatten the feature map before feeding into fully connected layer
        
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc2(x)
        return x


class CloudCNN(nn.Module):
    """
    Cloud CNN for server-side deep processing.
    
    Takes local features from LocalFeatureExtractor as input and processes them
    with additional convolutional layers for deeper classification. This model
    is more computationally expensive and runs on the cloud server.
    
    Architecture:
    - Input: (batch, 32, 16, 16) feature maps from LocalFeatureExtractor
    - Conv Block 1: 32 → 64 channels, BN, LeakyReLU, Dropout, Pool (→ 8x8)
    - Conv Block 2: 64 → 128 channels, BN, LeakyReLU, Dropout, Pool (→ 4x4)
    - Conv Block 3: 128 → 256 channels, BN, LeakyReLU, Dropout (no pool)
    - Conv Block 4: 256 → 256 channels, BN, LeakyReLU, Dropout, Pool (→ 2x2)
    - Flatten: (batch, 256*2*2) = (batch, 1024)
    - FC1: 1024 → 256 with LeakyReLU
    - FC2: 256 → 64 with LeakyReLU
    - FC3: 64 → NUM_CLASSES (logits)
    - Output: (batch, NUM_CLASSES) unnormalized logits
    
    The Cloud CNN typically achieves higher accuracy than the Local Classifier
    due to its deeper architecture, at the cost of higher computational complexity.
    
    Note: NUM_CLASSES must be defined globally or imported before instantiation
    """
    
    def __init__(self,num_classes=10):
        super(CloudCNN, self).__init__()
        
        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(p=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        
        # Import NUM_CLASSES from global scope (set in main script)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Forward pass through cloud CNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Feature maps of shape (batch, 32, 16, 16) from LocalFeatureExtractor
        
        ReturnsNUM_CLASSES
        -------
        torch.Tensor
            Class logits of shape (batch, num_classes)
        """
        # Conv block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = self.pool1(x)  # 16x16 → 8x8
        
        # Conv block 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = self.pool2(x)  # 8x8 → 4x4
        
        # Conv block 3
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        
        # Conv block 4
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = self.dropout4(x)
        # print(f" print input shpate before flattening: {x.shape}")  # Debug p
        x = self.pool3(x)  # 4x4 → 2x2
        
        # Flatten and FC layers
        x = x.view(-1, 256 * 2 * 2)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.fc3(x)
        
        return x


class CloudLogitPredictor(nn.Module):
    """
    A lightweight model to predict cloud logits from local feature maps + local logits.

    This model takes the feature maps from the LocalFeatureExtractor AND the local
    classifier's output logits to approximate the CloudCNN's output.
    The local logits provide valuable context about what the local model "sees".

    Architecture:
    - If logits_only=False (default): CNN encoder + local logits → cloud logits
    - If logits_only=True: Simple FC layers (local logits → cloud logits)
    """
    def __init__(self, input_channels=32, num_classes=10, latent_dim=256, logits_only=False):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.logits_only = logits_only
        
        if self.logits_only:
            # Simple FC-only architecture: local_logits → cloud_logits
            self.fc_layers = nn.Sequential(
                nn.Linear(num_classes, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        else:
            # CNN encoder for features
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            
            self.conv2 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1)  # 16×16 → 8×8
            self.bn2 = nn.BatchNorm2d(latent_dim)
            
            # Global Average Pooling
            self.gap = nn.AdaptiveAvgPool2d(1)
            
            # Final FC: feature_vec (latent_dim) + local_logits (num_classes) → cloud_logits
            self.fc = nn.Linear(latent_dim + num_classes, num_classes)

    def forward(self, local_features, local_logits=None):
        """
        Predict cloud logits from local features + local logits (or just logits).
        
        Args:
            local_features: (B, 32, 16, 16) from LocalFeatureExtractor (if logits_only=False)
                           OR (B, num_classes) local logits (if logits_only=True)
            local_logits: (B, num_classes) from LocalClassifier (only needed if logits_only=False)
        
        Returns:
            (B, num_classes) predicted cloud logits
        """
        if self.logits_only:
            # Simple FC path: local_logits → cloud_logits
            return self.fc_layers(local_features)  # local_features is actually local_logits here
        else:
            # CNN + logits path
            # Encode features
            x = F.relu(self.bn1(self.conv1(local_features)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.gap(x).flatten(1)  # (B, latent_dim)
            
            # Concatenate with local logits
            x = torch.cat([x, local_logits], dim=1)  # (B, latent_dim + num_classes)
            
            return self.fc(x)  # (B, num_classes)
