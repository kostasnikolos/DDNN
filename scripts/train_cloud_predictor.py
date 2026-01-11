"""
Train the Cloud Logit Predictor.

This script trains a lightweight model (`CloudLogitPredictor`) to predict the
logits that would be produced by the heavy CloudCNN, using only the local
feature extractor's output as input.

The purpose is to create a "cheap" proxy for the cloud's output, which can
then be used as an input feature for the final offload decision mechanism,
giving it a hint about the cloud's likely performance on a given sample.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data
from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN, CloudLogitPredictor
from src.models import LocalFeatureExtractor, CloudCNN, CloudLogitPredictor

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'cifar10'
BATCH_SIZE = 512
EPOCHS = 50
LR = 0.001
LOGITS_ONLY = False  # Set to True for simple FC-only architecture (local_logits → cloud_logits)
MODEL_SAVE_PATH = 'models/cloud_logit_predictor.pth' if not LOGITS_ONLY else 'models/cloud_logit_predictor_fc_only.pth'

def prepare_prediction_dataset(feature_extractor, local_classifier, cloud_cnn, dataloader, device):
    """
    Generates a dataset where X is (local_features, local_logits) and Y is cloud_logits.
    """
    feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    all_features = []
    all_local_logits = []
    all_cloud_logits = []
    print("Generating dataset for Cloud Logit Predictor (features + local_logits → cloud_logits)...")
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            local_features = feature_extractor(images)
            local_logits = local_classifier(local_features)
            cloud_logits = cloud_cnn(local_features)
            all_features.append(local_features.cpu())
            all_local_logits.append(local_logits.cpu())
            all_cloud_logits.append(cloud_logits.cpu())
    
    X_features = torch.cat(all_features)
    X_logits = torch.cat(all_local_logits)
    Y = torch.cat(all_cloud_logits)
    print(f"Dataset created. Features: {X_features.shape}, Local logits: {X_logits.shape}, Cloud logits: {Y.shape}")
    return TensorDataset(X_features, X_logits, Y)

def train_predictor(model, dataloader, epochs, lr, device, logits_only=False):
    """
    Trains the CloudLogitPredictor model.
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    mode_str = "FC-only (local_logits → cloud_logits)" if logits_only else "CNN+Logits (features + local_logits → cloud_logits)"
    print(f"\n--- Training Cloud Logit Predictor: {mode_str} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (features, local_logits, target_logits) in enumerate(dataloader):
            features = features.to(device)
            local_logits = local_logits.to(device)
            target_logits = target_logits.to(device)

            optimizer.zero_grad()
            if logits_only:
                predicted_logits = model(local_logits)  # Only local_logits
            else:
                predicted_logits = model(features, local_logits)  # Features + local_logits
            loss = criterion(predicted_logits, target_logits)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("--- Training Complete ---")

if __name__ == '__main__':
    # 1. Load pretrained DDNN models
    print("Loading pretrained DDNN models...")
    local_feature_extractor = LocalFeatureExtractor().to(DEVICE)
    local_classifier = LocalClassifier().to(DEVICE)
    cloud_cnn = CloudCNN().to(DEVICE)

    local_feature_extractor.load_state_dict(torch.load('models/local_feature_extractor.pth', map_location=DEVICE))
    local_classifier.load_state_dict(torch.load('models/local_classifier.pth', map_location=DEVICE))
    cloud_cnn.load_state_dict(torch.load('models/cloud_cnn.pth', map_location=DEVICE))
    print("✓ Models loaded.")

    # 2. Load data
    train_loader, _, _ = load_data(batch_size=BATCH_SIZE, dataset=DATASET_NAME)

    # 3. Create the special dataset for the predictor
    prediction_dataset = prepare_prediction_dataset(local_feature_extractor, local_classifier, cloud_cnn, train_loader, DEVICE)
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Initialize and train the predictor model
    cloud_logit_predictor = CloudLogitPredictor(latent_dim=256, logits_only=LOGITS_ONLY)
    train_predictor(cloud_logit_predictor, prediction_dataloader, EPOCHS, LR, DEVICE, logits_only=LOGITS_ONLY)

    # 5. Save the trained model
    print(f"Saving trained model to {MODEL_SAVE_PATH}")
    torch.save(cloud_logit_predictor.state_dict(), MODEL_SAVE_PATH)
    print("✓ Model saved.")
