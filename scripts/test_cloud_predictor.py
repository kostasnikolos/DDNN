"""
Test the Cloud Logit Predictor performance.

This script evaluates how well the CloudLogitPredictor approximates the
actual CloudCNN outputs. We measure:
- MSE (Mean Squared Error) between predicted and actual logits
- Classification accuracy when using predicted logits
- Top-1 and Top-5 agreement rates
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data
from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN, CloudLogitPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'cifar10'
BATCH_SIZE = 256
LOGITS_ONLY = False  # Set to True to test FC-only predictor

def evaluate_predictor(feature_extractor, local_classifier, cloud_cnn, predictor, dataloader, device):
    """
    Evaluate the CloudLogitPredictor against the actual CloudCNN.
    """
    feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    predictor.eval()
    
    total_samples = 0
    total_mse = 0.0
    
    # Classification metrics
    cloud_correct = 0
    predicted_correct = 0
    top1_agreement = 0
    top5_agreement = 0
    full_ranking_agreement = 0
    top3_order_agreement = 0
    
    print("\nEvaluating CloudLogitPredictor...")
    print("-" * 70)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Get local features and logits
            local_features = feature_extractor(images)
            local_logits = local_classifier(local_features)
            
            # Actual cloud logits
            actual_cloud_logits = cloud_cnn(local_features)
            
            # Predicted cloud logits (using features + local_logits OR just logits)
            if hasattr(predictor, 'logits_only') and predictor.logits_only:
                predicted_cloud_logits = predictor(local_logits)  # FC-only mode
            else:
                predicted_cloud_logits = predictor(local_features, local_logits)  # CNN+logits mode
            
            # MSE between predicted and actual logits
            mse = F.mse_loss(predicted_cloud_logits, actual_cloud_logits, reduction='sum')
            total_mse += mse.item()
            
            # Classification accuracy
            actual_preds = actual_cloud_logits.argmax(dim=1)
            predicted_preds = predicted_cloud_logits.argmax(dim=1)
            
            cloud_correct += (actual_preds == labels).sum().item()
            predicted_correct += (predicted_preds == labels).sum().item()
            
            # Top-1 agreement (do both predict the same class?)
            top1_agreement += (actual_preds == predicted_preds).sum().item()
            
            # Top-5 agreement (is actual top-1 in predicted top-5?)
            _, predicted_top5 = predicted_cloud_logits.topk(5, dim=1)
            top5_match = (predicted_top5 == actual_preds.unsqueeze(1)).any(dim=1)
            top5_agreement += top5_match.sum().item()
            
            # Full ranking agreement (all 10 classes in same order)
            actual_ranking = actual_cloud_logits.argsort(dim=1, descending=True)
            predicted_ranking = predicted_cloud_logits.argsort(dim=1, descending=True)
            full_match = (actual_ranking == predicted_ranking).all(dim=1)
            full_ranking_agreement += full_match.sum().item()
            
            # Top-3 order agreement (top 3 classes in same order)
            top3_match = (actual_ranking[:, :3] == predicted_ranking[:, :3]).all(dim=1)
            top3_order_agreement += top3_match.sum().item()
            
            total_samples += images.size(0)
    
    # Calculate metrics
    avg_mse = total_mse / total_samples
    cloud_acc = 100.0 * cloud_correct / total_samples
    predicted_acc = 100.0 * predicted_correct / total_samples
    top1_agree = 100.0 * top1_agreement / total_samples
    top5_agree = 100.0 * top5_agreement / total_samples
    full_ranking_agree = 100.0 * full_ranking_agreement / total_samples
    top3_order_agree = 100.0 * top3_order_agreement / total_samples
    
    print(f"Total samples evaluated: {total_samples}")
    print(f"\n{'Metric':<40} {'Value':>15}")
    print("=" * 70)
    print(f"{'Actual Cloud CNN Accuracy:':<40} {cloud_acc:>14.2f}%")
    print(f"{'Predictor Overall Success Rate:':<40} {top1_agree:>14.2f}%")
    print(f"{'Top-3 Order Agreement:':<40} {top3_order_agree:>14.2f}%")
    print(f"{'Top-5 Agreement Rate:':<40} {top5_agree:>14.2f}%")
    print(f"{'Full Ranking Agreement (all 10):':<40} {full_ranking_agree:>14.2f}%")
    print(f"{'MSE (Logits):':<40} {avg_mse:>15.6f}")
    print("=" * 70)
    
    print("\n💡 Interpretation:")
    print(f"   - Top-1 Agreement = {top1_agree:.2f}%: Predictor matches cloud's top prediction")
    print(f"   - Top-3 Order = {top3_order_agree:.2f}%: Top-3 classes in exact same order")
    print(f"   - Full Ranking = {full_ranking_agree:.2f}%: All 10 classes in exact same order")
    print(f"   - MSE = {avg_mse:.4f}: Average squared error between all 10 logits")
    
    # Decision criterion
    print("\n⚖️  Assessment:")
    if top1_agree >= 85:
        print("   ✅ EXCELLENT: Predictor closely mimics cloud behavior")
    elif top1_agree >= 75:
        print("   ✓ GOOD: Predictor provides useful approximation")
    elif top1_agree >= 65:
        print("   ⚠️  FAIR: Predictor may help but limited accuracy")
    else:
        print("   ❌ POOR: Predictor is too inaccurate to be useful")
    
    return {
        'mse': avg_mse,
        'cloud_acc': cloud_acc,
        'predicted_acc': predicted_acc,
        'top1_agreement': top1_agree,
        'top5_agreement': top5_agree,
        'top3_order_agreement': top3_order_agree,
        'full_ranking_agreement': full_ranking_agree
    }

if __name__ == '__main__':
    mode_str = "FC-only" if LOGITS_ONLY else "CNN+Logits"
    print("="*70)
    print(f"CLOUD LOGIT PREDICTOR EVALUATION ({mode_str} mode)")
    print("="*70)
    
    # Load models
    print("\n[1/4] Loading models...")
    local_feature_extractor = LocalFeatureExtractor().to(DEVICE)
    local_classifier = LocalClassifier().to(DEVICE)
    cloud_cnn = CloudCNN().to(DEVICE)
    cloud_logit_predictor = CloudLogitPredictor(latent_dim=256, logits_only=LOGITS_ONLY).to(DEVICE)
    
    local_feature_extractor.load_state_dict(torch.load('models/local_feature_extractor.pth', map_location=DEVICE))
    local_classifier.load_state_dict(torch.load('models/local_classifier.pth', map_location=DEVICE))
    cloud_cnn.load_state_dict(torch.load('models/cloud_cnn.pth', map_location=DEVICE))
    
    # Load appropriate predictor model
    predictor_path = 'models/cloud_logit_predictor_fc_only.pth' if LOGITS_ONLY else 'models/cloud_logit_predictor.pth'
    cloud_logit_predictor.load_state_dict(torch.load(predictor_path, map_location=DEVICE))
    print(f"   ✓ All models loaded (predictor: {predictor_path})")
    
    # Load test data
    print("\n[2/4] Loading test data...")
    _, _, test_loader = load_data(batch_size=BATCH_SIZE, dataset=DATASET_NAME)
    print(f"   ✓ Test dataset ready")
    
    # Evaluate
    print("\n[3/4] Running evaluation...")
    results = evaluate_predictor(
        local_feature_extractor,
        local_classifier,
        cloud_cnn,
        cloud_logit_predictor,
        test_loader,
        DEVICE
    )
    
    print("\n[4/4] Evaluation complete!")
    print("\n" + "="*70)
