"""
Script to analyze the Oracle vs Optimized Rule performance gap.

This script helps explain why Oracle achieves significantly higher accuracy
than the Optimized Rule by categorizing samples into:
  - Disagreement samples (noisy labels)
  - Borderline samples (ambiguous cases)
  - Rational failures (coin toss scenarios)
  - Normal samples

It also analyzes how different borderline thresholds affect performance.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data
from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN, OffloadMechanism, CloudLogitPredictor
from src.utils import compute_bks_input_for_deep_offload, calculate_b_star, create_3d_data_deep
from src.models import OffloadDatasetCNN
from src.training import train_deep_offload_mechanism
from src.evaluation import (
    analyze_oracle_optimized_gap, 
    analyze_borderline_threshold_sensitivity
)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CONFIGURATION
# ============================================================================
MODE = 'load'  # 'train' to train DDNN from scratch, 'load' to load pretrained models
DATASET_NAME = 'cifar10'
BATCH_SIZE = 256
L0 = 0.54  # Target local percentage
TAU_BORDER = 0.01  # Borderline threshold
INPUT_MODE = 'logits_with_bk_pred'  # Using CloudLogitPredictor (86% agreement)
OFFLOAD_EPOCHS = 10  # Increased for better convergence
DDNN_EPOCHS = 50  # Only used if MODE='train'
LOCAL_WEIGHT = 0.7  # Weight for local loss in DDNN training
NUM_CLASSES = 10
LOGITS_ONLY_PREDICTOR = False  # Set to True to use FC-only CloudLogitPredictor
# ★ Soft Labels Configuration (disabled when using regression)
USE_SOFT_LABELS = False  # Disable - we're using regression now
SOFT_LABEL_TEMPERATURE = 3.0

# ★ Regression Mode Configuration
USE_REGRESSION_MODE = False  # Back to classification for hybrid mode

# ★ NEW: Hybrid Mode Configuration
FEAT_LATENT_DIM = 32  # Compact latent dimension


if __name__ == '__main__':
    print("="*80)
    print("ORACLE vs OPTIMIZED RULE GAP ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Mode: {MODE}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  L0: {L0}")
    print(f"  Borderline threshold (τ): {TAU_BORDER}")
    print(f"  Input mode: {INPUT_MODE}")
    print(f"  Offload training epochs: {OFFLOAD_EPOCHS}")
    if INPUT_MODE == 'hybrid':
        print(f"  ★ Hybrid Mode: feat_latent_dim={FEAT_LATENT_DIM}")
    print(f"  ★ Regression Mode: {USE_REGRESSION_MODE}")
    if not USE_REGRESSION_MODE and USE_SOFT_LABELS:
        print(f"  ★ Soft Labels: {USE_SOFT_LABELS} (temperature={SOFT_LABEL_TEMPERATURE})")
    if MODE == 'train':
        print(f"  DDNN training epochs: {DDNN_EPOCHS}")
        print(f"  Local weight: {LOCAL_WEIGHT}")

    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    train_loader, val_loader, test_loader = load_data(BATCH_SIZE, dataset=DATASET_NAME)

    # ============================================================================
    # LOAD TRAINED MODELS
    # ============================================================================
    print(f"\n{'='*80}")
    if MODE == 'train':
        print("TRAINING DDNN MODELS")
    else:
        print("LOADING PRETRAINED MODELS")
    print(f"{'='*80}")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Initialize models
    local_feature_extractor = LocalFeatureExtractor().to(device)
    local_classifier = LocalClassifier(num_classes=NUM_CLASSES).to(device)
    cloud_cnn = CloudCNN(num_classes=NUM_CLASSES).to(device)

    if MODE == 'train':
        # Train DDNN from scratch
        from src.training import train_DDNN
        from src.utils import initialize_optimizers
        
        print("\nTraining DDNN models...")
        
        # Create dummy offload mechanism for initialize_optimizers
        dummy_offload = OffloadMechanism(input_mode='logits_plus', num_classes=NUM_CLASSES).to(device)
        
        cnn_optimizer, _ = initialize_optimizers(
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            dummy_offload
        )
        
        train_DDNN(
            train_loader,
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            cnn_optimizer,
            LOCAL_WEIGHT,
            DDNN_EPOCHS
        )
        
        # Save models
        torch.save(local_feature_extractor.state_dict(),
                   os.path.join(models_dir, f"local_feature_extractor_{DATASET_NAME}.pth"))
        torch.save(local_classifier.state_dict(),
                   os.path.join(models_dir, f"local_classifier_{DATASET_NAME}.pth"))
        torch.save(cloud_cnn.state_dict(),
                   os.path.join(models_dir, f"cloud_cnn_{DATASET_NAME}.pth"))
        
        print(f"✓ DDNN models trained and saved to {models_dir}/")
        
    else:  # MODE == 'load'
        # Load pretrained models
        local_feature_extractor.load_state_dict(
            torch.load(os.path.join(models_dir, f"local_feature_extractor_{DATASET_NAME}.pth"),
                       map_location=device)
        )
        local_classifier.load_state_dict(
            torch.load(os.path.join(models_dir, f"local_classifier_{DATASET_NAME}.pth"),
                       map_location=device)
        )
        cloud_cnn.load_state_dict(
            torch.load(os.path.join(models_dir, f"cloud_cnn_{DATASET_NAME}.pth"),
                       map_location=device)
        )
        
        print("✓ DDNN models loaded successfully")

    # Load CloudLogitPredictor if using logits_with_bk_pred mode
    cloud_predictor = None
    if INPUT_MODE == 'logits_with_bk_pred':
        print("\n[Special] Loading CloudLogitPredictor for bk prediction...")
        cloud_predictor = CloudLogitPredictor(num_classes=NUM_CLASSES, logits_only=LOGITS_ONLY_PREDICTOR).to(device)
        predictor_path = os.path.join(models_dir, "cloud_logit_predictor_fc_only.pth" if LOGITS_ONLY_PREDICTOR else "cloud_logit_predictor.pth")
        cloud_predictor.load_state_dict(
            torch.load(predictor_path, map_location=device)
        )
        cloud_predictor.eval()
        mode_str = "FC-only" if LOGITS_ONLY_PREDICTOR else "CNN+Logits"
        print(f"✓ CloudLogitPredictor loaded successfully ({mode_str} mode)")
    elif INPUT_MODE == 'logits_with_real_bk':
        print("\n[TESTING MODE] Using REAL cloud logits (not available in production)")
        print("This tests the theoretical upper bound with perfect cloud information")

    # ============================================================================
    # TRAIN OFFLOAD MECHANISM
    # ============================================================================
    print(f"\n{'='*80}")
    print("TRAINING OFFLOAD MECHANISM")
    print(f"{'='*80}")

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Compute bks and features
    with torch.no_grad():
        all_features, all_bks, all_labels, all_logits, all_images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor,
                local_classifier,
                cloud_cnn,
                train_loader,
                method=0,
                device=device
            )

    b_star = calculate_b_star(all_bks, L0)
    print(f"✓ Computed b* = {b_star:.4f}")

    # Create offload dataset
    combined_data = create_3d_data_deep(
        all_bks, all_features, all_logits, all_images, all_labels,
        input_mode=INPUT_MODE
    )

    offload_dataset = OffloadDatasetCNN(
        combined_data, b_star,
        input_mode=INPUT_MODE,
        include_bk=False,
        use_oracle_labels=False,
        local_clf=local_classifier,
        cloud_clf=cloud_cnn,
        device=device,
        # Soft Labels (only when not in regression mode)
        use_soft_labels=USE_SOFT_LABELS and not USE_REGRESSION_MODE,
        soft_label_temperature=SOFT_LABEL_TEMPERATURE,
        # Regression Mode - target is raw bk values
        regression_target=USE_REGRESSION_MODE
    )

    offload_loader = DataLoader(offload_dataset, batch_size=BATCH_SIZE)

    # Initialize offload mechanism
    offload_model = OffloadMechanism(
        input_mode=INPUT_MODE,
        num_classes=NUM_CLASSES,
        regression_mode=USE_REGRESSION_MODE,
        feat_latent_dim=FEAT_LATENT_DIM if INPUT_MODE == 'hybrid' else 32
    ).to(device)

    optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

    # Train
    train_deep_offload_mechanism(
        offload_model, val_loader, optimizer, offload_loader,
        local_feature_extractor, local_classifier, cloud_cnn,
        b_star, scheduler,
        input_mode=INPUT_MODE, device=device,
        epochs=OFFLOAD_EPOCHS, lr=1e-3, stop_threshold=0.9,
        num_classes=NUM_CLASSES,
        regression_mode=USE_REGRESSION_MODE,  # ★ Use MSELoss for regression
        cloud_predictor=cloud_predictor  # ★ NEW: Pass CloudLogitPredictor
    )

    print("✓ Offload mechanism trained")

    # ============================================================================
    # ANALYSIS 1: ORACLE vs OPTIMIZED GAP
    # ============================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 1: ORACLE vs OPTIMIZED RULE GAP")
    print(f"{'='*80}")

    gap_results = analyze_oracle_optimized_gap(
        offload_model,
        local_feature_extractor,
        local_classifier,
        cloud_cnn,
        test_loader,
        b_star,
        L0=L0,
        tau_border=TAU_BORDER,
        input_mode=INPUT_MODE,
        device=device,
        dataset_name=DATASET_NAME,
        plot=True,
        num_classes=NUM_CLASSES,
        regression_mode=USE_REGRESSION_MODE,  # ★ Pass regression flag
        cloud_predictor=cloud_predictor  # ★ NEW: Pass CloudLogitPredictor
    )

    print("\n" + "="*80)
    print("GAP ANALYSIS SUMMARY")
    print("="*80)
    print(f"  Performance Gap: {gap_results['gap']:.2f}%")
    print(f"  Oracle Accuracy: {gap_results['oracle_acc']:.2f}%")
    print(f"  Optimized Accuracy: {gap_results['optimized_acc']:.2f}%")
    print(f"\n  Disagreement samples: {gap_results['disagreement_count']} ({gap_results['disagreement_misclass']:.1f}% misclass)")
    print(f"  Borderline samples: {gap_results['borderline_count']} ({gap_results['borderline_misclass']:.1f}% misclass)")
    print(f"  Rational failures: {gap_results['rational_failure_count']} ({gap_results['rational_failure_misclass']:.1f}% misclass)")
    print(f"  Normal samples: {gap_results['normal_count']} ({gap_results['normal_misclass']:.1f}% misclass)")

    # ============================================================================
    # ANALYSIS 2: BORDERLINE THRESHOLD SENSITIVITY
    # ============================================================================
    # print(f"\n{'='*80}")
    # print("ANALYSIS 2: BORDERLINE THRESHOLD SENSITIVITY")
    # print(f"{'='*80}")

    # # Test different threshold values
    # tau_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    # # tau_values = [0.01]

    # threshold_results = analyze_borderline_threshold_sensitivity(
    #     offload_model,
    #     local_feature_extractor,
    #     local_classifier,
    #     cloud_cnn,
    #     test_loader,
    #     b_star,
    #     L0=L0,
    #     tau_values=tau_values,
    #     input_mode=INPUT_MODE,
    #     device=device,
    #     dataset_name=DATASET_NAME,
    #     plot=True,
    #     num_classes=NUM_CLASSES,
    #     regression_mode=USE_REGRESSION_MODE  # ★ Pass regression flag
    # )

    # print("\n" + "="*80)
    # print("THRESHOLD SENSITIVITY SUMMARY")
    # print("="*80)
    # print(f"  As τ increases, more samples are classified as 'borderline'")
    # print(f"  Optimal τ appears to be around {tau_values[2]:.2f} - {tau_values[3]:.2f}")
    # print(f"  where the distinction between borderline and normal samples is most meaningful.")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nPlots saved:")
    print(f"  1. oracle_optimized_gap_{DATASET_NAME}_L0{int(L0*100)}.png")
    print(f"  2. borderline_threshold_sensitivity_{DATASET_NAME}_L0{int(L0*100)}.png")
