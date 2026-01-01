"""
Multiple-run experiment to compare oracle labels vs bk-threshold labels
for training the offload mechanism. We train DDNN once, then run N trials
of training the offload mechanism with both labeling methods, and report
average accuracy and local percentage.

IMPORTANT: We calibrate thresholds to ensure SAME local percentage for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# Import all necessary components from latest_deep_offload
from latest_deep_offload import (
    # Data loading
    load_data,
    
    # Model classes
    LocalFeatureExtractor,
    LocalClassifier,
    CloudCNN,
    OffloadMechanism,
    
    # DDNN training functions
    initialize_DDNN_models,
    initialize_DDNN_optimizers,
    train_DDNN,
    
    # Offload mechanism training functions
    compute_bks_input_for_deep_offload,
    calculate_b_star,
    create_3d_data_deep,
    OffloadDatasetCNN,
    train_deep_offload_mechanism,
    test_DDNN_with_optimized_rule,
    evaluate_offload_decision_accuracy_CNN_test,
    my_oracle_decision_function,
    
    # Global variables
    device,
    NUM_CLASSES
)

def oracle_noise_metrics(offload_dataset,
                         local_feat_extr, local_clf, cloud_cnn,
                         bk_star, tau=0.02,
                         input_mode='feat',
                         device='cuda', batch_size=256,
                         oracle_decision_func=my_oracle_decision_function, plot: bool = True):
    """
    Evaluate the *upper performance bound* that an off‑load decision network
    could theoretically achieve, and quantify how much training‐label noise
    exists in the current (bk‑based) supervision.
    """
    loader = DataLoader(offload_dataset, batch_size=batch_size, shuffle=False)

    local_feat_extr.eval()
    local_clf.eval()
    cloud_cnn.eval()

    N = oracle_ok = noisy = borders = 0
    routed_local = 0
    
    # Track label distributions
    bk_local_count = 0  # dom_lab == 0
    bk_cloud_count = 0  # dom_lab == 1
    oracle_local_count = 0  # oracle_route == 0
    oracle_cloud_count = 0  # oracle_route == 1
    
    with torch.no_grad():
        for x_tensor, dom_lab, true_lab, features, bk_val in loader:
            x_tensor, dom_lab, true_lab, features, bk_val = \
                x_tensor.to(device), dom_lab.to(device), true_lab.to(device), features.to(device), bk_val.to(device)

            loc_logits   = local_clf(features)
            cloud_logits = cloud_cnn(features)

            oracle_route = oracle_decision_func(loc_logits, cloud_logits, true_lab, bk_star)

            loc_pred   = loc_logits.argmax(1)
            cloud_pred = cloud_logits.argmax(1)
            final_pred = torch.where(oracle_route == 0, loc_pred, cloud_pred)
            oracle_ok += (final_pred == true_lab).sum().item()

            noisy += (oracle_route.float() != dom_lab).sum().item()
            borders += (bk_val.abs() < tau).sum().item()

            N += true_lab.size(0)
            routed_local += (oracle_route == 0).sum().item()
            
            # Count label distributions
            bk_local_count += (dom_lab == 0).sum().item()
            bk_cloud_count += (dom_lab == 1).sum().item()
            oracle_local_count += (oracle_route == 0).sum().item()
            oracle_cloud_count += (oracle_route == 1).sum().item()
    
    metrics = {
        "oracle_acc" : 100 * oracle_ok / N,
        "noise_rate" : 100 * noisy   / N,
        "border_rate": 100 * borders / N,
        "local_rate":  100 * routed_local / N,
        "bk_local_pct": 100 * bk_local_count / N,
        "bk_cloud_pct": 100 * bk_cloud_count / N,
        "oracle_local_pct": 100 * oracle_local_count / N,
        "oracle_cloud_pct": 100 * oracle_cloud_count / N
    }

    if plot:
        # Create figure with 2 subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # ========== TOP SUBPLOT: Noise Rate & Border Rate ==========
        labels_top = ['Noise Rate', 'Border Rate']
        values_top = [metrics["noise_rate"], metrics["border_rate"]]
        colors_top = ['#FF6B6B', '#FFA500']  # Red for noise, Orange for border
        
        bars_top = ax1.bar(labels_top, values_top, color=colors_top, alpha=0.8, width=0.6)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Label Quality Metrics', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars_top, values_top):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add explanatory text below bars
        ax1.text(0, -15, 'Mislabeled Samples', 
                ha='center', fontsize=9, style='italic', color='#FF6B6B')
        ax1.text(1, -15, f'Samples with\n|Bx| < {tau}', 
                ha='center', fontsize=9, style='italic', color='#FFA500')
        
        # ========== BOTTOM SUBPLOT: Label Distribution Comparison ==========
        x_pos = np.array([0, 1])
        width = 0.35  # Αλλαγή από 0.4 σε 0.35
        
        # BK Labeling (Bx-based)
        bk_local_bar = ax2.bar(x_pos[0] - width/2, metrics["bk_local_pct"], width, 
                               label='Local (0)', color='#4CAF50', alpha=0.8)
        bk_cloud_bar = ax2.bar(x_pos[0] + width/2, metrics["bk_cloud_pct"], width, 
                               label='Cloud (1)', color='#F44336', alpha=0.8)
        
        # Oracle Labeling
        oracle_local_bar = ax2.bar(x_pos[1] - width/2, metrics["oracle_local_pct"], width, 
                                   color='#4CAF50', alpha=0.8)
        oracle_cloud_bar = ax2.bar(x_pos[1] + width/2, metrics["oracle_cloud_pct"], width, 
                                   color='#F44336', alpha=0.8)
        
        # Add value labels on bars
        for bar_set in [bk_local_bar, bk_cloud_bar, oracle_local_bar, oracle_cloud_bar]:
            for bar in bar_set:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Percentage of Samples (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Label Distribution: Optimized Rule vs Oracle', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Optimized Rule\nBx Labeling', 'Oracle Labeling'], 
                           fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Overall title
        fig.suptitle('MetaDataset Labeling Analysis', 
                    fontsize=15, fontweight='bold', y=0.98)
        
        # Add summary box at bottom
        summary_text = (
            f"Total: {N:,} | "
            f"Noise: {metrics['noise_rate']:.1f}% | "
            f"Border: {metrics['border_rate']:.1f}% | "
            f"Bx Local: {metrics['bk_local_pct']:.1f}% | "
            f"Oracle Local: {metrics['oracle_local_pct']:.1f}%"
        )
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        # Save the figure
        output_path = 'oracle_noise_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved oracle metrics plot: {output_path}")
        plt.close(fig)
    return metrics


def find_threshold_for_target_local_percentage(
    offload_mechanism,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    test_loader,
    b_star,
    target_local_percentage,
    input_mode='logits_plus',
    tolerance=1.0,
    max_iterations=20
):
    """
    Binary search to find threshold that gives target_local_percentage.
    """
    print(f"\n[Calibration] Searching for threshold to achieve {target_local_percentage:.1f}% local processing...")
    
    target_normalized = target_local_percentage / 100.0
    
    threshold = 0.5
    threshold_low = 0.0
    threshold_high = 1.0
    
    best_threshold = threshold
    best_local_perc = 0.0
    best_accuracy = 0.0
    
    for iteration in range(max_iterations):
        local_perc, overall_acc = test_DDNN_with_optimized_rule(
            offload_mechanism=offload_mechanism,
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            test_loader=test_loader,
            b_star=b_star,
            threshold=threshold,
            input_mode=input_mode,
            device=device
        )
        
        print(f"  Iter {iteration+1}/{max_iterations}: τ={threshold:.4f} → local%={local_perc:.2f}%, acc={overall_acc:.2f}%")
        
        if abs(local_perc - target_local_percentage) < abs(best_local_perc - target_local_percentage):
            best_threshold = threshold
            best_local_perc = local_perc
            best_accuracy = overall_acc
        
        if abs(local_perc - target_local_percentage) <= tolerance:
            print(f"  ✓ Found threshold τ={threshold:.4f} with local%={local_perc:.2f}% (target: {target_local_percentage:.1f}%)")
            return threshold, local_perc, overall_acc
        
        if local_perc < target_local_percentage:
            threshold_high = threshold
            threshold_low = target_normalized
        else:
            threshold_high = target_normalized
            threshold_low = threshold
        
        threshold = (threshold_low + threshold_high) / 2.0
    
    print(f"  ⚠ Max iterations reached. Best: τ={best_threshold:.4f}, local%={best_local_perc:.2f}% (target: {target_local_percentage:.1f}%)")
    return best_threshold, best_local_perc, best_accuracy

def train_ddnn_once(train_loader, val_loader, batch_size=256, epochs_DDNN=60, 
                    local_weight=0.7, learning_rate=0.001, dataset_name='cifar10'):
    """
    Train the DDNN (local feature extractor + local classifier + cloud CNN) once.
    """
    print("="*80)
    print("TRAINING DDNN MODELS (ONE-TIME)")
    print("="*80)
    
    local_feature_extractor, local_classifier, cloud_cnn = initialize_DDNN_models()
    
    cnn_optimizer = initialize_DDNN_optimizers(
        local_feature_extractor, 
        local_classifier, 
        cloud_cnn, 
        lr=learning_rate
    )
    
    train_DDNN(
        train_loader,
        local_feature_extractor,
        local_classifier,
        cloud_cnn,
        cnn_optimizer,
        local_weight,
        epochs_DDNN
    )
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    torch.save(local_feature_extractor.state_dict(), 
               os.path.join(models_dir, f"local_feature_extractor_{dataset_name}.pth"))
    torch.save(local_classifier.state_dict(), 
               os.path.join(models_dir, f"local_classifier_{dataset_name}.pth"))
    torch.save(cloud_cnn.state_dict(), 
               os.path.join(models_dir, f"cloud_cnn_{dataset_name}.pth"))
    
    print(f"\n✓ DDNN models trained and saved to {models_dir}/")
    
    return local_feature_extractor, local_classifier, cloud_cnn


def run_single_experiment(
    use_oracle: bool,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    train_loader,
    val_loader,
    test_loader,
    batch_size=256,
    L0=0.54,
    epochs_optimization=70,
    input_mode='logits_plus',
    learning_rate=0.001,
    weight_decay=1e-4,
    dropout_prob=0.1,
    target_local_percentage=None
):
    """
    Run a single training experiment with specified oracle label setting.
    """
    
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    all_local_features, all_bks, all_labels, all_logits, all_images = \
        compute_bks_input_for_deep_offload(
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            train_loader,
            method=0,
            device=device
        )
    
    b_star = calculate_b_star(all_bks, L0)
    
    combined_data = create_3d_data_deep(
        all_bks, 
        all_local_features, 
        all_logits, 
        all_images, 
        all_labels, 
        input_mode=input_mode
    )
    
    offload_dataset = OffloadDatasetCNN(
        combined_data, 
        b_star,
        input_mode=input_mode,
        include_bk=False,
        use_oracle_labels=use_oracle,
        local_clf=local_classifier,
        cloud_clf=cloud_cnn,
        device=device
    )
    
    offload_loader = DataLoader(offload_dataset, batch_size=batch_size, shuffle=False)
    
    if input_mode == 'logits_plus':
        input_dim = NUM_CLASSES + 2
    elif input_mode == 'logits':
        input_dim = NUM_CLASSES
    else:
        input_dim = None
        
    if input_mode == 'feat':
        deep_offload_model = OffloadMechanism(
            input_shape=(32, 16, 16),
            input_mode=input_mode,
            conv_dims=[64, 128, 256],
            num_layers=1,
            fc_dims=[256, 128, 1],
            dropout_p=0.35,
            latent_in=(1,)
        ).to(device)
    else:
        deep_offload_model = OffloadMechanism(
            input_shape=(input_dim,),
            input_mode=input_mode,
            fc_dims=[256, 128, 64, 32, 1],
            dropout_p=dropout_prob,
            latent_in=(1,)
        ).to(device)
    
    optimizer = torch.optim.Adam(
        deep_offload_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5
    )
    
    train_deep_offload_mechanism(
        offload_mechanism=deep_offload_model,
        val_loader=val_loader,
        offload_optimizer=optimizer,
        offload_loader=offload_loader,
        local_feature_extractor=local_feature_extractor,
        local_classifier=local_classifier,
        cloud_cnn=cloud_cnn,
        b_star=b_star,
        offload_scheduler=scheduler,
        input_mode=input_mode,
        device=device,
        threshold=0.5,
        epochs=epochs_optimization,
        lr=learning_rate,
        stop_threshold=0.9
    )
    
    if target_local_percentage is not None:
        threshold, local_percentage, overall_accuracy = find_threshold_for_target_local_percentage(
            offload_mechanism=deep_offload_model,
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            test_loader=test_loader,
            b_star=b_star,
            target_local_percentage=target_local_percentage,
            input_mode=input_mode,
            tolerance=2.0,
            max_iterations=20
        )
    else:
        threshold = 0.5
        local_percentage, overall_accuracy = test_DDNN_with_optimized_rule(
            offload_mechanism=deep_offload_model,
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            test_loader=test_loader,
            b_star=b_star,
            threshold=threshold,
            input_mode=input_mode,
            device=device
        )
    
    offload_val_acc = evaluate_offload_decision_accuracy_CNN_test(
        deep_offload_model,
        local_feature_extractor,
        local_classifier,
        cloud_cnn,
        val_loader,
        b_star,
        threshold=threshold,
        input_mode=input_mode,
        device=device
    )
    
    return {
        'overall_accuracy': overall_accuracy,
        'local_percentage': local_percentage,
        'offload_val_acc': offload_val_acc,
        'threshold': threshold
    }


def run_comparison_experiments(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    train_loader,
    val_loader,
    test_loader,
    batch_size=256,
    L0=0.54,
    epochs_optimization=70,
    input_mode='logits_plus',
    learning_rate=0.001,
    weight_decay=1e-4,
    dropout_prob=0.1
):
    """
    Run both Oracle and BK-threshold experiments and return results.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: ORACLE LABELING")
    print("="*80)
    oracle_result = run_single_experiment(
        use_oracle=True,
        local_feature_extractor=local_feature_extractor,
        local_classifier=local_classifier,
        cloud_cnn=cloud_cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        batch_size=batch_size,
        L0=L0,
        epochs_optimization=epochs_optimization,
        input_mode=input_mode,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout_prob=dropout_prob,
        target_local_percentage=None
    )
    
    print(f"\n✓ Oracle result → Acc: {oracle_result['overall_accuracy']:.2f}%, Local%: {oracle_result['local_percentage']:.2f}%, Val: {oracle_result['offload_val_acc']:.2f}%, τ={oracle_result['threshold']:.4f}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: ORIGINAL (BK-threshold) WITH CALIBRATION")
    print("="*80)
    print(f"Target: Match oracle's local% = {oracle_result['local_percentage']:.2f}%")
    
    bk_result = run_single_experiment(
        use_oracle=False,
        local_feature_extractor=local_feature_extractor,
        local_classifier=local_classifier,
        cloud_cnn=cloud_cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        batch_size=batch_size,
        L0=L0,
        epochs_optimization=epochs_optimization,
        input_mode=input_mode,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout_prob=dropout_prob,
        target_local_percentage=oracle_result['local_percentage']
    )
    
    print(f"\n✓ Original (BK) result → Acc: {bk_result['overall_accuracy']:.2f}%, Local%: {bk_result['local_percentage']:.2f}%, Val: {bk_result['offload_val_acc']:.2f}%, τ={bk_result['threshold']:.4f}")
    
    return oracle_result, bk_result


def plot_experiment_results(oracle_result, bk_result, output_path='comparing_bx_values_methods.png'):
    """
    Create visualization comparing Oracle vs BK-threshold experiment results.
    """
    labels = ['original', 'oracle labeling']
    overall_accs = [bk_result['overall_accuracy'], oracle_result['overall_accuracy']]
    local_perc = [bk_result['local_percentage'], oracle_result['local_percentage']]
    val_accs = [bk_result['offload_val_acc'], oracle_result['offload_val_acc']]
    thresholds = [bk_result['threshold'], oracle_result['threshold']]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    
    ax0 = ax[0]
    bars1 = ax0.bar(x - width/2, overall_accs, width, label='Overall DDNN Acc (%)', color='steelblue')
    bars2 = ax0.bar(x + width/2, val_accs, width, label='Offload Val Acc (%)', color='orange')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=11)
    ax0.set_ylabel('Accuracy (%)', fontsize=11)
    ax0.set_title('Comparing Accuracy: Original vs Oracle Labeling', fontsize=12, fontweight='bold')
    ax0.legend(fontsize=10)
    ax0.grid(axis='y', linestyle='--', alpha=0.4)
    ax0.set_ylim([0, 100])
    
    ax1 = ax[1]
    bars3 = ax1.bar(x, local_perc, width*1.5, label='Local %', color='green', alpha=0.7)
    
    for bar in bars3:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1_twin = ax1.twinx()
    line = ax1_twin.plot(x, thresholds, marker='o', color='red', linestyle='--', 
                         label='Threshold τ', linewidth=2.5, markersize=10)
    
    for xi, thresh in zip(x, thresholds):
        ax1_twin.text(xi, thresh, f'{thresh:.3f}', ha='center', va='bottom', 
                     fontsize=9, color='red', fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Local Percentage (%)', color='green', fontsize=11, fontweight='bold')
    ax1_twin.set_ylabel('Threshold τ', color='red', fontsize=11, fontweight='bold')
    ax1.set_title('Local % and Selected Threshold', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1_twin.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_ylim([0, 100])
    ax1_twin.set_ylim([0, 1])
    
    plt.suptitle('Comparing Bx Values Methods', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved comparison plot: {output_path}")


def print_final_summary(oracle_result, bk_result):
    """
    Print final summary of comparison results.
    """
    overall_accs = [bk_result['overall_accuracy'], oracle_result['overall_accuracy']]
    local_perc = [bk_result['local_percentage'], oracle_result['local_percentage']]
    thresholds = [bk_result['threshold'], oracle_result['threshold']]
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n  Original (BK)  → Overall Acc: {overall_accs[0]:.2f}%, Local%: {local_perc[0]:.2f}%, τ={thresholds[0]:.4f}")
    print(f"  Oracle         → Overall Acc: {overall_accs[1]:.2f}%, Local%: {local_perc[1]:.2f}%, τ={thresholds[1]:.4f}")
    
    acc_diff = overall_accs[1] - overall_accs[0]
    local_diff = abs(local_perc[1] - local_perc[0])
    
    print(f"\n📊 Comparison:")
    print(f"  ▪ Accuracy difference (Oracle - Original): {acc_diff:+.2f}%")
    print(f"  ▪ Local % difference: {local_diff:.2f}%")
    
    if local_diff <= 2.0:
        print(f"  ✓ Local percentages matched successfully (within 2% tolerance)")
    else:
        print(f"  ⚠ Local percentages differ by {local_diff:.2f}%")
    
    if abs(acc_diff) > 0.5:
        winner = "Oracle labeling" if acc_diff > 0 else "Original (BK-threshold)"
        print(f"\n✓ Winner: {winner} (better by {abs(acc_diff):.2f}% accuracy)")
    else:
        print(f"\n≈ Both methods perform similarly (difference: {abs(acc_diff):.2f}%)")
    
    print("\n" + "="*80)


def main():
    """
    Main function: analyze dataset labeling and optionally run experiments.
    """
    # ========================================================================
    # SETTINGS
    # ========================================================================
    mode = 'load'
    
    batch_size = 256
    L0 = 0.54
    epochs_DDNN = 30
    epochs_optimization = 1
    input_mode = 'logits_plus'
    learning_rate = 0.001
    weight_decay = 1e-4
    dropout_prob = 0.1
    local_weight = 0.7
    dataset_name = 'cifar10'
    
    print("="*80)
    print("META DATASET LABELING ANALYSIS")
    print("="*80)
    print(f"\nSettings:")
    print(f"  Mode: {mode.upper()}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  L0: {L0}")
    print(f"  Input mode: {input_mode}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\nLoading data...")
    train_loader, val_loader, test_loader = load_data(batch_size, dataset=dataset_name)
    
    # ========================================================================
    # LOAD DDNN MODELS
    # ========================================================================
    models_dir = "models"
    ddnn_model_exists = all([
        os.path.exists(os.path.join(models_dir, f"local_feature_extractor_{dataset_name}.pth")),
        os.path.exists(os.path.join(models_dir, f"local_classifier_{dataset_name}.pth")),
        os.path.exists(os.path.join(models_dir, f"cloud_cnn_{dataset_name}.pth"))
    ])
    
    if mode == 'train':
        print(f"\n[MODE: TRAIN] Training DDNN models from scratch...")
        local_feature_extractor, local_classifier, cloud_cnn = train_ddnn_once(
            train_loader, val_loader, batch_size, epochs_DDNN, 
            local_weight, learning_rate, dataset_name
        )
    elif mode == 'load':
        if not ddnn_model_exists:
            print(f"\n[ERROR] mode='load' but DDNN models not found in '{models_dir}/'")
            print("Please set mode='train' first to create the models.")
            return
        print(f"\n[MODE: LOAD] Loading pre-trained DDNN models from '{models_dir}/'...")
        local_feature_extractor, local_classifier, cloud_cnn = initialize_DDNN_models()
        local_feature_extractor.load_state_dict(
            torch.load(os.path.join(models_dir, f"local_feature_extractor_{dataset_name}.pth"),
                      weights_only=True)
        )
        local_classifier.load_state_dict(
            torch.load(os.path.join(models_dir, f"local_classifier_{dataset_name}.pth"),
                      weights_only=True)
        )
        cloud_cnn.load_state_dict(
            torch.load(os.path.join(models_dir, f"cloud_cnn_{dataset_name}.pth"),
                      weights_only=True)
        )
        print("✓ DDNN models loaded successfully")
    else:
        print(f"\n[ERROR] Invalid mode '{mode}'. Use 'train' or 'load'.")
        return
    
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    # ========================================================================
    # CREATE OFFLOAD DATASET AND ANALYZE LABELING
    # ========================================================================
    print("\nCreating offload dataset...")
    
    # Compute features and bk values
    all_local_features, all_bks, all_labels, all_logits, all_images = \
        compute_bks_input_for_deep_offload(
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            train_loader,
            method=0,
            device=device
        )
    
    b_star = calculate_b_star(all_bks, L0)
    
    combined_data = create_3d_data_deep(
        all_bks, 
        all_local_features, 
        all_logits, 
        all_images, 
        all_labels, 
        input_mode=input_mode
    )
    

    metrics_dataset = OffloadDatasetCNN(combined_data, b_star, input_mode=input_mode, include_bk=True)

    metrics = oracle_noise_metrics(metrics_dataset,
                                local_feature_extractor,
                                local_classifier,
                                cloud_cnn,
                                bk_star=b_star,
                                tau=0.01,
                                input_mode='feat',
                                device=device)
    # ========================================================================
    # OPTIONALLY RUN EXPERIMENTS
    # ========================================================================
    # Uncomment to run full experiments
    # oracle_result, bk_result = run_comparison_experiments(
    #     local_feature_extractor=local_feature_extractor,
    #     local_classifier=local_classifier,
    #     cloud_cnn=cloud_cnn,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     batch_size=batch_size,
    #     L0=L0,
    #     epochs_optimization=epochs_optimization,
    #     input_mode=input_mode,
    #     learning_rate=learning_rate,
    #     weight_decay=weight_decay,
    #     dropout_prob=dropout_prob
    # )
    # plot_experiment_results(oracle_result, bk_result)
    # print_final_summary(oracle_result, bk_result)
    
    print("\nDone.")

if __name__ == "__main__":
    main()