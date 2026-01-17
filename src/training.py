"""
Training functions for DDNN and Offload Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import os

from src.utils import my_oracle_decision_function
import src.data.data_loader  # Import module to access dynamic NUM_CLASSES


# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DDNN TRAINING FUNCTIONS
# ============================================================================

def train_step_for_DDNN(local_feature_extractor, local_classifier, cloud_cnn, images, labels, optimizer  , local_weight=0.5):
    
    
    """
    Train step that updates both local and cloud networks using cross-entropy.
    Weighted sum of local_loss and cloud_loss.

    Returns total_loss, local_loss, cloud_loss, local_weight, cloud_weight,
    local_features_cpu, bks_cpu.
    The bks_cpu is the cost difference (local_cost - cloud_cost) for each sample
    so we can store it for the offload mechanism.
    """
    
    local_feature_extractor.train()
    local_classifier.train()
    cloud_cnn.train()

    # Forward pass through Local CNN
    # images, labels = images, labels
    images, labels = images.to(device), labels.to(device)

    local_features = local_feature_extractor(images)
    local_predictions = local_classifier(local_features)

    # Forward pass through Cloud CNN for all samples
    cloud_predictions = cloud_cnn(local_features)

    # Compute local loss and cloud loss for all samples, 
    # even if  local or cloud network ist used for the classification the batch will be processed b both classifiers leading to nonzero loss
    local_loss = F.cross_entropy(local_predictions, labels)     # local loss is affected also from the uncertainty of the decision due to the cross entropy calculation.  
    #Not only on the accuracy of the model
    cloud_loss = F.cross_entropy(cloud_predictions, labels)
    
    
    # # Apply complementary weighting on the networks , cloud_weight shoud be 1- local_weight
    # if L0 <= 0:  # check if the local network is in use, if its not set its weight to zero 
    #     local_weight= 0
    #     local_loss = torch.tensor(0.0, requires_grad=True).to(local_loss.device)
    # elif L0 >=1:
    #     local_weight= 1
    #     cloud_loss = torch.tensor(0.0, requires_grad=True).to(cloud_loss.device)
    # else:
    #     local_weight  = random.uniform(0, 1)  # Higher loss => lower weight
    # cloud_weight = 1 - local_weight  # Cloud weight is complementary to local weight
    # local_weight  = random.uniform(0, 1)  # Higher loss => lower weight
    local_weight= random.uniform(0, 1)
    
   
    # local_weight = random.uniform(0, 1)  # Higher loss => lower weight
    cloud_weight = 1 - local_weight  # Cloud weight is complementary to local weight
    
    
    # Weighted total loss
    total_loss = local_weight * local_loss + cloud_weight * cloud_loss
    
    
    

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()




    # Update weights
    optimizer.step()
    
    
    return total_loss.item(), local_loss.item(), cloud_loss.item(), local_weight, cloud_weight


def train_DDNN(train_loader,local_feature_extractor,local_classifier,cloud_cnn,cnn_optimizer,local_weight,epochs_DDNN):
    """
    Train the Distributed Deep Neural Network (DDNN), consisting
    of a local feature extractor + local classifier + cloud CNN,
    using a specified weighting between local and cloud loss.

    Steps:
      - Each batch is passed through local_feature_extractor, local_classifier, and cloud_cnn.
      - We compute local_loss = cross_entropy(local_out, labels).
      - We compute cloud_loss = cross_entropy(cloud_out, labels).
      - Weighted total loss = local_weight*local_loss + (1 - local_weight)*cloud_loss.
      - Backpropagate and optimize.

    Args:
        train_loader (DataLoader): Training dataset loader.
        local_feature_extractor (nn.Module): Local feature extractor network.
        local_classifier (nn.Module): Local classifier network.
        cloud_cnn (nn.Module): Cloud classifier network.
        cnn_optimizer (torch.optim.Optimizer): Optimizer for DDNN parameters.
        local_weight (float): Weighting factor for local_loss. Remainder for cloud_loss.
        epochs_DDNN (int): Number of training epochs.

    Returns:
        None. Prints out progress each epoch (losses, times, local_weight, etc.).
    """
    epoch_running=1        # epoch counter
    
    for epoch in range(epochs_DDNN):
        start_time = time.time()        # measure time
        running_loss = 0.0
        total_batches = len(train_loader)
        total_local_weight = 0.0
        total_cloud_weight = 0.0
        running_cloud_loss = 0.0
        running_local_loss = 0.0


        for images, labels in train_loader:
            images, labels = images, labels
            total_loss, local_loss, cloud_loss, local_weight, cloud_weight= train_step_for_DDNN(
                local_feature_extractor, local_classifier, cloud_cnn, images, labels, cnn_optimizer , local_weight
            )
            
            running_local_loss+= local_loss
            running_cloud_loss+= cloud_loss
            running_loss += total_loss
            total_local_weight += local_weight
            total_cloud_weight += cloud_weight
           
                    
        
        epoch_duration = time.time() - start_time
        avg_loss = running_loss / total_batches
        avg_local_weight= total_local_weight/ total_batches
        avg_cloud_weight= total_cloud_weight/ total_batches
        avg_local_loss = running_local_loss/ total_batches
        avg_cloud_loss = running_cloud_loss/ total_batches

        print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds - Local Weight: {avg_local_weight} Cloud Weight: {avg_cloud_weight}   Average Total Loss: {avg_loss:.4f} Average Local Loss: {avg_local_loss} Average Cloud Loss: {avg_cloud_loss} "')
        



    #     # update epoch counter
        epoch_running+=1 


# ============================================================================
# OFFLOAD MECHANISM TRAINING
# ============================================================================

# NOTE: evaluate_offload_decision_accuracy_CNN_train is imported from src.evaluation
# in train_deep_offload_mechanism to avoid circular imports


def train_deep_offload_mechanism(
    offload_mechanism,
    val_loader,
    offload_optimizer,
    offload_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    b_star,
    offload_scheduler,
    input_mode= 'feat',
    device='cuda',
    threshold=0.5,
    epochs=25,
    lr=1e-4,
    stop_threshold=0.9,
    patience=20 , # Number of epochs to wait for early stopping
    return_history=False ,
    num_classes=10,
    cloud_predictor=None,  # CloudLogitPredictor for logits_with_bk_pred mode
    dataset_name='cifar10'  # Dataset name for checkpoint saving
):
    """
    EXW AFAIRESEI TO TRAIN TESTING SE KATHE EPOCH GIA LOGOUS TAXYTHTAS
    Trains the offload mechanism (offload_mechanism) for a given number of epochs,
    and after each epoch, evaluates the offload accuracy on the same training data (offload_loader).
    If the offload accuracy exceeds 'stop_threshold' (e.g., 0.9), the training loop stops early.

    Args:
        offload_mechanism (nn.Module): The offload model.
        offload_optimizer (torch.optim.Optimizer): Optimizer for the offload model.
        offload_loader (DataLoader): Loader containing (local_feats/figures/logs, label_for_offload,bk_val,local_feats).
        local_feature_extractor (nn.Module): Local feature extractor.
        local_classifier (nn.Module): Local classifier network.
        cloud_cnn (nn.Module): Cloud CNN network.
        device (str): 'cuda' or 'cpu'.
        epochs (int): Number of training epochs.
        lr (float): Learning rate (if needed).
        stop_threshold (float): Early stopping threshold for accuracy (e.g., 0.9 for 90%).
        patience (int): Number of epochs to wait without improvement before early stopping.

    Returns:
        None
    """
    # Import here to avoid circular dependency
    from src.evaluation import (
        evaluate_offload_decision_accuracy_CNN_test,
        evaluate_offload_decision_accuracy_CNN_train
    )
    
    # === Create models directory if it doesn't exist and set checkpoint path ===
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, f"best_offload_mechanism_{dataset_name}.pth")

    # Choose loss function based on input mode
    if input_mode == 'logits_predicted_regression':
        criterion = nn.MSELoss()
        print(f"  [Regression] Using MSELoss for bk prediction")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # -- We add these lists to store accuracies per epoch --
    train_acc_list = []  # Will hold offload_train_acc per epoch
    test_acc_list = []   # Will hold offload_test_acc per epoch

    best_val_acc = -float('inf')  # Best test accuracy for early stopping
    epochs_no_improve = 0  # Epoch counter without improvement

    for epoch in range(epochs):
        offload_mechanism.train()
        total_loss = 0.0

        # -- Training loop --   
        for x_tensor, labels, real_label, feature_tensor in offload_loader:         #x_tensor is feautres/fig/logs depending on input_mode
            # local_feats: shape (batch, 8192) if flattened
            # labels: shape (batch,)
            x_tensor = x_tensor.to(device)
            feature_tensor = feature_tensor.to(device)
            labels = labels.unsqueeze(1).to(device)  # shape: (batch,1)

            # Forward pass: compute logits
            if input_mode == 'hybrid':
                logits = offload_mechanism(x_tensor, feat=feature_tensor)
            elif input_mode == 'logits_with_bk_pred':
                # Compute predicted bk using cloud_predictor
                with torch.no_grad():
                    # Get local logits from x_tensor (first num_classes elements)
                    local_logits = x_tensor[:, :num_classes]
                    if hasattr(cloud_predictor, 'logits_only') and cloud_predictor.logits_only:
                        predicted_cloud_logits = cloud_predictor(local_logits)  # FC-only mode
                    else:
                        predicted_cloud_logits = cloud_predictor(feature_tensor, local_logits)  # CNN+logits mode
                    local_probs = F.softmax(local_logits, dim=1)
                    predicted_cloud_probs = F.softmax(predicted_cloud_logits, dim=1)
                    predicted_bk = predicted_cloud_probs - local_probs  # (batch, 10)
                # Concatenate logits_plus with predicted_bk
                combined_input = torch.cat([x_tensor, predicted_bk], dim=1)  # (batch, 22)
                logits = offload_mechanism(combined_input)
            elif input_mode == 'logits_with_real_bk':
                # ★ TESTING: Compute REAL bk using actual cloud logits
                with torch.no_grad():
                    cloud_cnn.eval()
                    real_cloud_logits = cloud_cnn(feature_tensor)
                    local_probs = F.softmax(x_tensor[:, :num_classes], dim=1)
                    real_cloud_probs = F.softmax(real_cloud_logits, dim=1)
                    real_bk = real_cloud_probs - local_probs  # (batch, 10)
                # Concatenate logits_plus with real_bk
                combined_input = torch.cat([x_tensor, real_bk], dim=1)  # (batch, 22)
                logits = offload_mechanism(combined_input)
            elif input_mode == 'logits_predicted_regression':
                # ★ Regression approach - predict bk directly
                # TRAINING: Uses REAL cloud logits (available during training)
                # Input: local_logits (10) + real_cloud_logits (10) = 20-dim
                local_logits = x_tensor[:, :num_classes]
                with torch.no_grad():
                    cloud_cnn.eval()
                    real_cloud_logits = cloud_cnn(feature_tensor)
                # Concatenate local and REAL cloud logits for training
                combined_input = torch.cat([local_logits, real_cloud_logits], dim=1)  # (batch, 20)
                logits = offload_mechanism(combined_input)
            else:
                logits = offload_mechanism(x_tensor)

            # Compute loss
            loss = criterion(logits, labels)

            # Backprop
            offload_optimizer.zero_grad()
            loss.backward()
            offload_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(offload_loader)

        with torch.no_grad():
            offload_mechanism.eval()

            offload_train_acc = evaluate_offload_decision_accuracy_CNN_train(
                offload_loader,
                local_feature_extractor,
                local_classifier,
                cloud_cnn,
                offload_mechanism,
                b_star,
                threshold,
                input_mode=input_mode,
                num_classes=num_classes,
                cloud_predictor=cloud_predictor
            )
            offload_val_acc = evaluate_offload_decision_accuracy_CNN_test(
                offload_mechanism, local_feature_extractor, local_classifier, cloud_cnn, val_loader, b_star, threshold,
                input_mode=input_mode,
                cloud_predictor=cloud_predictor
            )

        offload_scheduler.step(offload_val_acc)

        current_lr = offload_optimizer.param_groups[0]['lr']
        train_acc_list.append(offload_train_acc)
        test_acc_list.append(offload_val_acc)
        
        # -- Early stopping check based on patience parameter --
        if offload_val_acc > best_val_acc:
            best_val_acc = offload_val_acc
            epochs_no_improve = 0
            # Save best model in models directory
            torch.save(offload_mechanism.state_dict(), checkpoint_path)
            # print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1

        # -- Early stopping check based on stop_threshold --
        # if (offload_train_acc / 100.0) >= stop_threshold:
        #     print(f"Stopping early because train accuracy {offload_train_acc:.2f}% >= {stop_threshold*100:.2f}%")
        #     break

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement. Best validation accuracy: {best_val_acc:.2f}%")
            # Load best model from models directory
            offload_mechanism.load_state_dict(torch.load(checkpoint_path))
            break

        # Print progress every epoch
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Train Acc: {offload_train_acc:.2f}%, Val Acc: {offload_val_acc:.2f}%, LR: {current_lr:.6f}")
    
    # Load the best model at the end if it exists
    if os.path.isfile(checkpoint_path):
        offload_mechanism.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")
    else:
        # no checkpoint was written (e.g. 1 epoch + early stop),
        # so we just keep the final weights in memory
        print(f"[train_deep_offload_mechanism] no checkpoint found at {checkpoint_path}, skipping load.")
        
        
    # -- After training, plot the accuracies across epochs --
    epochs_range = range(1, len(train_acc_list) + 1)
    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs_range, train_acc_list, label="Train Accuracy", marker='o')
    # plt.plot(epochs_range, test_acc_list, label="Test Accuracy", marker='s')
    # plt.title("Offload Mechanism Accuracy vs Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy (%)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    if return_history:
        return {
            'train_accs': train_acc_list,
            'val_accs': test_acc_list
        }
