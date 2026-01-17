"""
Utility functions for DDNN training, evaluation, and analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_models():
    """
        Initializes all models, the DDNN models (local_feature_extractor,local_classifier,cloud), the offload mechanism model and returns them
    
    """
    from src.models import LocalFeatureExtractor, LocalClassifier, CloudCNN, OffloadMechanism
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_feature_extractor = LocalFeatureExtractor().to(device)
    local_classifier = LocalClassifier().to(device)
    cloud_cnn = CloudCNN().to(device)
    offload_mechanism = OffloadMechanism().to(device)
    return local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism

# Initialize optimizers
def initialize_optimizers(local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism ,lr=0.001):
    """
    Initializes all optimizers, DDNN optimizers for local_feature_extractor,local_classifier,cloud, offload mechanism model optimizer and returns them

    """
    cnn_optimizer = optim.Adam(list(local_feature_extractor.parameters()) + list(local_classifier.parameters()) + list(cloud_cnn.parameters()), lr=lr)
    offload_optimizer = optim.Adam(offload_mechanism.parameters(),0.005 ) #working with 0.000000001

    return  cnn_optimizer, offload_optimizer


# ============================================================================
# BENEFIT CALCULATION AND THRESHOLD FUNCTIONS
# ============================================================================

def calculate_b_star( all_bks, L0=0.3):
    """_summary_
    Calculate  critical b* such that L0 percent of tasks have benefits lower than b*
    For a sample if bk<b* its defined as an easy task and gets classified locally
    For example, if L0=0.3 => b_star is the 30th percentile of all bks.
    We interpret bks<b_star => classify locally.
    
    ALGO:
    1) Sort bks in increasing order
    2) find critical b*
    Args:
        all_bks (_type_): _description_
        L0 (float, optional): _description_. Defaults to 0.3.
    """
    
    # Debug print: Check the range and basic statistics of bks
    # print(f' LO USED: {L0}')
    # print(f"Mean of bks: {np.mean(all_bks)}")
    # print(f"Median of bks: {np.median(all_bks)}")
    # print(f"Standard deviation of bks: {np.std(all_bks)}")
    # print(f"Max of bks: {np.max(all_bks)}, Min of bks: {np.min(all_bks)}")
    
    #  Plot histogram for visualization
    # plt.hist(all_bks, bins=50, alpha=0.75, color='blue')
    # plt.title('Distribution of bks')
    # plt.xlabel('bk values')
    # plt.ylabel('Frequency')
    # plt.show()
    
    # Calculate b* as the L0 percentile of all bks
    b_star = np.percentile(all_bks, L0 * 100)
    # print(f"Calculated b*: {b_star}")
    
    return b_star


def calibrate_threshold(model, loader, *, device='cuda'):
    """
    Επιστρέφει το κατώφλι τ ώστε
    P(pred_prob > τ) == P(label == 1)  (ίδιο cloud% με τα labels).
    """
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for x, y_off, *_ in loader:
            p = torch.sigmoid(model(x.to(device))).squeeze(1).cpu()
            probs.append(p)
            labels.append(y_off)

    probs  = torch.cat(probs).numpy()      # N,
    labels = torch.cat(labels).numpy()     # N,

    cloud_ratio = (labels == 1).mean()     # το ξέραμε, απλώς το μετράμε
    tau = np.quantile(probs, 1 - cloud_ratio)
    return float(tau), cloud_ratio


# ============================================================================
# ENTROPY AND PROBABILITY FUNCTIONS
# ============================================================================

def calculate_normalized_entropy(predictions):
    """
    Takes logits => compute softmax => normalized entropy = -(Sum p*log(p))/log(#classes).
    Higher => more uncertain, lower => more confident.
    """
    
    # Apply softmax to logits to get probabilities
    probabilities = F.softmax(predictions, dim=1)
    
    # predictions is a tensor of shape (batch_size, num_classes)
    epsilon = 1e-5  # To prevent log(0)
    log_c = torch.log(torch.tensor(probabilities.shape[1], dtype=torch.float32))  # |C| = number of classes (usually 10)
    entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1) / log_c
    return entropy


def get_top2_probs(prob_tensor):
    """
    Given a probability tensor of shape (batch, num_classes),
    return the top two probabilities for each sample.
    """
    sorted_probs, _ = torch.sort(prob_tensor, descending=True, dim=1)
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    return p1, p2

def normalized_entropy(logits):
    """
    Computes the normalized entropy for each sample given logits.
    The normalization is done by dividing by log(num_classes).
    """
    probs = F.softmax(logits, dim=1)
    epsilon = 1e-5  # to avoid log(0)
    num_classes = probs.shape[1]
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1) / torch.log(torch.tensor(float(num_classes)))
    return entropy


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def create_3d_data_deep(bks, feats, logits, imgs, labels, *, input_mode='feat'):
    """
    Builds combined_data = list of tuples:
        (x_representation, bk_value, real_label)

    input_mode
      'feat'   → use feature‑map   tensor (32,16,16)
      'logits' → use logits        tensor (10,)
      'img'/'figure' → use raw image tensor (3,32,32)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined_data = []
    for i in range(len(bks)):
        if input_mode == 'feat':
            x_tensor = feats[i]
        elif input_mode == 'shallow_feat':
            x_tensor = feats[i]
        elif input_mode == 'logits':
            x_tensor = logits[i]
        elif input_mode in ('img', 'figure'):
            x_tensor = imgs[i]
        elif input_mode == 'logits_plus':
        # όχι: logits_i = logits[i]
            logits_i = torch.tensor(logits[i], dtype=torch.float32, device=device)
            probs    = F.softmax(logits_i, dim=0)
            top2     = torch.topk(probs, 2).values
            margin   = top2[0] - top2[1]
            entropy = (-probs * torch.log(probs + 1e-9)).sum() / torch.log(torch.tensor(probs.size(0), dtype=torch.float32))
            x_tensor = torch.cat([logits_i, margin.view(1), entropy.view(1)])  # shape (12,)
                                    # ⇒ (12,)
        elif input_mode == 'hybrid':
            # ★ HYBRID mode: x_tensor = logits_plus, feats are passed separately
            logits_i = torch.tensor(logits[i], dtype=torch.float32, device=device)
            probs    = F.softmax(logits_i, dim=0)
            top2     = torch.topk(probs, 2).values
            margin   = top2[0] - top2[1]
            entropy = (-probs * torch.log(probs + 1e-9)).sum() / torch.log(torch.tensor(probs.size(0), dtype=torch.float32))
            x_tensor = torch.cat([logits_i, margin.view(1), entropy.view(1)])  # shape (12,)
            # Features are stored in the 4th element of the tuple
        elif input_mode == 'logits_with_bk_pred':
            # ★ NEW: logits_plus + predicted_bk (computed during training)
            logits_i = torch.tensor(logits[i], dtype=torch.float32, device=device)
            probs    = F.softmax(logits_i, dim=0)
            top2     = torch.topk(probs, 2).values
            margin   = top2[0] - top2[1]
            entropy = (-probs * torch.log(probs + 1e-9)).sum() / torch.log(torch.tensor(probs.size(0), dtype=torch.float32))
            x_tensor = torch.cat([logits_i, margin.view(1), entropy.view(1)])  # shape (12,)
            # predicted_bk will be concatenated during training/evaluation
        elif input_mode == 'logits_with_real_bk':
            # ★ TESTING: logits_plus + real_bk (using actual cloud logits)
            logits_i = torch.tensor(logits[i], dtype=torch.float32, device=device)
            probs    = F.softmax(logits_i, dim=0)
            top2     = torch.topk(probs, 2).values
            margin   = top2[0] - top2[1]
            entropy = (-probs * torch.log(probs + 1e-9)).sum() / torch.log(torch.tensor(probs.size(0), dtype=torch.float32))
            x_tensor = torch.cat([logits_i, margin.view(1), entropy.view(1)])  # shape (12,)
            # real_bk will be concatenated during training/evaluation
        elif input_mode == 'logits_predicted_regression':
            # ★ NEW: Regression mode - just store local logits
            # Input: local_logits (10) + cloud_logits (10) = 20-dim
            # Cloud logits are concatenated during training/evaluation
            logits_i = torch.tensor(logits[i], dtype=torch.float32, device=device)
            x_tensor = logits_i  # shape (10,) - just the local logits
        else:
            raise ValueError("input_mode must be 'feat'|'logits'|'img'|'logits_plus'|'hybrid'|'logits_with_bk_pred'|'logits_with_real_bk'|'logits_predicted_regression'")
        combined_data.append((x_tensor, float(bks[i]), int(labels[i]), feats[i]))
    return combined_data


# ============================================================================
# BK COMPUTATION FUNCTION
# ============================================================================

def compute_bks_input_for_deep_offload(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    data_loader,
    method=0,
    beta=0.2,
    device='cuda',
    entropy_threshold=0.1  # Add a default or pass it as an argument4
    ,tie_threshold=0.05,
    improvement_scale=0.1
):
    """
    This unified function computes bk = local_cost - cloud_cost for each sample
    under four possible methods (selected by 'method'):
      - method=0: 'original' approach
      - method=1: 'binary difference penalty'
      - method=2: 'difference penalty'
      - method=3: 'new logic' (the special tie-entropy approach)

    Args:
      local_feature_extractor (nn.Module): Extracts local features from images.
      local_classifier (nn.Module): Local classification head.
      cloud_cnn (nn.Module): Cloud classification head.
      data_loader (DataLoader): The dataset from which we compute bk values.
      method (int): 0,1,2,3 specifying which cost logic to follow.
      beta (float): Penalty factor used in some tie cases.
      device (str): 'cuda' or 'cpu'.
      return_features (bool): Whether to return local features as well.
      entropy_threshold (float): If method=3, threshold for local entropy.

    Returns:
      If return_features=True:
         (all_local_features, all_bks)
      else:
         (None, all_bks).

      - all_local_features is shape (N, 32,16,16) for CIFAR-10
      - all_bks is shape (N,) with one bk per sample
    """


    def get_top2_probs(prob_tensor):
        """Επιστρέφει p1, p2 (με descending σειρά) για κάθε δείγμα."""
        sorted_p, _ = torch.sort(prob_tensor, descending=True, dim=1)
        p1 = sorted_p[:, 0]
        p2 = sorted_p[:, 1]
        return p1, p2

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    all_features_list = [] # save all the features
    all_bks_list = []       #save all bks
    all_labels_list = []    #save all true labels
    all_logits_list = []    #save all logits
    all_images_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            # 1) Forward: local feats => local_out => cloud_out
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)  # raw logits
            cloud_out = cloud_cnn(local_feats)         # raw logits
            logitsL= local_out # save logits to take them out

            # 2) Predictions & correctness
            local_pred = local_out.argmax(dim=1)
            cloud_pred = cloud_out.argmax(dim=1)
            local_correct_mask = (local_pred == labels)
            cloud_correct_mask = (cloud_pred == labels)

            # 3) Probabilities of the correct class
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)
            local_prob_correct = local_probs[range(batch_size), labels]
            cloud_prob_correct = cloud_probs[range(batch_size), labels]

            # Initialize cost arrays
            local_cost = torch.zeros(batch_size, device=device)
            cloud_cost = torch.zeros(batch_size, device=device)
            local_cost = 1.0 - local_prob_correct
            cloud_cost = 1.0 - cloud_prob_correct
            # Decide cost logic
            if method == 0:
                # ---- METHOD=0 (Original Approach) ----
                local_cost = 1.0 - local_prob_correct
                cloud_cost = 1.0 - cloud_prob_correct

                # (Optional tie penalty if needed)
                # both_correct_mask = local_correct_mask & cloud_correct_mask
                # idx_tie = both_correct_mask.nonzero(as_tuple=True)[0]
                # cloud_cost[idx_tie] += beta * (cloud_prob_correct[idx_tie] - local_prob_correct[idx_tie])

            elif method == 1:
                # ---- METHOD=1 (Binary difference penalty) ----
                # (i) local=1, cloud=0 => (0,1)
                mask_l1_c0 = local_correct_mask & (~cloud_correct_mask)
                idx_l1_c0 = mask_l1_c0.nonzero(as_tuple=True)[0]
                if len(idx_l1_c0) > 0:
                    local_cost[idx_l1_c0] = 0.0
                    cloud_cost[idx_l1_c0] = 1.0

                # (ii) local=0, cloud=1 => (1,0)
                mask_l0_c1 = (~local_correct_mask) & cloud_correct_mask
                idx_l0_c1 = mask_l0_c1.nonzero(as_tuple=True)[0]
                if len(idx_l0_c1) > 0:
                    local_cost[idx_l0_c1] = 1.0
                    cloud_cost[idx_l0_c1] = 0.0

                # (iii) local=0, cloud=0 => (1,1)
                mask_l0_c0 = (~local_correct_mask) & (~cloud_correct_mask)
                idx_l0_c0 = mask_l0_c0.nonzero(as_tuple=True)[0]
                if len(idx_l0_c0) > 0:
                    local_cost[idx_l0_c0] = 1.0
                    cloud_cost[idx_l0_c0] = 1.0

                # (iv) tie => local=1, cloud=1 => local_cost=1 - loc_prob, cloud_cost= ...
                mask_l1_c1 = local_correct_mask & cloud_correct_mask
                idx_l1_c1 = mask_l1_c1.nonzero(as_tuple=True)[0]
                if len(idx_l1_c1) > 0:
                    loc_probs = local_prob_correct[idx_l1_c1]
                    cld_probs = cloud_prob_correct[idx_l1_c1]
                    local_cost[idx_l1_c1] = 1.0 - loc_probs
                    cloud_cost[idx_l1_c1] = (1.0 - cld_probs) + beta * (cld_probs - loc_probs)

            elif method == 2:
                # ---- METHOD=2 (Difference penalty) ----
                local_cost = 1.0 - local_prob_correct
                cloud_cost = 1.0 - cloud_prob_correct
                # tie => cloud_cost += beta*(cloud_prob - local_prob)
                both_correct_mask = local_correct_mask & cloud_correct_mask
                idx_tie = both_correct_mask.nonzero(as_tuple=True)[0]
                if len(idx_tie) > 0:
                    diff_probs = cloud_prob_correct[idx_tie] - local_prob_correct[idx_tie]
                    cloud_cost[idx_tie] += 1.2 * diff_probs

            elif method == 3:
                
                cost_local = 1.0 - local_prob_correct
                cost_cloud = 1.0 - cloud_prob_correct
                
                # ---- METHOD=3 (New Logic) ----
                #
                # 1) local=correct, cloud=wrong => local_cost=0, cloud_cost=1
                mask_l1_c0 = local_correct_mask & (~cloud_correct_mask)
                idx = mask_l1_c0.nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    local_cost[idx] = 0.0
                    cloud_cost[idx] = 1.0

                # 2) local=wrong, cloud=correct => local_cost=1, cloud_cost=0
                mask_l0_c1 = (~local_correct_mask) & cloud_correct_mask
                idx = mask_l0_c1.nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    local_cost[idx] = 1.0
                    cloud_cost[idx] = 0.0

                # 3) both wrong => local_cost=1 - local_prob, cloud_cost=1 - (cloud_prob - local_prob)
                mask_l0_c0 = (~local_correct_mask) & (~cloud_correct_mask)
                idx = mask_l0_c0.nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    lp = local_prob_correct[idx]
                    cp = cloud_prob_correct[idx]
                    local_cost[idx] = 1.0 - lp
                    cloud_cost[idx] = 1.0 - cp   # = 1 + lp - cp

                # 4) both correct => tie => check local entropy
                diff_margins= cloud_prob_correct- local_prob_correct

                

                # 2) Βρίσκουμε ποιες είναι "ισοπαλία" (|marginLocal - marginCloud| < tie_threshold)
                tie_mask = (diff_margins < tie_threshold)
                if tie_mask.any():
                    # Υπολογίζουμε εντροπία local και cloud για εκείνα τα δείγματα
                    local_ent = calculate_normalized_entropy(local_out[tie_mask])
                    cloud_ent = calculate_normalized_entropy(cloud_out[tie_mask])

                    # όποιο έχει μικρότερη ent => πιο "σίγουρο" => δίνουμε λίγο χαμηλότερο cost
                    cheaper_local_mask = (local_ent < entropy_threshold)  # boolean mask *μέσα* στο tie
                    # cheaper_cloud_mask = ~cheaper_local_mask

                    # Φτιάχνουμε δείκτες στο tie_mask
                    tie_indices = tie_mask.nonzero(as_tuple=True)[0]

                    idx_local_cheaper = tie_indices[cheaper_local_mask]
                    idx_cloud_cheaper = tie_indices[~cheaper_local_mask]

                    # Ρίχνουμε λίγο το cost_local σε όσα local_ent < cloud_ent
                    cost_local[idx_local_cheaper] -= improvement_scale
                    # Αλλιώς ρίχνουμε λίγο το cost_cloud
                    cost_cloud[idx_cloud_cheaper] -= improvement_scale

                    # Κόψε τυχόν αρνητικά cost στο 0
                    cost_local = torch.clamp(cost_local, min=0.0)
                    cost_cloud = torch.clamp(cost_cloud, min=0.0)
                    local_cost = cost_local
                    cloud_cost = cost_cloud
                
                # mask_l1_c1 = local_correct_mask & cloud_correct_mask
                # idx = mask_l1_c1.nonzero(as_tuple=True)[0]
                # if len(idx) > 0:
                #     local_out_tie = local_out[idx]
                #     local_entropy = calculate_normalized_entropy_from_logits(local_out_tie)
                #     # If local_entropy < threshold => local=0,cloud=1
                #     cond_mask = (local_entropy < entropy_threshold)
                #     idx_true = idx[cond_mask]
                #     local_cost[idx_true] = 0.0
                #     cloud_cost[idx_true] = 1.0

                #     # else => local=1, cloud=0
                #     idx_false = idx[~cond_mask]
                #     local_cost[idx_false] = 1.0
                #     cloud_cost[idx_false] = 0.0
                    
            elif method == 4:
                # 1) Βρίσκουμε top2 για local και cloud
                p1_local, p2_local = get_top2_probs(local_probs)
                p1_cloud, p2_cloud = get_top2_probs(cloud_probs)

                marginLocal = p1_local - p2_local
                marginCloud = p1_cloud - p2_cloud

                # 2) cost_local = 1 - marginLocal
                #    cost_cloud = 1 - marginCloud
                cost_local = 1.0 - marginLocal
                cost_cloud = 1.0 - marginCloud

                local_cost = cost_local
                cloud_cost = cost_cloud
                

            # ================ ΝΕΟΣ METHOD=5 (Confidence Margin + Tie-break Entropy) ================
            elif method == 5:
                # 1) Margin-based costs όπως method=4
                p1_local, p2_local = get_top2_probs(local_probs)
                p1_cloud, p2_cloud = get_top2_probs(cloud_probs)

                marginLocal = p1_local - p2_local
                marginCloud = p1_cloud - p2_cloud

                cost_local = 1.0 - marginLocal
                cost_cloud = 1.0 - marginCloud

                # 2) Βρίσκουμε ποιες είναι "ισοπαλία" (|marginLocal - marginCloud| < tie_threshold)
                diff_margins = torch.abs(marginLocal - marginCloud)
                tie_mask = (diff_margins < tie_threshold)

                if tie_mask.any():
                    # Υπολογίζουμε εντροπία local και cloud για εκείνα τα δείγματα
                    local_ent = calculate_normalized_entropy(local_out[tie_mask])
                    cloud_ent = calculate_normalized_entropy(cloud_out[tie_mask])

                    # όποιο έχει μικρότερη ent => πιο "σίγουρο" => δίνουμε λίγο χαμηλότερο cost
                    cheaper_local_mask = (local_ent < entropy_threshold)  # boolean mask *μέσα* στο tie
                    # cheaper_cloud_mask = ~cheaper_local_mask

                    # Φτιάχνουμε δείκτες στο tie_mask
                    tie_indices = tie_mask.nonzero(as_tuple=True)[0]

                    idx_local_cheaper = tie_indices[cheaper_local_mask]
                    idx_cloud_cheaper = tie_indices[~cheaper_local_mask]

                    # Ρίχνουμε λίγο το cost_local σε όσα local_ent < cloud_ent
                    cost_local[idx_local_cheaper] -= improvement_scale
                    # Αλλιώς ρίχνουμε λίγο το cost_cloud
                    cost_cloud[idx_cloud_cheaper] -= improvement_scale

                    # Κόψε τυχόν αρνητικά cost στο 0
                    cost_local = torch.clamp(cost_local, min=0.0)
                    cost_cloud = torch.clamp(cost_cloud, min=0.0)

                local_cost = cost_local
                cloud_cost = cost_cloud
            
            # ================ ΝΕΟΣ METHOD=6 (Confidence Tie + Overconfidence Penalty) ================
            elif method == 6:
                # Υπολογισμός βασικού κόστους όπως στο αρχικό
                local_cost = 1.0 - local_prob_correct
                cloud_cost = 1.0 - cloud_prob_correct

                # Υπολογισμός top2 margins
                p1_local, p2_local = get_top2_probs(local_probs)
                margin_local = p1_local - p2_local

                # 1) Περίπτωση ισοπαλίας (local σωστό & cloud σωστό)
                both_correct_mask = local_correct_mask & cloud_correct_mask
                idx_both_correct = both_correct_mask.nonzero(as_tuple=True)[0]
                if len(idx_both_correct) > 0:
                    # Αν margin_local >= 0.5 => confidence στο local => μειώνουμε cloud
                    margin_local_both = margin_local[idx_both_correct]
                    confident_local_mask = margin_local_both >= 0.5
                    idx_confident_local = idx_both_correct[confident_local_mask]
                    if len(idx_confident_local) > 0:
                        penalty = 1.4 * (cloud_prob_correct[idx_confident_local] - local_prob_correct[idx_confident_local])
                        cloud_cost[idx_confident_local] += penalty

                # 2) Περίπτωση local λάθος & cloud σωστό ή και τα δύο λάθος
                # (i) Local λάθος, Cloud σωστό
                mask_l0_c1 = (~local_correct_mask) & cloud_correct_mask
                idx_l0_c1 = mask_l0_c1.nonzero(as_tuple=True)[0]
                if len(idx_l0_c1) > 0:
                    margin_local_l0_c1 = margin_local[idx_l0_c1]
                    overconf_mask = margin_local_l0_c1 >= 0.5
                    idx_overconf = idx_l0_c1[overconf_mask]
                    if len(idx_overconf) > 0:
                        local_cost[idx_overconf] += 0.7 * local_prob_correct[idx_overconf]

                # (ii) Local λάθος, Cloud λάθος
                mask_l0_c0 = (~local_correct_mask) & (~cloud_correct_mask)
                idx_l0_c0 = mask_l0_c0.nonzero(as_tuple=True)[0]
                if len(idx_l0_c0) > 0:
                    margin_local_l0_c0 = margin_local[idx_l0_c0]
                    overconf_mask = margin_local_l0_c0 >= 0.5
                    idx_overconf = idx_l0_c0[overconf_mask]
                    if len(idx_overconf) > 0:
                        local_cost[idx_overconf] += 0.7 * local_prob_correct[idx_overconf]

            # 5) bk = local_cost - cloud_cost
            bks = local_cost - cloud_cost
            all_bks_list.extend(bks.cpu().numpy())

            
            all_features_list.extend(local_feats.cpu().numpy())

            all_labels_list.extend(labels.cpu().numpy())
            all_logits_list.extend(logitsL.cpu().numpy())
            all_images_list.extend(images.cpu())
    return all_features_list, all_bks_list, all_labels_list,all_logits_list,all_images_list


# ============================================================================
# ORACLE DECISION FUNCTION
# ============================================================================

def my_oracle_decision_function(local_out, cloud_out, labels, b_star=0.0, not_binary_decision=False):
    """
    Oracle decision function with optional continuous scoring mode.
    
    Parameters
    ----------
    local_out : Tensor (B, num_classes)
        Local classifier logits
    cloud_out : Tensor (B, num_classes)
        Cloud classifier logits
    labels : Tensor (B,)
        Ground truth labels
    b_star : float
        Threshold for binary decision (used only when not_binary_decision=False)
    not_binary_decision : bool
        • False (default) → returns binary {0,1} based on b_star
        • True → returns continuous score for ranking:
            -1  if only local correct
            +1  if only cloud correct
            bk  if tie (both correct or both wrong)
    
    Returns
    -------
    decisions : Tensor (B,)
        • Binary (0/1) if not_binary_decision=False
        • Continuous scores if not_binary_decision=True
    """
    batch_size = labels.size(0)
    local_probs = F.softmax(local_out, dim=1)
    cloud_probs = F.softmax(cloud_out, dim=1)

    local_pred = local_out.argmax(dim=1)
    cloud_pred = cloud_out.argmax(dim=1)

    local_correct_mask = (local_pred == labels)
    cloud_correct_mask = (cloud_pred == labels)

    # Compute bk (cost difference)
    local_prob_correct = local_probs[range(batch_size), labels]
    cloud_prob_correct = cloud_probs[range(batch_size), labels]
    cost_local = 1.0 - local_prob_correct
    cost_cloud = 1.0 - cloud_prob_correct
    bk = cost_local - cost_cloud

    if not_binary_decision:
        # Return continuous scores
        scores = torch.zeros(batch_size, device=labels.device, dtype=torch.float32)
        
        for i in range(batch_size):
            lc_ok = local_correct_mask[i].item()
            cc_ok = cloud_correct_mask[i].item()
            
            if lc_ok and not cc_ok:
                # Only local correct → prefer local strongly
                scores[i] = -1.0
            elif cc_ok and not lc_ok:
                # Only cloud correct → prefer cloud strongly
                scores[i] = +1.0
            else:
                # Tie (both correct or both wrong) → use bk
                scores[i] = bk[i]
        
        return scores
    
    else:
        # Binary decision based on b_star
        sc_list = []
        for i in range(batch_size):
            lc_ok = float(local_correct_mask[i].item())
            cc_ok = float(cloud_correct_mask[i].item())

            if lc_ok == 1 and cc_ok == 0:
                sc = -1.0
            elif lc_ok == 0 and cc_ok == 1:
                sc = +1.0
            elif lc_ok == 0 and cc_ok == 0:
                sc = bk[i].item()
            else:  # both correct
                sc = bk[i].item()

            sc_list.append(sc)

        sc_tensor = torch.tensor(sc_list, device=labels.device)
        decisions = (sc_tensor >= b_star).long()
        return decisions
