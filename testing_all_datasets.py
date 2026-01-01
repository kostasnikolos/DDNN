import torch

import torch.nn as nn                                   # import the layers of neural network library
import torch.optim as optim                             # import the torch opzimer library
import torch.nn.functional as F                         # import the activation functions library
from torch.utils.data import Dataset,DataLoader, random_split   # import a data optimizer 
import torchvision.transforms as transforms             # import a data transformer
from torchvision import datasets                        # import the dataset librry
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
from scipy.stats import pearsonr
import math
from typing import Dict, Tuple, List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.data import Subset
# Set device to GPU for faster training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_LABELS = [
  "airplane","automobile","bird","cat","deer",
  "dog","frog","horse","ship","truck"
]
from torch.nn.functional import softmax

from torch.optim.lr_scheduler import ReduceLROnPlateau
NUM_CLASSES = 10  # CIFAR-100 has 100 classes , CIFAR-10 has 10 classes
# Define the Local CNN Feature Extractor
class LocalFeatureExtractor(nn.Module):
    """
        Creates local features from the input image using a basic CNN Block
        Conv2D-> Batch Normilization -> Leaky Relu -> Dropout
        The output shape for CIFAR-10 input (3x32x32) after one polling is (32,16,16)
    
    """
    def __init__(self):
        super(LocalFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 32 filters of size 3x3
        self.bn = nn.BatchNorm2d(32)  # Batch normalization for conv1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Add pooling layer to reduce spatial dimensions by half
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x = F.relu(self.bn(self.conv1(x)))  # Apply convolution, batch normalization, then ReLU activation
        x = F.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.01)
        x = self.pool(x)  # Apply max pooling
        x = self.dropout(x)  # Apply dropout for regularization
        return x             

# Define the Local CNN Classifier
class LocalClassifier(nn.Module):
    """
    Acts as the local classifier on top of the local feature extractor.
    It flattens the feature map then uses 1FC Layer to classify the image.
    It outputs 10 logits for the CIFAR-10 classes

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(LocalClassifier, self).__init__()
        # Fully connected layer to classify based on the local features
        self.fc1 = nn.Linear(32 * 16 * 16, 64)  # 32 channels, 32x32 feature maps, 10 output classes
        
        self.fc2 = nn.Linear(64, NUM_CLASSES)
    def forward(self, x):

        x = x.view(-1, 32 * 16 * 16)  # Flatten the feature map before feeding into fully connected layer
        
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc2(x)
        return x

# Define the Cloud CNN
class CloudCNN(nn.Module):
    """
    The cloud CNN takes the local features from LocalFeatureExtractor as input
    ( the output shape (batch,32,16,16) ) and further processes them
    with additional convolutional layers for deeper classification.
    Then  does multiple conv blocks and final FC layer, outputing 10 logits for the classes.
    """
    def __init__(self):
        super(CloudCNN, self).__init__()
        # Convolutional layers with dropout and batch normalization
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for conv1
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization for conv2
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization for conv3
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # Batch normalization for conv4
        self.dropout4 = nn.Dropout(p=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 *  2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)  # Apply Leaky ReLU after conv and batch normalization
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = self.dropout4(x)
        x = self.pool3(x)
        # print(f" print input shpate before flattening: {x.shape}")  # Debug p

        x = x.view(-1, 256 * 2 * 2)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.fc3(x)
        return x
# Define The Offline Mechanism CNNclass OffloadMechanism(nn.Module):

        
def load_data(batch_size=128, dataset='cifar10'):
    """
    Returns (train_loader, val_loader, test_loader)
    Supported datasets:
        cifar10, cifar100, cinic10, svhn, gtsrb32
    """
    # ---------- 1. κοινά transforms ---------- #
    normalize_3 = transforms.Normalize((0.5, 0.5, 0.5),
                                       (0.5, 0.5, 0.5))

    tf_augm = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # κρατά 32×32
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize_3
    ])
    tf_plain = transforms.Compose([
        transforms.ToTensor(),
        normalize_3
    ])

    data_root = './data'  # ρίζα όλων των σετ

    # ---------- 2. επιλογή dataset ---------- #
    if dataset == 'cifar10':
        train_set = datasets.CIFAR10(data_root, train=True,
                                     download=True, transform=tf_augm)
        test_set  = datasets.CIFAR10(data_root, train=False,
                                     download=True, transform=tf_plain)

    elif dataset == 'cifar100':
        train_set = datasets.CIFAR100(data_root, train=True,
                                      download=True, transform=tf_augm)
        test_set  = datasets.CIFAR100(data_root, train=False,
                                      download=True, transform=tf_plain)

    # ---------------- CINIC-10 ---------------- #
    elif dataset == 'cinic10':
        # data/cinic10/{train,valid,test}/<class>/*.png
        base = os.path.join(data_root, 'cinic10')
        train_set = datasets.ImageFolder(os.path.join(base, 'train'),
                                         tf_augm)
        val_set   = datasets.ImageFolder(os.path.join(base, 'valid'),
                                         tf_plain)   # έτοιμο split
        test_set  = datasets.ImageFolder(os.path.join(base, 'test'),
                                         tf_plain)

    # ---------------- SVHN ---------------- #
    elif dataset == 'svhn':
        # χρησιμοποιούμε ΜΟΝΟ το officiel train split - 73 k δείγματα
        train_set = datasets.SVHN(data_root, split='train',
                                  download=True, transform=tf_augm)
        test_set  = datasets.SVHN(data_root, split='test',
                                  download=True, transform=tf_plain)
        # 10 % → validation
        val_len = int(len(train_set) * 0.1)
        train_len = len(train_set) - val_len
        train_set, val_set = random_split(train_set, [train_len, val_len])

    # ---------------- GTSRB-32 ---------------- #
    elif dataset == 'gtsrb32':
        # κατεβάζεις από Kaggle: data/gtsrb32/{train,test}/<class>/*.png
        base = os.path.join(data_root, 'gtsrb32')
        full_train = datasets.ImageFolder(os.path.join(base, 'train'),
                                          tf_augm)
        test_set   = datasets.ImageFolder(os.path.join(base, 'test'),
                                          tf_plain)
        # 10 % → validation
        val_len = int(len(full_train) * 0.1)
        train_len = len(full_train) - val_len
        train_set, val_set = random_split(full_train, [train_len, val_len])

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    # ---------- 3. validation split (όπου δεν υπάρχει έτοιμο) ---------- #
    if dataset not in ('cinic10', 'svhn', 'gtsrb32'):
        val_len = int(len(train_set) * 0.1)
        train_len = len(train_set) - val_len
        train_set, val_set = random_split(train_set, [train_len, val_len])

    # ---------- 4. DataLoaders ---------- #
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader




def initialize_DDNN_models():
    """_summary_
        Initializes the DDNN models and returns them. the local feature extractor , the local classifier and the cloud.
    """
    local_feature_extractor = LocalFeatureExtractor().to(device)
    local_classifier = LocalClassifier().to(device)
    cloud_cnn = CloudCNN().to(device)
    return local_feature_extractor, local_classifier, cloud_cnn


def initialize_offload_model():
    """_summary_
    Initializes the offload mechanism and returns it
        _
    """
    offload_mechanism = DeepOffloadMechanism().to(device)
    return  offload_mechanism

# Initialize models
def initialize_models():
    """
        Initializes all models, the DDNN models (local_feature_extractor,local_classifier,cloud), the offload mechanism model and returns them
    
    """
    local_feature_extractor = LocalFeatureExtractor().to(device)
    local_classifier = LocalClassifier().to(device)
    cloud_cnn = CloudCNN().to(device)
    offload_mechanism = DeepOffloadMechanism().to(device)
    return local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism

# Initialize optimizers
def initialize_optimizers(local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism ,lr=0.001):
    """
    Initializes all optimizers, DDNN optimizers for local_feature_extractor,local_classifier,cloud, offload mechanism model optimizer and returns them

    """
    cnn_optimizer = optim.Adam(list(local_feature_extractor.parameters()) + list(local_classifier.parameters()) + list(cloud_cnn.parameters()), lr=lr)
    offload_optimizer = optim.Adam(offload_mechanism.parameters(),0.005 ) #working with 0.000000001

    return  cnn_optimizer, offload_optimizer

def initialize_DDNN_optimizers(local_feature_extractor, local_classifier, cloud_cnn, lr=0.001):
    """
        Initializes DDNN optimizers
    """
    cnn_optimizer = optim.Adam(list(local_feature_extractor.parameters()) + list(local_classifier.parameters()) + list(cloud_cnn.parameters()), lr=lr)
    return  cnn_optimizer

def initialize_offload_optimizer(offload_mechanism, Ir=0.001):
    """ Initializes offload mechanism optimizer"""
    offload_optimizer = optim.Adam(offload_mechanism.parameters(),0.005 ) #καλο το 0.005

    return   offload_optimizer


# Training step
# Train both Local and Cloud CNN on all samples , caluclate and save Bk's ,local features
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
def test_DDNN_with_entropy(local_feature_extractor, local_classifier,
                           cloud_cnn, data_loader,
                           target_local_percent,
                           tolerance=0.5,
                           device='cuda'):
    """
    Evaluate the DDNN using an entropy‑based decision rule.
    The function computes *normalized* entropy (range 0‑1) for each local
    prediction, then selects the entropy threshold that yields
    `target_local_percent` (± tolerance) of samples processed locally.

    Args
    ----
    local_feature_extractor, local_classifier, cloud_cnn : torch.nn.Module
    data_loader            : DataLoader with evaluation set
    target_local_percent   : float, desired percentage of local samples
    tolerance              : float, allowed deviation in percentage points
    device                 : 'cuda' or 'cpu'

    Returns
    -------
    local_acc, cloud_acc, overall_acc : float
    local_classified, cloud_classified : int
    """

    # ------------------------------------------------------------------ #
    # 1) Forward pass once – gather per‑sample data
    # ------------------------------------------------------------------ #
    entropies      = []
    local_ok_mask  = []
    cloud_ok_mask  = []

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Local branch
            feats       = local_feature_extractor(images)
            logits_loc  = local_classifier(feats)
            probs_loc   = torch.softmax(logits_loc, dim=1)

            # *** Normalized entropy ***
            C = logits_loc.shape[1]                        # num classes
            entropy_raw = -torch.sum(probs_loc *
                                     torch.log(probs_loc + 1e-9), dim=1)
            entropy_norm = entropy_raw / math.log(C)       # range [0,1]

            # Cloud branch
            logits_cloud = cloud_cnn(feats)

            # Store to CPU
            entropies.append(entropy_norm.cpu())
            local_ok_mask.append(
                (logits_loc.argmax(1).cpu() == labels.cpu()).int())
            cloud_ok_mask.append(
                (logits_cloud.argmax(1).cpu() == labels.cpu()).int())

    entropies   = torch.cat(entropies).numpy()          # shape (N,)
    local_ok    = torch.cat(local_ok_mask).numpy()      # bool/int
    cloud_ok    = torch.cat(cloud_ok_mask).numpy()
    N           = len(entropies)

    # ------------------------------------------------------------------ #
    # 2) Choose threshold to hit target_local_percent
    # ------------------------------------------------------------------ #
    target_k = int(round(target_local_percent / 100 * N))
    target_k = max(0, min(target_k, N - 1))

    order       = entropies.argsort()
    sorted_ent  = entropies[order]

    k = target_k
    threshold = sorted_ent[k]

    def local_pct(th):       # helper
        return 100.0 * (entropies <= th).mean()

    actual_pct = local_pct(threshold)
    if abs(actual_pct - target_local_percent) > tolerance:
        direction = -1 if actual_pct > target_local_percent else 1
        while 0 <= k + direction < N:
            k += direction
            threshold = sorted_ent[k]
            actual_pct = local_pct(threshold)
            if abs(actual_pct - target_local_percent) <= tolerance:
                break

    # ------------------------------------------------------------------ #
    # 3) Final split & accuracy
    # ------------------------------------------------------------------ #
    is_local          = entropies <= threshold
    local_classified  = int(is_local.sum())
    cloud_classified  = N - local_classified

    local_correct     = int(local_ok[is_local].sum())
    cloud_correct     = int(cloud_ok[~is_local].sum())

    local_acc   = 100.0 * local_correct  / local_classified if local_classified else 0.0
    cloud_acc   = 100.0 * cloud_correct  / cloud_classified if cloud_classified else 0.0
    overall_acc = 100.0 * (local_correct + cloud_correct) / N

    # ------------------------------------------------------------------ #
    # 4) Console log (optional)
    # ------------------------------------------------------------------ #
    print(f"[Entropy] Target local%={target_local_percent:.2f} → "
          f"Threshold={threshold:.4f} (normalized) → Local%={actual_pct:.2f}")
    print(f"Samples Local={local_classified} | Cloud={cloud_classified}")
    print(f"Local Acc={local_acc:.2f} | Cloud Acc={cloud_acc:.2f} "
          f"| Overall Acc={overall_acc:.2f}")
    print(f"Mean normalized entropy: {entropies.mean():.4f}")

    return (local_acc, cloud_acc, overall_acc,
            local_classified, cloud_classified, actual_pct,threshold)


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


def plot_bk_distribution(all_bks, b_star):
    """
    Plot the distribution of bk values and mark the b* threshold.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(all_bks, bins=50, alpha=0.7, color='blue', label='bk values')
    plt.axvline(x=b_star, color='red', linestyle='--', label=f'b* = {b_star:.2f}')
    plt.title('Distribution of bk values')
    plt.xlabel('bk')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def create_3d_data_with_figures(all_bks, all_local_features):
    """
    Δημιουργεί 3D δομή δεδομένων με τη μορφή (index, bk, figure).
    """
    combined_data = []
    
    for i in range(len(all_bks)):
        bk_val = all_bks[i]
        feat_3d = all_local_features[i]      # shape (32,16,16)
        feat_flat = feat_3d.reshape(-1)      # shape (8192,)
        combined_data.append((i, bk_val, feat_flat))
    
    return combined_data



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


def oracle_rule(local_feature_extractor, 
                              local_classifier, 
                              cloud_cnn, 
                              data_loader, 
                              device='cuda'):
    """
    Επιστρέφει ένα dict: {percent_local -> oracle_accuracy}
    για διάφορα ποσοστά τοπικής ταξινόμησης (0..100%), 
    χρησιμοποιώντας το νέο score:
      - score = +1 αν (local=σωστό, cloud=λάθος)
      - score = -1 αν (local=λάθος, cloud=σωστό)
      - score = 0  αν (local=λάθος, cloud=λάθος)
      - score = local_prob - cloud_prob αν (local=σωστό, cloud=σωστό)
    όπου local_prob, cloud_prob οι πιθανότητες σωστής κλάσης (softmax).
    """

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Θα αποθηκεύσουμε για κάθε δείγμα:
    #   score[i], local_correct[i], cloud_correct[i]
    score_list = []
    local_correct_list = []
    cloud_correct_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # === 1) Forward
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)   # shape: (batch, num_classes)
            cloud_out = cloud_cnn(local_feats)

            # === 2) Ποια είναι η "σωστή" πιθανότητα
            local_probs = torch.softmax(local_out, dim=1)  # (batch, num_classes)
            cloud_probs = torch.softmax(cloud_out, dim=1)

            # πιθανότητα για τη σωστή κλάση
            batch_size = labels.size(0)
            local_prob_correct = local_probs[range(batch_size), labels]
            cloud_prob_correct = cloud_probs[range(batch_size), labels]

            # === 3) Είναι σωστό local/cloud;
            local_pred = local_out.argmax(dim=1)
            cloud_pred = cloud_out.argmax(dim=1)
            local_correct_mask = (local_pred == labels)  # bool
            cloud_correct_mask = (cloud_pred == labels)  # bool

            # === 4) Υπολογισμός score για κάθε δείγμα στο batch
            for i in range(batch_size):
                lc_ok = local_correct_mask[i].item()   # 0.0 or 1.0
                cc_ok = cloud_correct_mask[i].item()
                lp = local_prob_correct[i].item()
                cp = cloud_prob_correct[i].item()

                if lc_ok == 1 and cc_ok == 0:
                    sc = +1.0
                elif lc_ok == 0 and cc_ok == 1:
                    sc = -1.0
                elif lc_ok == 0 and cc_ok == 0:
                    sc = 0.0
                else:
                    # lc_ok=1 and cc_ok=1 => tie-break με prob
                    sc = lp - cp
                    sc = +1.0

                score_list.append(sc)
                local_correct_list.append(lc_ok)
                cloud_correct_list.append(cc_ok)

    # => μετατροπή σε PyTorch Tensor (ή NumPy)
    score_tensor = torch.tensor(score_list)
    local_correct_tensor = torch.tensor(local_correct_list)
    cloud_correct_tensor = torch.tensor(cloud_correct_list)

    N = score_tensor.size(0)

    # === 5) Ταξινόμηση δειγμάτων κατά φθίνουσα σειρά score
    sorted_indices = torch.argsort(score_tensor, descending=True)

    # === 6) Δοκιμάζουμε ποσοστά 0..100%
    results = {}
    for p in range(0, 101, 10):
        num_local = int(N * (p / 100.0))
        local_idx = sorted_indices[:num_local]
        cloud_idx = sorted_indices[num_local:]

        # πόσα σωστά από local_idx => local_correct_tensor[local_idx].sum()
        # πόσα σωστά από cloud_idx => cloud_correct_tensor[cloud_idx].sum()
        local_correct_count = local_correct_tensor[local_idx].sum().item()
        cloud_correct_count = cloud_correct_tensor[cloud_idx].sum().item()
        overall_correct = local_correct_count + cloud_correct_count

        overall_acc = overall_correct / N * 100.0
        results[p] = overall_acc

    return results


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

def create_3d_data_deep(bks, feats, logits, imgs, labels, *, input_mode='feat'):
    """
    Builds combined_data = list of tuples:
        (x_representation, bk_value, real_label)

    input_mode
      'feat'   → use feature‑map   tensor (32,16,16)
      'logits' → use logits        tensor (10,)
      'img'/'figure' → use raw image tensor (3,32,32)
    """
    combined_data = []
    for i in range(len(bks)):
        if input_mode == 'feat':
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
        else:
            raise ValueError("input_mode must be 'feat'|'logits'|'img'|'logits_plus'")
        combined_data.append((x_tensor, float(bks[i]), int(labels[i]), feats[i]))
    return combined_data



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



def analyze_correlation_metrics(loader, local_feature_extractor, local_classifier, cloud_cnn, entropy_threshold=0.2):
    """
    This function computes and prints Pearson correlation coefficients between various metrics
    for samples selected as local using the Entropy method. The metrics include:
      - Normalized entropy (from local classifier logits)
      - Margin: (p1 - p2) where p1 is the top probability and p2 is the second highest.
      - Sum_rest: the sum of probabilities for classes 3 to N (e.g., p3+p4+...+p10)
    
    Parameters:
      loader: DataLoader for the dataset to analyze.
      local_feature_extractor, local_classifier, cloud_cnn: The DDNN models.
      entropy_threshold: Threshold used to decide if a sample is local (entropy < threshold).
    """
    import torch
    from torch.nn.functional import softmax
    import numpy as np

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    
    entropy_list = []
    margin_list = []
    sum_rest_list = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            
            # Forward pass
            local_feats = local_feature_extractor(images)
            local_logits = local_classifier(local_feats)
            
            # Compute probabilities and normalized entropy
            local_probs = softmax(local_logits, dim=1)
            epsilon = 1e-5
            num_classes = local_probs.shape[1]
            ent = -torch.sum(local_probs * torch.log(local_probs + epsilon), dim=1) / torch.log(torch.tensor(float(num_classes)))
            
            # Get top-2 probabilities and compute margin
            sorted_probs, _ = torch.sort(local_probs, descending=True, dim=1)
            p1 = sorted_probs[:, 0]
            p2 = sorted_probs[:, 1]
            margin = p1 - p2
            # Compute sum_rest for classes 3 to N
            sum_rest = torch.sum(sorted_probs[:, 2:], dim=1)
            
            # Select samples based on entropy threshold (local if entropy < threshold)
            mask = (ent < entropy_threshold)
            sel_indices = mask.nonzero(as_tuple=True)[0]
            if sel_indices.numel() > 0:
                entropy_list.extend(ent[sel_indices].cpu().numpy().tolist())
                margin_list.extend(margin[sel_indices].cpu().numpy().tolist())
                sum_rest_list.extend(sum_rest[sel_indices].cpu().numpy().tolist())
    
    # Compute Pearson correlations
    if len(entropy_list) > 0 and len(margin_list) > 0:
        corr_entropy_margin, p_val1 = pearsonr(entropy_list, margin_list)
        corr_entropy_sumrest, p_val2 = pearsonr(entropy_list, sum_rest_list)
        corr_margin_sumrest, p_val3 = pearsonr(margin_list, sum_rest_list)
        
        print("Correlation Analysis for local samples (Entropy method):")
        print(f"Pearson correlation between entropy and margin: {corr_entropy_margin:.4f} (p={p_val1:.4g})")
        print(f"Pearson correlation between entropy and sum_rest: {corr_entropy_sumrest:.4f} (p={p_val2:.4g})")
        print(f"Pearson correlation between margin and sum_rest: {corr_margin_sumrest:.4f} (p={p_val3:.4g})")
    else:
        print("Not enough data to compute correlations.")
        
class DeepOffloadMechanism(nn.Module):
    """
    Flexible offload decision network that supports three input modes:
      * 'feat'   : feature map tensor  (B, C, H, W)
      * 'img'    : raw image tensor   (B, 3, H, W)
      * 'logits' : vector logits      (B, D)
    Skip‑concat of the original input is handled automatically; the builder
    updates the 'in_channels' of subsequent Conv layers so that a mismatch
    can never occur.

    Args
    ----
    input_shape  : tuple
        (C, H, W) for 'feat' / 'img',  or  (D,) for 'logits'.
    input_mode   : str
        'feat' (default) | 'img' | 'logits'
    conv_dims    : list[int]
        Output channels per Conv block. Ignored when mode == 'logits'.
    num_layers   : int
        Number of Conv-BN-ReLU layers per block.
    fc_dims      : list[int]
        Hidden sizes of the FC tail; last element must be 1 (binary logit).
    dropout_p    : float
        Dropout probability.
    dropout_idx  : iterable[int]
        Global Conv-layer indices (0‑based) where Dropout is inserted.
    latent_in    : iterable[int]
        Block indices where the original input is concatenated channel‑wise.
        Concatenation is followed by an *automatic* in_channel adjustment.
    """
    def __init__(self,
                 input_shape  = (32, 16, 16),
                 input_mode   = 'feat',
                 conv_dims    = (64, 128, 256),
                 num_layers   = 1,
                 fc_dims      = (256, 128, 1),
                 dropout_p    = 0.25,
                 dropout_idx  = (),
                 latent_in    = ()):
        super().__init__()
        self.mode = input_mode
        self.latent_in = set(latent_in)

        # ------------------------------------------------------------------ #
        # Build convolutional blocks (skipped if mode == 'logits')
        # ------------------------------------------------------------------ #
        self.conv_blocks = nn.ModuleList()
        C0 = input_shape[0] if (self.mode != 'logits'  and self.mode!= 'logits_plus') else None
        in_ch = C0
        layer_counter = 0  # global Conv-layer index

        if input_mode not in ('logits', 'logits_plus'):
            if len(input_shape) != 3:
                raise ValueError("input_shape must be (C,H,W) for 'feat'/'img'")
            H, W = input_shape[1], input_shape[2]

            for blk_id, out_ch in enumerate(conv_dims):
                layers = []
                for _ in range(num_layers):
                    layers += [
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.LeakyReLU(0.01, inplace=True)
                    ]
                    if layer_counter in dropout_idx:
                        layers.append(nn.Dropout(dropout_p))
                    # update trackers
                    in_ch = out_ch
                    layer_counter += 1

                # add block
                self.conv_blocks.append(nn.Sequential(*layers))

                # MaxPool after every block except the last
                if blk_id < len(conv_dims) - 1:
                    self.conv_blocks.append(nn.MaxPool2d(2))
                    H, W = H // 2, W // 2

                # ---- automatic in_channel adjustment for skip‑concat ----
                if blk_id in self.latent_in:
                    in_ch += C0  # next block will receive extra C0 channels

            self.flat_dim = in_ch * H * W
        else:
            # logits mode: no Conv blocks, flat vector input
            if len(input_shape) != 1:
                raise ValueError("input_shape must be (D,) for 'logits'")
            self.flat_dim = input_shape[0]

        # ------------------------------------------------------------------ #
        # Build fully connected tail
        # ------------------------------------------------------------------ #
        self.fc_blocks = nn.ModuleList()
        last_in = self.flat_dim
        for i, hidden in enumerate(fc_dims[:-1]):
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(last_in, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)
            ))
            if i in self.latent_in and self.mode in ('logits', 'logits_plus'):
                last_in = hidden + self.flat_dim
            else:
                last_in = hidden

        self.fc_out = nn.Linear(last_in, fc_dims[-1])

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #
    def forward(self, x):
        if self.mode != 'logits' and self.mode != 'logits_plus':
            x0 = x  # original input for potential skip
            blk_id = 0
            for module in self.conv_blocks:
                x = module(x)
                # Only Conv blocks are counted in blk_id, not MaxPool
                if isinstance(module, nn.Sequential):
                    if blk_id in self.latent_in:
                        # spatial alignment if needed
                        if x0.size(2) != x.size(2):
                            x0_resized = F.adaptive_avg_pool2d(x0, x.shape[2:])
                        else:
                            x0_resized = x0
                        x = torch.cat([x, x0_resized], dim=1)
                    blk_id += 1
            x = torch.flatten(x, 1)
            for block in self.fc_blocks:     
                x = block(x)     
        else:
            x = x.view(x.size(0), -1)
            x0 = x
            for i, block in enumerate(self.fc_blocks):
                x = block(x)
                if i in self.latent_in and self.mode in ('logits', 'logits_plus'):
                    x = torch.cat([x, x0], dim=1)
        

        return self.fc_out(x)
# Offload dataset class that returns features , offload decision as label , real sample label
class OffloadDatasetCNN(Dataset):
    """
    Dataset that feeds the Deep-offload decision network.

    Parameters
    ----------
    combined_data : list[tuple]
        Each tuple should contain
            (x_repr, bk_val, true_label, feat_tensor)
        where `x_repr` is either raw logits or features depending on `input_mode`.
    b_star : float
        Threshold on bk used in the *pure* rule.
    input_mode : {'feat', 'logits', 'logits_plus'}
        How the decision network will consume the input.
    include_bk : bool
        If True, `__getitem__` also returns the scalar bk_val.
    filter_mask : np.ndarray[bool] | None
        Optional boolean mask to select a subset of combined_data.
    use_oracle_labels : bool
        • False (default) → label = 1  if bk ≥ b_star  else 0      (old behaviour)
        • True            → oracle-style rule (see docstring).
    local_clf, cloud_clf : torch.nn.Module | None
        Required *only* when use_oracle_labels=True.  Should be eval()-mode.
    device : str
        Device where the two classifiers will be executed ('cpu' or 'cuda').
    """
    def __init__(
            self,
            combined_data,
            b_star,
            *,
            input_mode: str = 'feat',
            include_bk: bool = False,
            filter_mask=None,
            use_oracle_labels: bool = False,
            local_clf=None,
            cloud_clf=None,
            device: str = 'cpu'
        ):
        if filter_mask is not None:
            self.combined_data = [combined_data[i] for i, f in enumerate(filter_mask) if f]
        else:
            self.combined_data = combined_data

        self.b_star       = b_star
        self.mode         = input_mode
        self.include_bk   = include_bk
        self.use_oracle   = use_oracle_labels
        self.local_clf    = local_clf
        self.cloud_clf    = cloud_clf
        self.device       = device

        if self.use_oracle:
            assert local_clf and cloud_clf, "Need both classifiers"
            local_clf.eval();  cloud_clf.eval()

            oracle_lbls = []
            with torch.no_grad():
                for _, bk, y_true, feat in self.combined_data:
                    # tensor to device
                    feat_t = torch.tensor(feat, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                    loc_ok = (local_clf(feat_t).argmax().item()  == y_true)
                    cld_ok = (cloud_clf(feat_t).argmax().item()  == y_true)

                    if   loc_ok and not cld_ok: oracle_lbls.append(0.0)
                    elif cld_ok and not loc_ok: oracle_lbls.append(1.0)
                    else:                       oracle_lbls.append(
                                                  1.0 if bk >= b_star else 0.0)

            self.oracle_labels = torch.tensor(oracle_lbls, dtype=torch.float32)
        else:
            self.oracle_labels = None   

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.combined_data)

    

    # ------------------------------------------------------------
    def __getitem__(self, idx):
        x_repr, bk_val, true_lbl, feat_tensor = self.combined_data[idx]

        # -------------------- label assignment -------------------
        # ➋ Label assignment -------------------------------------------
        if self.oracle_labels is not None:
            label_offload = self.oracle_labels[idx]

        else:
            # classic bk-threshold rule
            label_offload = 1.0 if bk_val >= self.b_star else 0.0

        # -------------------- build tensors ----------------------
        if isinstance(label_offload, torch.Tensor):
            y_offload = label_offload.clone().detach().float()
        else:
            y_offload = torch.tensor(label_offload, dtype=torch.float32)
        # y_offload = torch.tensor(label_offload, dtype=torch.float32)
        y_cifar   = torch.tensor(true_lbl,     dtype=torch.long)

        if self.mode == 'feat':
            x_tensor = torch.as_tensor(feat_tensor, dtype=torch.float32)

        elif self.mode == 'logits':
            if isinstance(x_repr, torch.Tensor):
                x_tensor = x_repr.clone().detach().float()
            else:
                x_tensor = torch.tensor(x_repr,     dtype=torch.float32)
                
        elif self.mode == 'logits_plus':
            if isinstance(x_repr, torch.Tensor):
                x_tensor = x_repr.clone().detach().float()
            else:
                x_tensor = torch.tensor(x_repr,     dtype=torch.float32)
        else:
            raise ValueError("input_mode must be 'feat'|'logits'|'logits_plus'")

        if self.include_bk:
            return x_tensor, y_offload, y_cifar,torch.tensor(feat_tensor, dtype=torch.float32), torch.tensor(bk_val, dtype=torch.float32)
        else:
            return x_tensor, y_offload, y_cifar,torch.tensor(feat_tensor, dtype=torch.float32)

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
    epochs=50,
    lr=1e-4,
    stop_threshold=0.9,
    patience=20  # Number of epochs to wait for early stopping
):
    """
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
    # === Create models directory if it doesn't exist and set checkpoint path ===
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "best_offload_mechanism.pth")

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
            labels = labels.unsqueeze(1).to(device)  # shape: (batch,1)

            # Forward pass: compute logits
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
                input_mode=input_mode   
            )
            offload_val_acc = evaluate_offload_decision_accuracy_CNN_test(
                offload_mechanism, local_feature_extractor, local_classifier, cloud_cnn, val_loader, b_star,threshold,input_mode=input_mode
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
        if (offload_train_acc / 100.0) >= stop_threshold:
            print(f"Stopping early because train accuracy {offload_train_acc:.2f}% >= {stop_threshold*100:.2f}%")
            break

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement. Best validation accuracy: {best_val_acc:.2f}%")
            # Load best model from models directory
            offload_mechanism.load_state_dict(torch.load(checkpoint_path))
            break

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Offload Train Accuracy: {offload_train_acc:.2f}%, Offload Val Accuracy: {offload_val_acc:.2f}%, Learning Rate:{current_lr}")
    
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

def evaluate_offload_decision_accuracy_CNN_train(loader, local_feature_extractor, local_classifier, cloud_cnn, deep_offload_model, b_star,threshold=0.5, *, input_mode='feat', device='cuda'):
    """
    Evaluate the offload decision accuracy of the deep offload mechanism on the given dataset.
    This function:
      1. Computes a ground truth (gt) based on the DDNN predictions (using local_classifier and cloud_cnn)
         via the ORACLE labeling (not just bk threshold), and compares it with the provided labels from the offload dataset.
      2. Passes the unflattened local features through the deep offload mechanism to get its predictions
         and compares those with the computed gt.
    Returns:
      The offload decision accuracy (percentage) based on the deep offload mechanism.
    """
    correct_offload = 0
    total_samples = 0
    correct_gt_vs_loader = 0  # Accuracy between computed ground truth and loader labels
    
    deep_offload_model.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    with torch.no_grad():
        for x_tensor, offload_label, real_label, feature_tensor in loader:
            x_tensor   = x_tensor.to(device)
            feature_tensor = feature_tensor.to(device)
            offload_label = offload_label.to(device)
            real_label    = real_label.to(device)
            bs = real_label.size(0)

            # === 1) Υπολογίζεις local_probs, cloud_probs βάσει του ΠΡΑΓΜΑΤΙΚΟΥ label ===
            # if we are in logits mode we already have them no need to compute them again
            if input_mode == 'logits':
                local_logits = x_tensor
            elif input_mode == 'logits_plus':
                # we already have them bu we need to cut the first 10 dim because now its 12
                local_logits= x_tensor[:,:NUM_CLASSES]
            else:
                local_logits = local_classifier(feature_tensor)
            
            
            
            cloud_logits = cloud_cnn(feature_tensor)

            # === 1b) Oracle labeling ===
            computed_gt = my_oracle_decision_function(local_logits, cloud_logits, real_label, b_star=b_star).float()

            # === 2) Έλεγξε αν η stored offload_label ταιριάζει με computed_gt ===
            # (ιδανικά ~100%)
            # ψιλο unsused αλλα οκευ
            correct_gt_vs_loader += (computed_gt == offload_label).sum().item()
            # if not torch.equal(computed_gt,offload_label): print('Εχουμε θεμα  στην evaluate_offload_decision_train')
            
            # === 3) Δώσε local_feats στο deep_offload_model => offload απόφαση
            logits_offload = deep_offload_model(x_tensor)
            pred_offload   = (torch.sigmoid(logits_offload).squeeze(1) > threshold).float()

            # Σύγκρινε pred_offload με computed_gt
            correct_offload += (pred_offload == computed_gt).sum().item()
            total_samples   += bs

    gt_vs_loader_acc = 100.0 * correct_gt_vs_loader / total_samples
    offload_acc      = 100.0 * correct_offload / total_samples
    # print(f"Ground Truth vs OffloadDataset label accuracy: {gt_vs_loader_acc:.2f}%")
    # print(f"Deep Offload CNN decision accuracy vs ground truth: {offload_acc:.2f}%")
    return offload_acc

def my_oracle_decision_function(local_out, cloud_out, labels, b_star=0.0):
    """
    Oracle that matches the logic used to label offload_dataset:
    If both models agree in correctness (or are both wrong/right),
    prefer the one with lower cost (1 - confidence in correct class).

    Returns 0 = local, 1 = cloud
    """
    batch_size = labels.size(0)
    local_probs = F.softmax(local_out, dim=1)
    cloud_probs = F.softmax(cloud_out, dim=1)

    local_pred = local_out.argmax(dim=1)
    cloud_pred = cloud_out.argmax(dim=1)

    local_correct_mask = (local_pred == labels)  # bool
    cloud_correct_mask = (cloud_pred == labels)  # bool

    sc_list = []
    for i in range(batch_size):
        lc_ok = float(local_correct_mask[i].item())
        cc_ok = float(cloud_correct_mask[i].item())
        true_lbl = labels[i].item()
        loc_conf = local_probs[i, int(true_lbl)]
        cld_conf = cloud_probs[i, int(true_lbl)]

        # Define costs (same as used for bk)
        cost_local = 1.0 - loc_conf
        cost_cloud = 1.0 - cld_conf

        # Default: decide based on lower cost (same as bk)
        sc = cost_local - cost_cloud  # >0 → prefer cloud

        # Alternative logic (commented out):
        if lc_ok == 1 and cc_ok == 0:
            sc = -1.0
        elif lc_ok == 0 and cc_ok == 1:
            sc = +1.0
        elif lc_ok == 0 and cc_ok == 0:
            sc = cost_local - cost_cloud  # >0 → prefer cloud
        else:  # both correct
            sc = cost_local - cost_cloud  # >0 → prefer cloud

        sc_list.append(sc)

    sc_tensor = torch.tensor(sc_list, device=labels.device)

    # cloud if sc >= b_star ⇒ return 1 (offload)
    # local otherwise ⇒ return 0
    decisions = (sc_tensor >= b_star).long()
    return decisions


def evaluate_offload_decision_accuracy_CNN_test(
    offload_mechanism,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,  
    test_loader,
    b_star,
    threshold=0.5,
    *, input_mode='feat', device='cuda'
):
    """
    Evaluates the offload decisions on test data by comparing:
      1) "Real" label from the oracle labeling (uses correctness of local/cloud and bk threshold for ties)
      2) "Predicted" label from the offload mechanism (if offload_prob > threshold => 1, else 0)

    We compute bk = local_cost - cloud_cost for each test sample, then generate the "real" label
    based on whether bk >= b_star. Meanwhile, the offload mechanism predicts 1/0 via the logistic
    output. If they match, it's a correct decision.

    Args:
        offload_mechanism (nn.Module): The trained offload MLP.
        local_feature_extractor (nn.Module): Local feature extractor.
        local_classifier (nn.Module): Local classifier (for cost).
        cloud_cnn (nn.Module): Cloud classifier (for cost).
        test_loader (DataLoader): Test dataset loader (images, labels).
        b_star (float): The threshold for manually labeling each sample (bk >= b_star => label=1).
        threshold (float): Offload mechanism classification threshold (default=0.5).
        device (str): 'cuda' or 'cpu'.

    Returns:
        offload_accuracy (float): The percentage of correctly matched decisions between
                                  oracle labeling and offload mechanism output.
    """

    # Put models in eval mode
    offload_mechanism.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    correct_decisions = 0
    total_samples = 0

    # We won't backprop, so use no_grad
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size

            # 1) Extract local features
            local_feats = local_feature_extractor(images)

            # 2) Compute local and cloud logits
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # 3) Oracle labeling (0=local, 1=cloud)
            oracle_labels = my_oracle_decision_function(local_out, cloud_out, labels, b_star=b_star).float()

            # 4) Offload mechanism prediction
            if input_mode == 'logits':
                logits = offload_mechanism(local_out)
            elif input_mode == 'img':
                logits = offload_mechanism(images)         # shape (batch, 1)
            elif input_mode== 'feat': 
                logits = offload_mechanism(local_feats)
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                top2  = torch.topk(probs, 2, dim=1).values
                margin  = (top2[:,0] - top2[:,1]).unsqueeze(1)        # (B,1)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1,keepdim=True) \
                        / math.log(probs.size(1))
                dom_in = torch.cat([local_out, margin, entropy], dim=1)  # (B,12)
                logits = offload_mechanism(dom_in)
            probs = torch.sigmoid(logits).squeeze(1)
            predicted_label = (probs > threshold).float()

            # 5) Compare oracle_labels vs predicted_label
            correct_decisions += (oracle_labels == predicted_label).sum().item()

    offload_accuracy = (correct_decisions / total_samples) * 100.0
    # print(f"[evaluate_offload_decision_accuracy_CNN_test] Accuracy: {offload_accuracy:.2f}% "
        #   f"(Threshold={threshold}, b_star={b_star:.3f})")
    return offload_accuracy
def test_DDNN_with_optimized_rule(
    offload_mechanism,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    test_loader,
    b_star,
    threshold=0.5,
    input_mode= 'feat',
    device='cuda'
):
    """
    Evaluates the offload decisions on test data by comparing:
      1) "Real" label from oracle labeling (uses correctness of local/cloud and bk threshold for ties)
      2) "Predicted" label from the offload mechanism (if offload_prob > threshold => 1, else 0)

    Additionally, we now *use* the offload decision to classify each sample either locally (label=0)
    or via the cloud (label=1), measure the classification accuracy, and print:

      - Overall accuracy (percentage of samples correctly classified overall)
      - Local accuracy (percentage of local samples that are correct)
      - Cloud accuracy (percentage of cloud samples that are correct)
      - Percentage of samples that ended up local

    Returns:
        local_percentage, overall_acc
    """

    # Put models in eval mode
    offload_mechanism.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Variables for matching "real" label vs "predicted" label of offload (decision correctness)
    correct_decisions = 0
    total_samples = 0

    # Variables for measuring classification accuracy based on offload decision
    local_correct = 0
    local_total = 0
    cloud_correct = 0
    cloud_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size

            # 1) Extract local features
            local_feats = local_feature_extractor(images)

            # 2) Compute local and cloud logits
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # 3) Oracle labeling (0=local, 1=cloud)
            oracle_labels = my_oracle_decision_function(local_out, cloud_out, labels, b_star=b_star).float()

            # 4) Offload mechanism prediction
            if input_mode == 'logits':
                dom_in = local_out
            elif input_mode == 'img':
                dom_in = images
            elif input_mode == 'logits_plus':
                probs = F.softmax(local_out, dim=1)
                top2 = torch.topk(probs, 2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
                entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True) / math.log(probs.size(1))
                dom_in = torch.cat([local_out, margin, entropy], dim=1)
            else:
                dom_in = local_feats
            dom_logits = offload_mechanism(dom_in)
            dom_probs = torch.sigmoid(dom_logits).squeeze(1)
            predicted_label = (dom_probs > threshold).float()

            # 5) Compare oracle_labels vs predicted_label
            correct_decisions += (oracle_labels == predicted_label).sum().item()

            # 6) Use predicted_label to classify each sample locally (0) or via cloud (1)
            local_mask = (predicted_label == 0)
            cloud_mask = (predicted_label == 1)

            # Classify the local samples
            if local_mask.any():
                local_outputs = local_classifier(local_feats[local_mask])
                local_preds = local_outputs.argmax(dim=1)
                local_correct += (local_preds == labels[local_mask]).sum().item()
                local_total += local_mask.sum().item()

            # Classify the cloud samples
            if cloud_mask.any():
                cloud_outputs = cloud_cnn(local_feats[cloud_mask])
                cloud_preds = cloud_outputs.argmax(dim=1)
                cloud_correct += (cloud_preds == labels[cloud_mask]).sum().item()
                cloud_total += cloud_mask.sum().item()

    offload_accuracy = (correct_decisions / total_samples) * 100.0
    local_acc = (local_correct / local_total * 100.0) if local_total > 0 else 0.0
    cloud_acc = (cloud_correct / cloud_total * 100.0) if cloud_total > 0 else 0.0
    overall_acc = (local_correct + cloud_correct) / total_samples * 100.0
    local_percentage = (local_total / total_samples * 100.0)

    print(f"[evaluate_offload_decision_accuracy_CNN_test] Decision Accuracy: {offload_accuracy:.2f}% "
          f"(Threshold={threshold}, b_star={b_star:.3f})")
    print(f"[Classification Results] Overall Accuracy: {overall_acc:.2f}%")
    print(f"[Classification Results] Local Accuracy: {local_acc:.2f}%, Local samples: {local_total}")
    print(f"[Classification Results] Cloud Accuracy: {cloud_acc:.2f}%, Cloud samples: {cloud_total}")
    print(f"[Classification Results] Percentage Local: {local_percentage:.2f}%")

    return local_percentage, overall_acc
def test_difficulty_vs_L0(
    L0_values,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    train_loader,
    test_loader,
    val_loader,
    device='cuda',
    offload_epochs=70,
    batch_size=128,
    input_mode='logits',
    dataset_name= 'cifar10'
):
    """
    Iterate over a set of offload budgets (L0_values) and for each budget:
    Compute threshold b*, train and evaluate a new offload mechanism, and
    compare learned, entropy-based, and oracle decisions.
    Finally, plot accuracy metrics vs fraction of samples processed locally.
    """


    

    local_percent_list = []
    offload_train_acc_list = []
    offload_test_acc_list = []
    ddnn_offload_acc_list = []
    entropy_local_percent_list = []
    entropy_overall_acc_list = []
    oracle_local_percent_list = []
    oracle_overall_acc_list = []

    # Precompute training set features, logits, and bk values
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
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

    for L0 in L0_values:
        # Determine optimal threshold for offloading
        b_star = calculate_b_star(all_bks, L0)

        # Build dataset tuples in specified mode
        combined_data = create_3d_data_deep(
            all_bks, all_features, all_logits, all_images, all_labels,
            input_mode=input_mode
        )
        offload_dataset = OffloadDatasetCNN(combined_data, b_star, input_mode=input_mode,include_bk=False,use_oracle_labels=False,local_clf=local_classifier,cloud_clf=cloud_cnn,device=device)
        offload_loader = DataLoader(offload_dataset, batch_size=batch_size)

        # Infer network input shape
        if input_mode == 'logits':
            in_shape = (all_logits[0].shape[-1],)
        elif input_mode == 'logits_plus':
            in_shape = (all_logits[0].shape[-1] + 2,)
        else:
            sample_x, _, _ = offload_dataset[0]
            in_shape = tuple(sample_x.shape)

        # Instantiate and train decision network
        offload_model = DeepOffloadMechanism(
            input_shape=in_shape,
            input_mode=input_mode,
            fc_dims=[256,128,64,32,1],
            dropout_p=0.1,
            latent_in=(1,)
        ).to(device)
        optimizer = torch.optim.Adam(offload_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
        
        
        train_deep_offload_mechanism(
            offload_model, val_loader, optimizer, offload_loader,
            local_feature_extractor, local_classifier, cloud_cnn,
            b_star, scheduler, input_mode, device,
            epochs=offload_epochs, lr=1e-3, stop_threshold=0.9
        )

        # Offload mechanism accuracies
        off_train = evaluate_offload_decision_accuracy_CNN_train(
            offload_loader, local_feature_extractor, local_classifier,
            cloud_cnn, offload_model, b_star, input_mode=input_mode, device=device
        )
        off_test = evaluate_offload_decision_accuracy_CNN_test(
            offload_model, local_feature_extractor, local_classifier,
            cloud_cnn, test_loader, b_star, input_mode=input_mode, device=device
        )

        # DDNN overall via learned mechanism
        local_pct, ddnn_off = test_DDNN_with_optimized_rule(
            offload_model, local_feature_extractor, local_classifier,
            cloud_cnn, test_loader, b_star, input_mode=input_mode, device=device
        )

        # DDNN via entropy heuristic
        _, _, ent_acc, _, _, ent_pct,entropy_threshold = test_DDNN_with_entropy(
            local_feature_extractor, local_classifier, cloud_cnn,
            test_loader, target_local_percent=local_pct, device=device
        )

        # DDNN via oracle decision function
        oracle_acc, oracle_pct = test_DDNN_with_oracle(
            local_feature_extractor, local_classifier, cloud_cnn,
             test_loader, b_star, device=device
        )

        # Store metrics
        local_percent_list.append(local_pct)
        offload_train_acc_list.append(off_train)
        offload_test_acc_list.append(off_test)
        ddnn_offload_acc_list.append(ddnn_off)
        entropy_local_percent_list.append(ent_pct)
        entropy_overall_acc_list.append(ent_acc)
        oracle_local_percent_list.append(oracle_pct)
        oracle_overall_acc_list.append(oracle_acc)

        print(f"L0={L0:.2f} | local%={local_pct:.2f} | OffTrain={off_train:.2f} | "
              f"OffTest={off_test:.2f} | DDNN_Off={ddnn_off:.2f} | EntDDNN={ent_acc:.2f} | "
              f"Oracle={oracle_acc:.2f}")

    # Sort all lists by local_percent_list
    sort_idx = np.argsort(local_percent_list)
    local_percent_list = np.array(local_percent_list)[sort_idx]
    offload_train_acc_list = np.array(offload_train_acc_list)[sort_idx]
    offload_test_acc_list = np.array(offload_test_acc_list)[sort_idx]
    ddnn_offload_acc_list = np.array(ddnn_offload_acc_list)[sort_idx]
    entropy_local_percent_list = np.array(entropy_local_percent_list)[sort_idx]
    entropy_overall_acc_list = np.array(entropy_overall_acc_list)[sort_idx]
    oracle_local_percent_list = np.array(oracle_local_percent_list)[sort_idx]
    oracle_overall_acc_list = np.array(oracle_overall_acc_list)[sort_idx]
    
    # Plot all metrics vs local percentage
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(local_percent_list, offload_train_acc_list, 'o-', label='Offload Train')
    ax1.plot(local_percent_list, offload_test_acc_list, 's-', label='Offload Test')
    ax1.plot(oracle_local_percent_list, oracle_overall_acc_list, 'd-', label='Oracle')
    ax1.set_xlabel('Local %')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(local_percent_list, ddnn_offload_acc_list, '^--', label='DDNN Offload')
    ax2.plot(entropy_local_percent_list, entropy_overall_acc_list, 'x-.', label='DDNN Entropy')
    ax2.set_ylabel('DDNN Overall Acc (%)')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='best')
    plt.title(f'{dataset_name} | Accuracy vs Local % (mode={input_mode})')
    # plt.tight_layout()
    # plt.show()
    plt.tight_layout()
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    fname = f'difficulty_{dataset_name}_{input_mode}.png'
    fig.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

def test_DDNN_with_oracle(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    test_loader,
    b_star,
    oracle_decision_function= my_oracle_decision_function,
    device='cuda'
):
    """
    Evaluate DDNN accuracy using an oracle decision function that returns binary
    offload decisions directly based on provided logits and threshold b_star.
    """
    import torch
    import torch.nn.functional as F

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Collect logits and labels
    all_loc_logits = []
    all_cld_logits = []
    all_labels     = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            feats = local_feature_extractor(images)
            loc_logits = local_classifier(feats)
            cld_logits = cloud_cnn(feats)
            all_loc_logits.append(loc_logits)
            all_cld_logits.append(cld_logits)
            all_labels.append(labels)

    # Concatenate batches
    all_loc = torch.cat(all_loc_logits)
    all_cld = torch.cat(all_cld_logits)
    all_lbl = torch.cat(all_labels)

    # Obtain oracle decisions directly via the provided function
    # The oracle function expects torch tensors and returns a tensor of 0/1 decisions
    decisions = oracle_decision_function(
        all_loc,
        all_cld,
        all_lbl,
        b_star
    )  # shape (N,), dtype long
    mask = decisions.bool().to(device)

    # Compute predictions
    pred_loc = torch.argmax(all_loc, dim=1)
    pred_cld = torch.argmax(all_cld, dim=1)
    preds    = torch.where(mask, pred_cld, pred_loc)

    # Metrics
    correct = (preds == all_lbl).sum().item()
    total   = all_lbl.size(0)
    acc     = 100.0 * correct / total
    pct_loc = 100.0 * mask.float().mean().item()
    return acc, pct_loc





def analyze_mismatch_samples_with_stats(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    offload_mechanism,
    oracle_rule_function,
    threshold_offload=0.5,
    max_samples_to_plot=20
):
    """
    This enhanced version:
      1) Collects ALL mismatch samples between the oracle decision and the optimized rule.
      2) Computes summary statistics (mean, std) of Δp_class = (local_prob - cloud_prob)
         across all mismatch samples.
      3) Prints a single heatmap (1 x num_classes) with the average Δp_class.
      4) Optionally plots up to 'max_samples_to_plot' individual mismatch samples
         as small heatmaps, indicating the ground-truth label in the title.
      5) Creates a second heatmap with additional statistics for each class:
         - mismatch_count: How many mismatch samples belong to this class.
         - offload_cloud_count: Out of those mismatches, how many times did the
                                offload mechanism pick cloud (1)?
         - oracle_cloud_count:  Out of those mismatches, how many times did the
                                oracle pick cloud (1)?
    
    """

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    offload_mechanism.eval()

    mismatch_diff_list = []  # List of arrays shape=(10,) -> Δp for each mismatch
    mismatch_meta = []       # List of dicts with info about each mismatch sample

    # We'll also track how many mismatches occur per class, and how many times each side picks 'cloud'.
    mismatch_count = [0]*len(CIFAR10_LABELS)
    offload_cloud_count = [0]*len(CIFAR10_LABELS)
    oracle_cloud_count  = [0]*len(CIFAR10_LABELS)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to('cuda'), labels.to('cuda')

            # Extract local features
            local_feats = local_feature_extractor(images)

            # Get local/cloud logits
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # Offload mechanism decision (0=Local, 1=Cloud)
            offload_logits = offload_mechanism(local_feats)
            offload_probs = torch.sigmoid(offload_logits).squeeze(dim=1)
            offload_decisions = (offload_probs > threshold_offload).long()

            # Oracle decision (0=Local, 1=Cloud)
            oracle_decisions = oracle_rule_function(local_out, cloud_out, labels)

            # Find mismatch
            mismatch_mask = (offload_decisions != oracle_decisions)
            if mismatch_mask.any():
                # Convert logits to softmax probabilities
                local_probs = torch.softmax(local_out, dim=1)
                cloud_probs = torch.softmax(cloud_out, dim=1)

                mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=True)[0]
                for idx in mismatch_indices:
                    # Δp = LocalProb - CloudProb for each class
                    dp = local_probs[idx] - cloud_probs[idx]  # shape=(10,)
                    dp_np = dp.cpu().numpy()

                    true_label = labels[idx].item()
                    off_dec = offload_decisions[idx].item()
                    orac_dec = oracle_decisions[idx].item()

                    # Collect the Δp array
                    mismatch_diff_list.append(dp_np)

                    # Collect meta info
                    item_info = {
                        "image": images[idx].cpu(),  # CPU tensor for plotting
                        "true_label": true_label,
                        "oracle": orac_dec,      # 0=Local, 1=Cloud
                        "offload": off_dec       # 0=Local, 1=Cloud
                    }
                    mismatch_meta.append(item_info)

                    # Update mismatch stats per class
                    mismatch_count[true_label] += 1
                    if off_dec == 1:
                        offload_cloud_count[true_label] += 1
                    if orac_dec == 1:
                        oracle_cloud_count[true_label] += 1

    num_mismatches = len(mismatch_diff_list)
    print(f"Total mismatch samples: {num_mismatches}")
    if num_mismatches == 0:
        print("No mismatches found. Nothing to analyze or plot.")
        return

    # Convert mismatch diffs to a single NumPy array: shape=(num_mismatches, 10)
    mismatch_diff_array = np.stack(mismatch_diff_list, axis=0)

    # 1) Compute mean and std of Δp per class
    mean_diff = np.mean(mismatch_diff_array, axis=0)
    std_diff  = np.std(mismatch_diff_array,  axis=0)

    print("Mean of (LocalProb - CloudProb) across mismatch samples (per class):")
    for i, cls_name in enumerate(CIFAR10_LABELS):
        print(f"{cls_name}: mean={mean_diff[i]:.3f}, std={std_diff[i]:.3f}")

    # 2) Single heatmap for the mean Δp
    mean_diff_2d = mean_diff[np.newaxis, :]
    fig, ax = plt.subplots(figsize=(7, 2))
    cax = ax.imshow(mean_diff_2d, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(CIFAR10_LABELS)))
    ax.set_xticklabels(CIFAR10_LABELS, rotation=45, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(["mean Δp"])
    plt.colorbar(cax, ax=ax, label="LocalProb - CloudProb")
    plt.title("Average (LocalProb - CloudProb) over ALL mismatch samples")
    plt.tight_layout()
    plt.show()

    # 3) Optional: plot up to max_samples_to_plot mismatch samples individually
    max_plots = min(num_mismatches, max_samples_to_plot)
    chosen_indices = range(max_plots)
    for i, idx_mismatch in enumerate(chosen_indices):
        dp_vec = mismatch_diff_array[idx_mismatch]  # shape=(10,)
        info = mismatch_meta[idx_mismatch]

        # Build a 1x10 array for the heatmap
        dp_2d = dp_vec[np.newaxis, :]

        fig, ax = plt.subplots(figsize=(8, 2))
        cax = ax.imshow(dp_2d, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(CIFAR10_LABELS)))
        ax.set_xticklabels(CIFAR10_LABELS, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(["Δp"])
        plt.colorbar(cax, ax=ax, label="LocalProb - CloudProb")

        gt_label_str = CIFAR10_LABELS[info["true_label"]]
        orcl_str = info['oracle']  # 0=Local,1=Cloud
        offl_str = info['offload']

        # Mark the correct label with some sign on the x-axis (e.g. an asterisk)
        # We do it by altering the x-tick label only for the correct label index.
        tick_labels = []
        for j, cls_name in enumerate(CIFAR10_LABELS):
            if j == info["true_label"]:
                tick_labels.append(f"{cls_name}*")  # add asterisk for ground-truth
            else:
                tick_labels.append(cls_name)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        # Title
        title_str = (
            f"Mismatch Sample #{i+1}: TrueLabel = {gt_label_str}\n"
            f"OracleDecision={orcl_str}, OffloadDecision={offl_str} (1=Cloud, 0=Local)"
        )
        plt.title(title_str)
        plt.tight_layout()
        plt.show()

    # 4) Create a second heatmap with class-level stats:
    #    mismatch_count, offload_cloud_count, oracle_cloud_count
    # Row0 -> mismatch_count
    # Row1 -> offload_cloud_count
    # Row2 -> oracle_cloud_count
    stat_array = np.array([
        mismatch_count,
        offload_cloud_count,
        oracle_cloud_count
    ], dtype=float)

    row_labels = ["mismatch_count", "offload_cloud_count", "oracle_cloud_count"]

    fig, ax = plt.subplots(figsize=(9, 3))
    cax = ax.imshow(stat_array, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(CIFAR10_LABELS)))
    ax.set_xticklabels(CIFAR10_LABELS, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Optionally annotate each cell with numeric value
    for row_i in range(stat_array.shape[0]):
        for col_i in range(stat_array.shape[1]):
            val = stat_array[row_i, col_i]
            ax.text(col_i, row_i, f"{int(val)}", ha='center', va='center', color='black')

    plt.colorbar(cax, ax=ax, label="Count")
    plt.title("Mismatch class-level stats")
    plt.tight_layout()
    plt.show()

def analyze_mismatch_samples_with_stats_entropy(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    offload_mechanism,
    threshold_offload=0.5,
    threshold_entropy=0.2,
    max_samples_to_plot=20
):
    """
    This function compares the offload mechanism's decisions vs. an Entropy-based rule:
      - The Entropy-based rule: 0=Local if entropy < threshold_entropy, otherwise 1=Cloud.
      - The Offload mechanism: offload_probs > threshold_offload => Cloud, else Local.

    Steps:
      1) Collect ALL mismatch samples where (offload_decision != entropy_decision).
      2) For each mismatch sample, compute Δp_class = (local_prob - cloud_prob),
         and store them in mismatch_diff_list.
      3) Print a heatmap for the average Δp across mismatch samples.
      4) Optionally plot up to 'max_samples_to_plot' mismatch samples individually.
      5) Create a second heatmap with class-level stats:
         - mismatch_count       (# mismatches for that class)
         - offload_cloud_count (# times the offload picked cloud among mismatches)
         - entropy_cloud_count (# times the entropy rule picked cloud among mismatches)
    """

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    offload_mechanism.eval()

    mismatch_diff_list = []  # store the Δp arrays for all mismatch samples
    mismatch_meta = []       # store meta info (image, label, decisions)

    # Per-class counters for mismatch stats
    mismatch_count = [0]*len(CIFAR10_LABELS)
    offload_cloud_count = [0]*len(CIFAR10_LABELS)
    entropy_cloud_count = [0]*len(CIFAR10_LABELS)  # used to be oracle_cloud_count

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # 1) Local features
            local_feats = local_feature_extractor(images)

            # 2) Get local/cloud logits
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # 3) Offload mechanism decision (0=Local, 1=Cloud)
            offload_logits = offload_mechanism(local_feats)
            offload_probs = torch.sigmoid(offload_logits).squeeze(1)
            offload_decisions = (offload_probs > threshold_offload).long()

            # 4) Entropy-based decision
            #    0=Local if entropy < threshold_entropy, else 1=Cloud
            entropy_vals = calculate_normalized_entropy(local_out)
            entropy_decisions = (entropy_vals >= threshold_entropy).long()

            # 5) Mismatch mask
            mismatch_mask = (offload_decisions != entropy_decisions)
            if mismatch_mask.any():
                # Convert logits -> softmax probabilities
                local_probs = F.softmax(local_out, dim=1)
                cloud_probs = F.softmax(cloud_out, dim=1)

                mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=True)[0]
                for idx in mismatch_indices:
                    dp = local_probs[idx] - cloud_probs[idx]  # shape=(10,)
                    dp_np = dp.cpu().numpy()

                    true_label = labels[idx].item()
                    off_dec = offload_decisions[idx].item()
                    ent_dec = entropy_decisions[idx].item()

                    mismatch_diff_list.append(dp_np)
                    mismatch_meta.append({
                        "image": images[idx].cpu(),
                        "true_label": true_label,
                        "entropy_decision": ent_dec,
                        "offload": off_dec
                    })

                    # Update mismatch stats
                    mismatch_count[true_label] += 1
                    if off_dec == 1:
                        offload_cloud_count[true_label] += 1
                    if ent_dec == 1:
                        entropy_cloud_count[true_label] += 1

    num_mismatches = len(mismatch_diff_list)
    print(f"Total mismatch samples: {num_mismatches}")
    if num_mismatches == 0:
        print("No mismatches found. Nothing to analyze or plot.")
        return

    mismatch_diff_array = np.stack(mismatch_diff_list, axis=0)  # shape=(num_mismatches, 10)

    # Compute mean and std of Δp per class
    mean_diff = np.mean(mismatch_diff_array, axis=0)
    std_diff  = np.std(mismatch_diff_array, axis=0)

    print("Mean of (LocalProb - CloudProb) across mismatch samples (per class):")
    for i, cls_name in enumerate(CIFAR10_LABELS):
        print(f"{cls_name}: mean={mean_diff[i]:.3f}, std={std_diff[i]:.3f}")

    # 1) Heatmap for mean Δp
    mean_diff_2d = mean_diff[np.newaxis, :]
    fig, ax = plt.subplots(figsize=(7, 2))
    cax = ax.imshow(mean_diff_2d, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(CIFAR10_LABELS)))
    ax.set_xticklabels(CIFAR10_LABELS, rotation=45, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(["mean Δp"])
    plt.colorbar(cax, ax=ax, label="LocalProb - CloudProb")
    plt.title("Average (LocalProb - CloudProb) over ALL mismatch samples\n(Offload vs Entropy)")
    plt.tight_layout()
    plt.show()

    # 2) Plot up to max_samples_to_plot mismatch samples
    max_plots = min(num_mismatches, max_samples_to_plot)
    chosen_indices = range(max_plots)
    for i, idx_mismatch in enumerate(chosen_indices):
        dp_vec = mismatch_diff_array[idx_mismatch]  # shape=(10,)
        info = mismatch_meta[idx_mismatch]

        dp_2d = dp_vec[np.newaxis, :]

        fig, ax = plt.subplots(figsize=(8,2))
        cax = ax.imshow(dp_2d, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(CIFAR10_LABELS)))
        ax.set_xticklabels(CIFAR10_LABELS, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(["Δp"])
        plt.colorbar(cax, ax=ax, label="LocalProb - CloudProb")

        gt_label_str = CIFAR10_LABELS[info["true_label"]]
        ent_dec_str = info['entropy_decision']  # 0=Local,1=Cloud
        offl_str = info['offload']              # 0=Local,1=Cloud

        # Mark the correct label with an asterisk
        tick_labels = []
        for j, cls_name in enumerate(CIFAR10_LABELS):
            if j == info["true_label"]:
                tick_labels.append(f"{cls_name}*")
            else:
                tick_labels.append(cls_name)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        title_str = (
            f"Mismatch Sample #{i+1} - TrueLabel={gt_label_str}\n"
            f"EntropyDecision={ent_dec_str}, Offload={offl_str} (1=Cloud,0=Local)"
        )
        plt.title(title_str)
        plt.tight_layout()
        plt.show()

    # 3) Class-level heatmap for mismatch_count, offload_cloud_count, entropy_cloud_count
    stat_array = np.array([
        mismatch_count,
        offload_cloud_count,
        entropy_cloud_count
    ], dtype=float)

    row_labels = ["mismatch_count", "offload_cloud_count", "entropy_cloud_count"]

    fig, ax = plt.subplots(figsize=(9, 3))
    cax = ax.imshow(stat_array, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(CIFAR10_LABELS)))
    ax.set_xticklabels(CIFAR10_LABELS, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate each cell
    for row_i in range(stat_array.shape[0]):
        for col_i in range(stat_array.shape[1]):
            val = stat_array[row_i, col_i]
            ax.text(col_i, row_i, f"{int(val)}", ha='center', va='center', color='black')

    plt.colorbar(cax, ax=ax, label="Count")
    plt.title("Mismatch class-level stats (Offload vs Entropy)")
    plt.tight_layout()
    plt.show()


def analyze_feature_correlation_with_oracle(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    oracle_rule_function
):
    """
    This function:
    1) Passes each sample through local_feature_extractor -> local_classifier/cloud_cnn.
    2) Gets an oracle decision (0=local, 1=cloud) per sample.
    3) Collects flattened local features and the binary oracle decision for all samples.
    4) Computes Pearson correlation (r) between each feature dimension and the oracle decision.
       Returns arrays of shape (feat_dim,) with r-values and p-values.

    Comments are in English as requested.
    """

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    all_feats = []
    all_oracle = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to('cuda'), labels.to('cuda')

            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # Oracle decision 0/1
            oracle_decisions = oracle_rule_function(local_out, cloud_out, labels)

            # Flatten local features
            # feats_flat = local_feats.view(local_feats.size(0), -1).cpu().numpy()
            feats_flat= local_feats
            oracle_np = oracle_decisions.cpu().numpy()

            all_feats.append(feats_flat)
            all_oracle.append(oracle_np)

    # Assemble into numpy arrays
    all_feats = [f.cpu().numpy() for f in all_feats]
    all_feats = np.concatenate(all_feats, axis=0)
    
    all_oracle = np.concatenate(all_oracle, axis=0) # shape=(N,)

    # feat_dim = all_feats.shape
    N_samples = all_feats.shape[0]
    flattened_feats = all_feats.reshape(N_samples, -1)  # shape: (10000, 8192)
    num_features = flattened_feats.shape[1]

    # num_features= all_feats.shape[1]
    correlations = np.zeros(num_features, dtype=np.float32)
    pvalues = np.zeros(num_features, dtype=np.float32)

    # Calculate Pearson correlation per feature dimension
    for d in range(num_features):
        feat_col = flattened_feats[:, d]
        r, p = pearsonr(feat_col, all_oracle)
        correlations[d] = r
        pvalues[d] = p

    return correlations, pvalues



def analyze_top_prob_diffs_vs_oracle(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    offload_mechanism,
    oracle_decision_func,
    threshold_offload=0.5,
    device='cuda'
):
    """
    Analyzes how the difference of top-2 probabilities and the difference
    between top-1 and the rest correlate with the oracle's decision.
    
    For each sample, we compute:
      1) local_top2_diff = (p1_local - p2_local)
      2) local_p1_minus_rest = p1_local - (1 - p1_local) = 2*p1_local - 1
      (the same two metrics for the cloud: cloud_top2_diff, cloud_p1_minus_rest)

    Then we collect:
      - Whether the offload mechanism chooses cloud (1) or local (0).
      - The oracle decision (1 or 0) via oracle_decision_func.
      - We mark if there's a mismatch (offload != oracle) or correct-match (offload == oracle).

    Finally, we compute and print average values of these diffs in three categories:
      [A] ALL samples
      [B] Mismatch samples (offload != oracle)
      [C] Correct-match samples (offload == oracle)

    We also produce a bar plot comparing these means for local vs cloud, 
    across the three categories (ALL vs mismatch vs correct).

    """

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    offload_mechanism.eval()

    # Lists to store values for ALL samples
    local_top2_all = []
    cloud_top2_all = []
    local_p1rest_all = []
    cloud_p1rest_all = []

    # Lists to store values for only the MISMATCH samples
    local_top2_mismatch = []
    cloud_top2_mismatch = []
    local_p1rest_mismatch = []
    cloud_p1rest_mismatch = []

    # Lists to store values for only the CORRECT-MATCH samples
    local_top2_correct = []
    cloud_top2_correct = []
    local_p1rest_correct = []
    cloud_p1rest_correct = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # 1) Forward pass for local and cloud networks
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)   # logits (bs, 10)
            cloud_out = cloud_cnn(local_feats)          # logits (bs, 10)

            # 2) Convert to probabilities
            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)

            # 3) Compute top-1 (p1) and top-2 (p2) for local and cloud
            local_sorted, _ = torch.sort(local_probs, descending=True, dim=1)
            cloud_sorted, _ = torch.sort(cloud_probs, descending=True, dim=1)

            p1_local = local_sorted[:, 0]
            p2_local = local_sorted[:, 1]
            p1_cloud = cloud_sorted[:, 0]
            p2_cloud = cloud_sorted[:, 1]

            # Differences for local
            local_diff_top2 = p1_local - p2_local
            local_diff_p1_rest = p1_local - (1.0 - p1_local)  # == 2*p1_local - 1

            # Differences for cloud
            cloud_diff_top2 = p1_cloud - p2_cloud
            cloud_diff_p1_rest = p1_cloud - (1.0 - p1_cloud)  # == 2*p1_cloud - 1

            # 4) Offload mechanism's decision
            offload_logits = offload_mechanism(local_feats)
            offload_probs = torch.sigmoid(offload_logits).squeeze(1)
            offload_decisions = (offload_probs > threshold_offload).long()

            # 5) Oracle decision
            oracle_decisions = oracle_decision_func(local_out, cloud_out, labels)

            # 6) Mismatch vs Correct-match
            mismatch_mask = (offload_decisions != oracle_decisions)
            correct_mask = (offload_decisions == oracle_decisions)

            # 7) Store values for ALL samples
            local_top2_all.extend(local_diff_top2.tolist())
            cloud_top2_all.extend(cloud_diff_top2.tolist())
            local_p1rest_all.extend(local_diff_p1_rest.tolist())
            cloud_p1rest_all.extend(cloud_diff_p1_rest.tolist())

            # 8) Store values for mismatch samples only
            if mismatch_mask.any():
                mismatch_idx = mismatch_mask.nonzero(as_tuple=True)[0]

                local_top2_mismatch.extend(local_diff_top2[mismatch_idx].tolist())
                cloud_top2_mismatch.extend(cloud_diff_top2[mismatch_idx].tolist())
                local_p1rest_mismatch.extend(local_diff_p1_rest[mismatch_idx].tolist())
                cloud_p1rest_mismatch.extend(cloud_diff_p1_rest[mismatch_idx].tolist())

            # 9) Store values for correct-match samples only
            if correct_mask.any():
                correct_idx = correct_mask.nonzero(as_tuple=True)[0]

                local_top2_correct.extend(local_diff_top2[correct_idx].tolist())
                cloud_top2_correct.extend(cloud_diff_top2[correct_idx].tolist())
                local_p1rest_correct.extend(local_diff_p1_rest[correct_idx].tolist())
                cloud_p1rest_correct.extend(cloud_diff_p1_rest[correct_idx].tolist())

    # ----- Now compute mean stats -----
    def mean_of_list(lst):
        if len(lst) == 0:
            return 0.0
        return float(np.mean(lst))

    # A) Means over ALL samples
    mean_local_top2_all = mean_of_list(local_top2_all)
    mean_cloud_top2_all = mean_of_list(cloud_top2_all)
    mean_local_p1rest_all = mean_of_list(local_p1rest_all)
    mean_cloud_p1rest_all = mean_of_list(cloud_p1rest_all)

    # B) Means over MISMATCH only
    mean_local_top2_mismatch = mean_of_list(local_top2_mismatch)
    mean_cloud_top2_mismatch = mean_of_list(cloud_top2_mismatch)
    mean_local_p1rest_mismatch = mean_of_list(local_p1rest_mismatch)
    mean_cloud_p1rest_mismatch = mean_of_list(cloud_p1rest_mismatch)

    # C) Means over CORRECT-MATCH only
    mean_local_top2_correct = mean_of_list(local_top2_correct)
    mean_cloud_top2_correct = mean_of_list(cloud_top2_correct)
    mean_local_p1rest_correct = mean_of_list(local_p1rest_correct)
    mean_cloud_p1rest_correct = mean_of_list(cloud_p1rest_correct)

    # ---- Print results ----
    print("=== TOP-2 DIFF and (p1 - rest) Stats vs Oracle ===")
    print("ALL SAMPLES:")
    print(f"  Local mean top2 diff:    {mean_local_top2_all:.4f}")
    print(f"  Cloud mean top2 diff:    {mean_cloud_top2_all:.4f}")
    print(f"  Local mean p1-rest:      {mean_local_p1rest_all:.4f}")
    print(f"  Cloud mean p1-rest:      {mean_cloud_p1rest_all:.4f}")
    print("\nMISMATCH SAMPLES ONLY:")
    print(f"  Local mean top2 diff:    {mean_local_top2_mismatch:.4f}")
    print(f"  Cloud mean top2 diff:    {mean_cloud_top2_mismatch:.4f}")
    print(f"  Local mean p1-rest:      {mean_local_p1rest_mismatch:.4f}")
    print(f"  Cloud mean p1-rest:      {mean_cloud_p1rest_mismatch:.4f}")
    print("\nCORRECT-MATCH SAMPLES ONLY:")
    print(f"  Local mean top2 diff:    {mean_local_top2_correct:.4f}")
    print(f"  Cloud mean top2 diff:    {mean_cloud_top2_correct:.4f}")
    print(f"  Local mean p1-rest:      {mean_local_p1rest_correct:.4f}")
    print(f"  Cloud mean p1-rest:      {mean_cloud_p1rest_correct:.4f}")

    # Optional: create a triple bar chart for easier comparison
    # We'll plot side-by-side bars for (Local top2, Cloud top2, Local p1-rest, Cloud p1-rest)
    # for three categories: ALL, MISMATCH, CORRECT
    import matplotlib.pyplot as plt

    metrics_all = [
        mean_local_top2_all,
        mean_cloud_top2_all,
        mean_local_p1rest_all,
        mean_cloud_p1rest_all
    ]
    metrics_mismatch = [
        mean_local_top2_mismatch,
        mean_cloud_top2_mismatch,
        mean_local_p1rest_mismatch,
        mean_cloud_p1rest_mismatch
    ]
    metrics_correct = [
        mean_local_top2_correct,
        mean_cloud_top2_correct,
        mean_local_p1rest_correct,
        mean_cloud_p1rest_correct
    ]

    x_labels = [
        "Local Top2", "Cloud Top2", "Local p1-rest", "Cloud p1-rest"
    ]

    x_positions = np.arange(len(x_labels))  # e.g. [0,1,2,3]
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))

    # We'll have triple bars side-by-side:
    # All => x_positions - bar_width
    # Mismatch => x_positions
    # Correct => x_positions + bar_width

    ax.bar(x_positions - bar_width, metrics_all, width=bar_width, label="All Samples")
    ax.bar(x_positions, metrics_mismatch, width=bar_width, label="Mismatch Only")
    ax.bar(x_positions + bar_width, metrics_correct, width=bar_width, label="Correct-Match Only")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=15)
    ax.set_ylabel("Mean Probability Difference")
    ax.set_title("Comparing Probability Diffs (Optimized Rule vs Oracle Decisions)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def analyze_top_prob_diffs_vs_entropy(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    offload_mechanism,
    entropy_threshold=0.2,
    threshold_offload=0.5,
    device='cuda'
):
    """
    This function analyzes how different probability metrics correlate with decisions made by an offload mechanism compared to an entropy-based decision rule.

    Specifically, for each sample, it computes:
      - The difference between the top two probabilities (top-2 diff) for both local and cloud models.
      - The difference between the top probability and the sum of the rest (p1-rest) for both local and cloud models.
      - The normalized entropy of local predictions.

    It categorizes samples into three groups:
      - ALL samples.
      - MISMATCH samples, where the offload mechanism's decision differs from the entropy-based decision.
      - CORRECT-MATCH samples, where both methods agree.

    Finally, it calculates and visualizes the mean values of these metrics for each group.
    """

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    offload_mechanism.eval()

    # Initialize metric lists for ALL, mismatch, and correct-match samples
    metrics = {key: {'local_top2': [], 'cloud_top2': [], 'local_p1rest': [], 'cloud_p1rest': [], 'local_entropy': []}
               for key in ['all', 'mismatch', 'correct']}

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            local_probs = F.softmax(local_out, dim=1)
            cloud_probs = F.softmax(cloud_out, dim=1)

            local_sorted, _ = torch.sort(local_probs, descending=True, dim=1)
            cloud_sorted, _ = torch.sort(cloud_probs, descending=True, dim=1)

            local_diff_top2 = local_sorted[:, 0] - local_sorted[:, 1]
            cloud_diff_top2 = cloud_sorted[:, 0] - cloud_sorted[:, 1]
            local_diff_p1_rest = 2 * local_sorted[:, 0] - 1
            cloud_diff_p1_rest = 2 * cloud_sorted[:, 0] - 1

            local_ent = calculate_normalized_entropy(local_out)

            offload_probs = torch.sigmoid(offload_mechanism(local_feats)).squeeze(1)
            offload_decisions = (offload_probs > threshold_offload).long()

            entropy_decisions = (local_ent >= entropy_threshold).long()

            mismatch_mask = (offload_decisions != entropy_decisions)
            correct_mask = (offload_decisions == entropy_decisions)

            for mask, key in zip([mismatch_mask, correct_mask], ['mismatch', 'correct']):
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]
                    metrics[key]['local_top2'].extend(local_diff_top2[idx].tolist())
                    metrics[key]['cloud_top2'].extend(cloud_diff_top2[idx].tolist())
                    metrics[key]['local_p1rest'].extend(local_diff_p1_rest[idx].tolist())
                    metrics[key]['cloud_p1rest'].extend(cloud_diff_p1_rest[idx].tolist())
                    metrics[key]['local_entropy'].extend(local_ent[idx].tolist())

            metrics['all']['local_top2'].extend(local_diff_top2.tolist())
            metrics['all']['cloud_top2'].extend(cloud_diff_top2.tolist())
            metrics['all']['local_p1rest'].extend(local_diff_p1_rest.tolist())
            metrics['all']['cloud_p1rest'].extend(cloud_diff_p1_rest.tolist())
            metrics['all']['local_entropy'].extend(local_ent.tolist())

    # Calculate means
    mean_metrics = {}
    for cat in ['all', 'mismatch', 'correct']:
        mean_metrics[cat] = {m: np.mean(metrics[cat][m]) if metrics[cat][m] else 0 for m in metrics[cat]}

    # Print results
    for cat in ['all', 'mismatch', 'correct']:
        print(f"\n{cat.upper()} SAMPLES:")
        for m in mean_metrics[cat]:
            print(f"  {m}: {mean_metrics[cat][m]:.4f}")

    # Plotting
    labels = ['Local Top2', 'Cloud Top2', 'Local p1-rest', 'Cloud p1-rest', 'Local Entropy']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (cat, offset) in enumerate(zip(['all', 'mismatch', 'correct'], [-width, 0, width])):
        values = [mean_metrics[cat]['local_top2'], mean_metrics[cat]['cloud_top2'],
                  mean_metrics[cat]['local_p1rest'], mean_metrics[cat]['cloud_p1rest'], mean_metrics[cat]['local_entropy']]
        ax.bar(x + offset, values, width, label=f"{cat.capitalize()} Samples")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean Values")
    ax.set_title("Probability Diffs & Entropy (Optimized Rule vs Entropy Decisions)")
    ax.legend()
    plt.tight_layout()
    plt.show()
def analyze_oracle_easy_hard_distribution(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    b_star,
    device='cuda'
):
    """
    Computes how many samples the oracle deems 'easy' vs 'hard', 
    using the my_oracle_decision_function that returns 0=>local, 1=>cloud.

    - easy_count: number of samples with oracle_decision=0
    - hard_count: number of samples with oracle_decision=1

    Plots a simple bar chart at the end.
    """
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    total_samples = 0
    easy_count = 0
    hard_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            # Forward
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # Oracle decision (0=>local,1=>cloud)
            oracle_decisions = my_oracle_decision_function(local_out, cloud_out, labels, b_star)
            # Count how many 0s, how many 1s
            # .sum() => counts how many 1
            # (batch_size - sum) => counts how many 0
            ones = oracle_decisions.sum().item()
            hard_count += ones
            easy_count += (batch_size - ones)

    print(f"Total samples: {total_samples}")
    print(f"Oracle-labeled 'Easy'(0): {easy_count} => {100.0*easy_count/total_samples:.2f}%")
    print(f"Oracle-labeled 'Hard'(1): {hard_count} => {100.0*hard_count/total_samples:.2f}%")

    # Plot distribution
    categories = ['Easy(0)', 'Hard(1)']
    values = [easy_count, hard_count]

    plt.figure(figsize=(4,4))
    plt.bar(categories, values, color=['green','red'], alpha=0.6)
    plt.title("Oracle-labeled Easy vs Hard (0=local,1=cloud)")
    for i, val in enumerate(values):
        plt.text(i, val+1, str(val), ha='center', fontsize=10)
    plt.ylabel("Number of samples")
    plt.show()

def compute_offload_confusion_matrix(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    deep_offload_model,
    b_star,
    device='cuda',
    threshold=0.5
):
    """
    Builds a confusion matrix comparing the Oracle's decision(0=local,1=cloud)
    vs the offload model's decision(0=local,1=cloud). 
    The offload model's decision is determined by:
      offload_probs = sigmoid(deep_offload_model(...))
      if offload_probs> threshold => 1 (cloud/hard), else =>0 (local/easy)

    Then we define:
      Positive => 'hard'(1)
      Negative => 'easy'(0)

      So:
        TP => (oracle=1, offload=1)
        FP => (oracle=0, offload=1)
        FN => (oracle=1, offload=0)
        TN => (oracle=0, offload=0)

    Finally, we print out the confusion matrix, plus accuracy, precision, recall, F1,
    and show a small heatmap.
    """
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    deep_offload_model.eval()

    TP = FP = FN = TN = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            bs = images.size(0)

            # 1) forward
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # 2) oracle => 0 or 1
            oracle_decisions = my_oracle_decision_function(local_out, cloud_out, labels,b_star)
            # shape (bs,)

            # 3) offload => 0 or 1
            # feats_flat = local_feats.view(bs, -1)
            feats_flat= local_feats
            logits = deep_offload_model(feats_flat)
            offload_probs = torch.sigmoid(logits).squeeze(1)
            offload_decisions = (offload_probs > threshold).long()

            # 4) confusion matrix update
            for i in range(bs):
                oracle_label = oracle_decisions[i].item()      # 0 or 1
                offload_label = offload_decisions[i].item()    # 0 or 1
                if oracle_label==1 and offload_label==1:
                    TP+=1
                elif oracle_label==0 and offload_label==1:
                    FP+=1
                elif oracle_label==1 and offload_label==0:
                    FN+=1
                else:
                    TN+=1

    total = TP+FP+FN+TN
    if total==0:
        print("No samples => empty confusion matrix.")
        return

    accuracy = 100.0*(TP+TN)/total
    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    print("Offload confusion matrix (Oracle=0/1 vs Offload=0/1). Positive=1='cloud/hard'.")
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}, total={total}")
    print(f"Accuracy={accuracy:.2f}%, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # plot small heatmap
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array([[TN, FP],
                       [FN, TP]], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(3,3))
    cax = ax.imshow(matrix, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax, ax=ax)
    ax.set_title("Offload Confusion Matrix\n(0=local,1=cloud)")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Offload=0(local)","Offload=1(cloud)"])
    ax.set_yticklabels(["Oracle=0(local)","Oracle=1(cloud)"])

    for (y,x), val in np.ndenumerate(matrix):
        ax.text(x, y, str(val), ha='center', va='center', fontsize=14, color='black')

    plt.tight_layout()
    plt.show()
def compute_offload_confusion_matrix_entropy(
    data_loader,
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    deep_offload_model,
    device='cuda',
    threshold=0.5,
    entropy_threshold=0.2
):
    """
    Builds a confusion matrix comparing the Entropy method's decision (0=local,1=cloud)
    vs the offload model's decision (0=local,1=cloud).
    
    The offload model's decision is determined by:
      offload_probs = sigmoid(deep_offload_model(...))
      if offload_probs > threshold => 1 (cloud), else => 0 (local)

    Entropy-based decision:
      normalized_entropy(local_classifier(local_features))
      if entropy < entropy_threshold => 0 (local), else => 1 (cloud)

    We define:
      Positive => 'cloud'(1)
      Negative => 'local'(0)

      So:
        TP => (entropy=1, offload=1)
        FP => (entropy=0, offload=1)
        FN => (entropy=1, offload=0)
        TN => (entropy=0, offload=0)

    Finally, we print out the confusion matrix, plus accuracy, precision, recall, F1,
    and show a small heatmap.
    """
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    deep_offload_model.eval()

    TP = FP = FN = TN = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            bs = images.size(0)

            # Forward pass to get local features
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)

            # Entropy decision
            entropy = calculate_normalized_entropy(local_out)
            entropy_decisions = (entropy >= entropy_threshold).long()  # 1 if entropy >= threshold (cloud), else 0 (local)

            # Offload decision
            offload_logits = deep_offload_model(local_feats)
            offload_probs = torch.sigmoid(offload_logits).squeeze(1)
            offload_decisions = (offload_probs > threshold).long()

            # Confusion matrix calculation
            for i in range(bs):
                entropy_label = entropy_decisions[i].item()      # 0 or 1
                offload_label = offload_decisions[i].item()      # 0 or 1
                if entropy_label == 1 and offload_label == 1:
                    TP += 1
                elif entropy_label == 0 and offload_label == 1:
                    FP += 1
                elif entropy_label == 1 and offload_label == 0:
                    FN += 1
                else:
                    TN += 1

    total = TP + FP + FN + TN
    if total == 0:
        print("No samples => empty confusion matrix.")
        return

    accuracy = 100.0 * (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("Offload confusion matrix (Entropy=0/1 vs Offload=0/1). Positive=1='cloud'.")
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}, total={total}")
    print(f"Accuracy={accuracy:.2f}%, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # Plot heatmap
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array([[TN, FP],
                       [FN, TP]], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(3, 3))
    cax = ax.imshow(matrix, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax, ax=ax)
    ax.set_title("Offload vs Entropy Confusion Matrix\n(0=local, 1=cloud)")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Offload=0(local)", "Offload=1(cloud)"])
    ax.set_yticklabels(["Entropy=0(local)", "Entropy=1(cloud)"])

    for (y, x), val in np.ndenumerate(matrix):
        ax.text(x, y, str(val), ha='center', va='center', fontsize=14, color='black')

    plt.tight_layout()
    plt.show()
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

    ------------------------------------------------------------------------
    Expected dataset layout
    -----------------------
    ``offload_dataset`` **must** have been instantiated with
    ``include_bk=True`` so that ``__getitem__`` returns::

        (feat_map,  y_offload,  y_cifar,  bk_val)
        # tensor    float32     long       float32
        # shape ──> (C,H,W)       ()          ()

    * ``feat_map``   : feature map from the local feature extractor  
    * ``y_offload``  : 0 = local, 1 = cloud, produced by rule *bk ≥ bk\***  
    * ``y_cifar``    : ground‑truth class label (0–9)  
    * ``bk_val``     : scalar C_local − C_cloud for this sample

    ------------------------------------------------------------------------
    What the function does
    ----------------------
    1. **Oracle routing** – for every mini‑batch it calls
       ``oracle_decision_func(local_logits, cloud_logits, labels)`` which
       returns *0* if the local path is correct, *1* if the cloud path is
       correct (ties resolved by higher soft‑max confidence).

    2. **Oracle‑ceiling accuracy** – combines the oracle route with the
       corresponding classifier prediction and counts how many samples are
       classified correctly.  This is the *best attainable* accuracy without
       re‑training either classifier.

    3. **Label‑noise rate** – compares the oracle route with the current
       off‑load label (derived from the bk rule).  If they disagree the
       sample is considered **noisy**, because the supervision instructs the
       DOM to do the opposite of the optimal decision.

    4. **Border‑rate** – classifies a sample as *borderline* when
       ``|bk_val| < tau``.  These cases have negligibly small cost
       difference; their labels are inherently unstable and tend to inject
       noise even if they coincide with the oracle in this pass.

    ------------------------------------------------------------------------
    Returns
    -------
    dict
        {
          "oracle_acc" : float   # % total samples classified correctly by oracle
          "noise_rate" : float   # % samples where (bk‑label) ≠ oracle route
          "border_rate": float   # % samples with |bk| < tau
        }

 
    """
    loader = DataLoader(offload_dataset, batch_size=batch_size, shuffle=False)

    local_feat_extr.eval()
    local_clf.eval()
    cloud_cnn.eval()

    N = oracle_ok = noisy = borders = 0
    # keep track of how many samples oracle sends to local
    routed_local=0
    with torch.no_grad():
        for x_tensor, dom_lab, true_lab, features, bk_val in loader:
            x_tensor, dom_lab, true_lab,features, bk_val = \
                x_tensor.to(device), dom_lab.to(device), true_lab.to(device), features.to(device), bk_val.to(device)

            loc_logits   = local_clf(features)
            cloud_logits = cloud_cnn(features)

            # oracle routing: 0 local, 1 cloud
            oracle_route = oracle_decision_func(loc_logits, cloud_logits, true_lab, bk_star)

            # oracle accuracy
            loc_pred   = loc_logits.argmax(1)
            cloud_pred = cloud_logits.argmax(1)
            # set final pred to the one the oracle chose. If oracle route==0 -> final_pred=loc_pred, if oracle_route==1 -> final_pred= cloud_pred
            final_pred = torch.where(oracle_route == 0, loc_pred, cloud_pred)
            oracle_ok += (final_pred == true_lab).sum().item()

            # disagreement with dom label
            noisy += (oracle_route.float() != dom_lab).sum().item()

            # border samples
            borders += (bk_val.abs() < tau).sum().item()

            N += true_lab.size(0)
            
            # count sample sent locally
            routed_local += (oracle_route == 0).sum().item()
    
    metrics = {
        "oracle_acc" : 100 * oracle_ok / N,
        "noise_rate" : 100 * noisy   / N,
        "border_rate": 100 * borders / N,
        "local_rate":  100 * routed_local / N
    }

    if plot:
        names = ["Oracle acc", "Noise rate", "Border rate","Local Rate"]
        values = [metrics["oracle_acc"], metrics["noise_rate"], metrics["border_rate"],metrics["local_rate"]]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(names, values)
        plt.ylim(0, 100)
        plt.ylabel("Percentage (%)")
        plt.title("Oracle ceiling & label‑noise metrics")
        # annotate bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, val + 1,
                     f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.show()
    return metrics

def build_clean_mask(offload_dataset,
                     local_feat_extr, local_clf, cloud_cnn,
                     oracle_decision_func,
                     b_star, tau=0.02,
                     *, input_mode='feat',
                     device='cuda', batch_size=126):
    """
    Returns a boolean tensor mask (len == len(offload_dataset)) where
      True  → sample is kept for training (oracle agrees & not near border)
      False → sample is filtered out.

    input_mode : 'img' | 'feat' | 'logits'
        must match how offload_dataset was constructed.
    """
    loader = DataLoader(offload_dataset,
                        batch_size=batch_size,
                        shuffle=False)

    keep_flags = torch.zeros(len(offload_dataset), dtype=torch.bool)
    idx_ptr    = 0

    local_feat_extr.eval(); local_clf.eval(); cloud_cnn.eval()

    with torch.no_grad():
        for x_tensor, dom_lab, true_lab, z_feats, bk_val in loader:
            # ------------------------------------------------------------------
            # Move to device
            x_tensor  = x_tensor.to(device)
            dom_lab   = dom_lab.to(device)
            true_lab  = true_lab.to(device)
            z_feats   = z_feats.to(device)          # (B,32,16,16)
            bk_val    = bk_val.to(device)           # (B,)

            # ------------------------------------------------------------------
            # Decide which feature map θα χρησιμοποιήσουμε
            if input_mode == 'logits':
                feats = z_feats                     # ήδη διαθέσιμο
                local_logits = x_tensor             # (B,10)
            elif input_mode == 'feat':
                feats = x_tensor                    # (B,32,16,16)
                local_logits = local_clf(feats)
            else:  # 'img'
                feats = local_feat_extr(x_tensor)   # εικόνα → feature-map
                local_logits = local_clf(feats)

            cloud_logits = cloud_cnn(feats)

            # ------------------------------------------------------------------
            # Oracle & border-distance
            oracle_route = oracle_decision_func(local_logits, cloud_logits, true_lab)  # 0/1

            agree   = (oracle_route.float() == dom_lab)
            not_brd = (bk_val.abs() >= tau)

            B = x_tensor.size(0)
            keep_flags[idx_ptr : idx_ptr + B] = (agree & not_brd).cpu()
            idx_ptr += B

    return keep_flags


def build_clean_mask_test(
        test_loader: DataLoader,
        local_feature_extractor: nn.Module,
        local_classifier: nn.Module,
        cloud_cnn: nn.Module,
        oracle_decision_fn,
        b_star: float,
        tau: float = 0.02,
        device: str | torch.device = "cuda",
        batch_size: int = 126) -> torch.BoolTensor:
    """
    Create a boolean mask for the CIFAR-10 *test* split that mimics the
    same cleaning criterion used for the training set.

    A sample is kept (mask == True) **iff**
        1.  The oracle routing decision agrees with the bk-derived label
            ( bk >= b*  → 1, else 0 ),  **AND**
        2.  |bk| >= tau   (far from the decision border).

    Parameters
    ----------
    test_loader : DataLoader
        Loader that iterates over raw (image, label) pairs — no shuffling
        so that indices are stable.
    local_feature_extractor : nn.Module
        CNN that converts raw images (3×32×32) to feature-maps (C×H×W).
    local_classifier : nn.Module
        2-layer head that maps feature-maps to 10-class logits.
    cloud_cnn : nn.Module
        Cloud branch that receives the *same* feature-maps.
    oracle_decision_fn : Callable
        Must accept (local_logits, cloud_logits, labels) and return a
        binary tensor (0 → keep local, 1 → offload).
    b_star : float
        Offloading threshold computed on the training set.
    tau : float, optional
        Margin from the threshold; samples with |bk| < tau are deemed
        too close to the border and removed.
    device : str or torch.device
        CUDA / CPU device for inference.
    batch_size : int, optional
        Mini-batch size for the forward pass.

    Returns
    -------
    torch.BoolTensor
        Mask of shape (N_test,) where True means "keep this sample".
    """
    loader = DataLoader(
        test_loader.dataset,      # reuse underlying Dataset
        batch_size=batch_size,
        shuffle=False)            # keep deterministic ordering

    keep_mask = torch.zeros(len(test_loader.dataset), dtype=torch.bool)

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    idx_ptr = 0
    with torch.no_grad():
        for images, labels in loader:
            images= images.to(device) # turn then to gpu device
            labels=  labels.to(device)
            B= labels.size(0)   #get the number of samples  (batch_size)
            # forward pass throufh DDNN pipeline
            features= local_feature_extractor(images)    # size:(B,C,H,W)
            local_logits = local_classifier(features)   # size:(B,10)
            cloud_logits= cloud_cnn(features)           # size:(B,10)
            
            # Compute bk and the coresponding binary
            # get the predictionafter softmaxing the logits of he classfiers
            local_probs = local_logits.softmax(dim=1)   # size: (B,10)
            cloud_probs = cloud_logits.softmax(dim=1)   # size: (B,10)
            
            # we new need to extract the probability given for the correct label
            correct_prob_local= local_probs[torch.arange(B),labels]
            correct_prob_cloud= cloud_probs[torch.arange(B),labels]
            bk = (1.0 - correct_prob_local) - (1.0 - correct_prob_cloud)                          # (B,)
            bk_label = (bk >= b_star).float()                     # (B,)
            
            # find oracle decision about the certain samples
            oracle_route = oracle_decision_fn(local_logits, cloud_logits, labels).float()       # (B,)
            
            # compare the 2 decisions
            agree   = (bk_label == oracle_route)                  # (B,)
            not_brd = (bk.abs() >= tau)                           # (B,)
            keep_batch = (agree & not_brd).cpu()                  # (B,)

            keep_mask[idx_ptr: idx_ptr + B] = keep_batch
            idx_ptr += B
    return keep_mask


def test_fixed_L0_across_modes(
    L0=0.54,
    modes=('feat','logits','logits_plus'),
    local_feature_extractor=None,
    local_classifier=None,
    cloud_cnn=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    device='cuda',
    offload_epochs=50,
    batch_size=128
):
    """
    Compare offload mechanism performance for a fixed budget L0 across different input modes.

    For each mode ('feat', 'logits', 'logits_plus'):
      1. Compute threshold b* that yields approximately L0 fraction local samples.
      2. Build dataset (features, logits, images) in that mode.
      3. Instantiate a new DeepOffloadMechanism with mode‑specific architecture.
      4. Train the offload decision network using train_deep_offload_mechanism.
      5. Measure offload decision accuracy on train and validation splits.

    Finally, print the tested L0, the theoretical local percentage (L0*100%),
    and the actual local percentage obtained from bk-values, then plot
    training vs validation offload accuracies for each mode.

    Args:
        L0 (float): Fraction of samples to process locally (0.0 to 1.0).
        modes (tuple[str]): Input modes to compare.
        local_feature_extractor (nn.Module): Local feature extractor network.
        local_classifier (nn.Module): Local classifier producing logits.
        cloud_cnn (nn.Module): Cloud classifier producing logits.
        train_loader (DataLoader): DataLoader for training split.
        val_loader (DataLoader): DataLoader for validation split.
        test_loader (DataLoader): DataLoader for test split (unused here).
        device (str): Compute device ('cuda' or 'cpu').
        offload_epochs (int): Number of epochs to train each offload mechanism.
        batch_size (int): Batch size for offload mechanism DataLoader.
    """
    # Prepare models and precompute inputs
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Compute features, bk-values, labels, logits, images once
    with torch.no_grad():
        feats, bks_list, labels, logits, images = \
            compute_bks_input_for_deep_offload(
                local_feature_extractor,
                local_classifier,
                cloud_cnn,
                train_loader,
                method=0,
                device=device
            )

    # Convert bk-values to tensor to compute actual local percentage
    bks_tensor = torch.tensor(bks_list, device=device)

    train_accs = []  # offload decision accuracy on training split
    val_accs = []    # offload decision accuracy on validation split
    
    best_mode_acc=0
    best_mode= None
    
    for mode in modes:
        # Compute the optimal bk-threshold b* for the given L0
        b_star = calculate_b_star(bks_tensor.tolist(), L0)

        # Build combined dataset for this mode
        combined = create_3d_data_deep(bks_tensor.tolist(), feats, logits, images, labels, input_mode=mode)
        ds = OffloadDatasetCNN(combined, b_star, input_mode=mode,include_bk=False,use_oracle_labels=False,local_clf=local_classifier,cloud_clf=cloud_cnn,device=device)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # Determine input shape for the offload mechanism
        if mode == 'logits':
            in_shape = (logits[0].shape[-1],)
        elif mode == 'logits_plus':
            in_shape = (logits[0].shape[-1] + 2,)
        else:
            sample_x = ds[0][0]
            in_shape  = tuple(sample_x.shape)

        # Instantiate DeepOffloadMechanism with mode-specific architecture
        if mode == 'feat':
            model = DeepOffloadMechanism(
                input_shape=in_shape,
                input_mode=mode,
                conv_dims=[64, 128, 256],
                num_layers=1,
                fc_dims=[128, 64, 1],
                dropout_p=0.35,
                latent_in=(1,)
            ).to(device)
        else:
            model = DeepOffloadMechanism(
                input_shape=in_shape,
                input_mode=mode,
                fc_dims=[128, 64, 32, 1],
                dropout_p=0.15,
                latent_in=(1,)
            ).to(device)

        # Setup optimizer & scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=5
        )

        # Train the offload mechanism
        train_deep_offload_mechanism(
            offload_mechanism=model,
            val_loader=val_loader,
            offload_optimizer=optimizer,
            offload_loader=loader,
            local_feature_extractor=local_feature_extractor,
            local_classifier=local_classifier,
            cloud_cnn=cloud_cnn,
            b_star=b_star,
            offload_scheduler=scheduler,
            input_mode=mode,
            device=device,
            epochs=offload_epochs,
            lr=1e-3,
            stop_threshold=0.95
        )

        # Evaluate offload decision accuracy on train split
        train_acc = evaluate_offload_decision_accuracy_CNN_train(
            loader,
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            model,
            b_star,
            input_mode=mode,
            device=device
        )
        # Evaluate offload decision accuracy on validation split
        val_acc = evaluate_offload_decision_accuracy_CNN_test(
            model,
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            val_loader,
            b_star,
            input_mode=mode,
            device=device
        )
        # save the best mode accuracy and name
        if val_acc > best_mode_acc:
            best_mode_acc=val_acc
            best_mode= mode
        
        actual_local_pct,overall_DDNN_acc=test_DDNN_with_optimized_rule(model,local_feature_extractor,local_classifier,
            cloud_cnn,test_loader,b_star,input_mode=mode,device=device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Mode={mode}: Offload Train Acc = {train_acc:.2f}%, Offload Val Acc = {val_acc:.2f}%")

    # Print summary with both theoretical and actual local percentages
    print(f"\nTested at L0 = {L0:.2f} (theoretical local % = {L0*100:.1f}%), "
          f"actual local % from bk-values = {actual_local_pct:.1f}%\n")

    # Plot training vs validation accuracy for each mode
    x = list(range(len(modes)))
    plt.figure(figsize=(8, 5))
    plt.plot(x, train_accs, 'o-', label='Train Accuracy')
    plt.plot(x, val_accs,   's-', label='Validation Accuracy')
    plt.xticks(x, modes)
    plt.xlabel('Input Mode')
    plt.ylabel('Offload Decision Accuracy (%)')
    plt.title(f'Offload Mechanism Performance at L0={L0:.2f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_difficulty_across_modes(
    L0_values,
    modes=('feat', 'logits', 'logits_plus'),
    local_feature_extractor=None,
    local_classifier=None,
    cloud_cnn=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,          
    device='cuda',
    offload_epochs=50,
    batch_size=128
):
    """
    Compare offload mechanism performance across a range of offload budgets (L0_values)
    and different input modes. For each L0 and mode, train a fresh DeepOffloadMechanism,
    evaluate its offload decision accuracy on both training and validation splits,
    and record the actual fraction of samples processed locally by calling
    test_DDNN_with_optimized_rule on the test_loader.

    Finally, plot train/validation accuracy curves for each mode versus each mode's
    own actual local %.
    """
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    # Precompute once: features, bk-values, labels, logits, images
    with torch.no_grad():
        feats, bks_list, labels, logits, images = compute_bks_input_for_deep_offload(
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            train_loader,
            method=0,
            device=device
        )

    # Prepare storage for metrics per mode
    train_accs = {mode: [] for mode in modes}
    val_accs   = {mode: [] for mode in modes}
    local_pcts = {mode: [] for mode in modes}

    for L0 in L0_values:
        b_star = calculate_b_star(bks_list, L0)

        for mode in modes:
            # build dataset & loader
            combined = create_3d_data_deep(bks_list, feats, logits, images, labels, input_mode=mode)
            ds       = OffloadDatasetCNN(combined, b_star, input_mode=mode,include_bk=False,use_oracle_labels=False,local_clf=local_classifier,cloud_clf=cloud_cnn,device=device)
            loader   = DataLoader(ds, batch_size=batch_size, shuffle=False)

            # infer input shape
            if mode == 'logits':
                in_shape = (logits[0].shape[-1],)
            elif mode == 'logits_plus':
                in_shape = (logits[0].shape[-1] + 2,)
            else:
                sample_x = ds[0][0]
                in_shape  = tuple(sample_x.shape)

            # instantiate model per mode
            if mode == 'feat':
                model = DeepOffloadMechanism(
                    input_shape=in_shape, input_mode=mode,
                    conv_dims=[64, 128, 256], num_layers=1,
                    fc_dims=[128, 64, 1], dropout_p=0.35,
                    latent_in=(1,)
                ).to(device)
            else:
                model = DeepOffloadMechanism(
                    input_shape=in_shape, input_mode=mode,
                    fc_dims=[128, 64, 32, 1], dropout_p=0.15,
                    latent_in=(1,)
                ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.7, patience=5
            )

            # train
            train_deep_offload_mechanism(
                offload_mechanism=model,
                val_loader=val_loader,
                offload_optimizer=optimizer,
                offload_loader=loader,
                local_feature_extractor=local_feature_extractor,
                local_classifier=local_classifier,
                cloud_cnn=cloud_cnn,
                b_star=b_star,
                offload_scheduler=scheduler,
                input_mode=mode,
                device=device,
                epochs=offload_epochs,
                lr=1e-3,
                stop_threshold=0.9
            )

            # evaluate train & val
            tr_acc = evaluate_offload_decision_accuracy_CNN_train(
                loader,
                local_feature_extractor,
                local_classifier,
                cloud_cnn,
                model,
                b_star,
                input_mode=mode,
                device=device
            )
            vl_acc = evaluate_offload_decision_accuracy_CNN_test(
                model,
                local_feature_extractor,
                local_classifier,
                cloud_cnn,
                val_loader,
                b_star,
                input_mode=mode,
                device=device
            )
            train_accs[mode].append(tr_acc)
            val_accs[mode].append(vl_acc)

            print(f"L0={L0:.2f} | Mode={mode} | TrainAcc={tr_acc:.2f}% | ValAcc={vl_acc:.2f}%")

            # measure actual local % on test set for this mode
            actual_local_pct,overall_DDNN_acc=test_DDNN_with_optimized_rule(model,local_feature_extractor,local_classifier,cloud_cnn,test_loader,b_star,input_mode=mode,device=device)
            
            local_pcts[mode].append(actual_local_pct)

    # summary
    print(f"\nTested L0 values: {L0_values}")
    for mode in modes:
        print(f"{mode} local%: {local_pcts[mode]}")

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in modes:
        ax.plot(local_pcts[mode], train_accs[mode], marker='o', linestyle='-', label=f"{mode} Train")
        ax.plot(local_pcts[mode], val_accs[mode],   marker='s', linestyle='--', label=f"{mode} Val")
    ax.set_xlabel('Actual Local %')
    ax.set_ylabel('Offload Decision Accuracy (%)')
    ax.set_title('Accuracy of Offload Mechanisms as a Function of Local %')
    ax.grid(True)
    ax.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
    plt.tight_layout()
    fig.savefig('modes_testing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_decision_comparison(
        bk_vals:      np.ndarray,
        decisions:    Dict[str, np.ndarray],
        *,
        b_star:       float = 0.0,
        show_all:     bool  = False,
        max_samples_display: int = 100,
        sample_ids:   List   = None,
        true_labels:  List   = None,
        class_names:  List   = None,
        annotate_labels: bool = True,
        jitter:       float  = 0.08,
        figsize:      Tuple[int, int] = (10, 3.2)
    ) -> Tuple[plt.Figure, np.ndarray]:
    """
    ------------------------------------------------------------------
    Visualises, on a single scatter-plot, how *each* decision rule
    (Oracle / Deep-offload / Entropy) assigns a sample to **local**
    or **cloud** execution as a function of its bk score.

    • **Colour encodes the rule**  (green = Oracle, orange = Deep-offload,
      purple = Entropy).
    • **Marker fill encodes the decision**  (hollow circle → local,
      filled circle → cloud).
    • An optional vertical dashed line (b_star) shows the oracle split.
    • Optionally, agreed-upon samples are shown in light-grey when
      `show_all=True`.
    • The function prints concise statistics to stdout and
      returns the indices of "disagreement" samples so you can inspect
      them later (e.g. look at the raw image/feature maps).

    Parameters
    ----------
    bk_vals
        1-D array (length **N**) with bk value for each test sample.
    decisions
        Dictionary with identical-length (N) int arrays.  Keys must include
        'Oracle', 'Deep-offload', and 'Entropy'.  Each array must contain
        **0 for local** and **1 for cloud**.
    b_star
        The oracle's threshold on bk.  Plotted as dashed grey line so you
        can instantly see on which side of the split a sample lies.
    show_all
        • False  → plot ONLY samples where at least two rules disagree
                   (focus on the interesting cases).
        • True   → overlay the full test set (agreed samples as faint squares
                   below the main axis) so you keep context.
    sample_ids
        Optional list/array of sample identifiers (filenames, indices, …) of
        length N.  They are *not* shown in the plot yet, but they propagate
        into the returned `mismatch_idx` for later debugging.
    jitter
        Standard deviation of vertical random noise applied to each point so
        overlapping dots remain distinguishable.
    figsize
        Matplotlib figure size.

    Returns
    -------
    fig
        The Matplotlib Figure object – keep / save it as you wish.
    mismatch_idx
        `np.ndarray` of integer indices (w.r.t. original test order) where
        at least one rule gives a different decision from the rest.

    Notes
    -----
    • The function prints, for **every** rule, the % of local/cloud
      decisions and the global mismatch percentage.
    • Because the arrays are flattened once, any torch tensors should be
      sent as `.cpu().numpy()` beforehand.
    ------------------------------------------------------------------
    """
    # ------------------------------------------------------------------
    # Sanity-check the input
    # ------------------------------------------------------------------
    method_names = list(decisions)                     # keep insertion order
    bk           = np.asarray(bk_vals).flatten()
    dec_stack    = np.vstack([np.asarray(decisions[m]).flatten()
                              for m in method_names])
    assert dec_stack.shape[1] == bk.size, \
        "bk_vals and all decision arrays must have the same length"

    # ------------------------------------------------------------------
    # Determine which samples are "interesting" (disagreement)
    # ------------------------------------------------------------------
    oracle_dec = decisions['Oracle']
    offload_dec = decisions['Deep-offload']
    mismatch_mask = (oracle_dec != offload_dec)

    # --- Sub-sampling for plotting only ---
    if show_all:
        total_N = len(bk)
        if (max_samples_display is not None) and (total_N > max_samples_display):
            # Pick random max_samples_display samples from ALL samples (agreed + mismatch)
            idx_keep = np.random.choice(np.arange(total_N), size=max_samples_display, replace=False)
            keep_mask = np.zeros(total_N, dtype=bool)
            keep_mask[idx_keep] = True
        else:
            keep_mask = np.ones(total_N, dtype=bool)
    else:
        if (max_samples_display is not None) and (mismatch_mask.sum() > max_samples_display):
            idx_mismatch = np.nonzero(mismatch_mask)[0]
            idx_keep = np.random.choice(idx_mismatch, size=max_samples_display, replace=False)
            keep_mask = np.zeros_like(mismatch_mask, dtype=bool)
            keep_mask[idx_keep] = True
        else:
            keep_mask = mismatch_mask

    bk_plot       = bk[keep_mask]
    dec_plot      = {m: dec_stack[i, keep_mask] for i, m in enumerate(method_names)}
    mismatch_idx  = np.nonzero(mismatch_mask)[0]

    # ------------------------------------------------------------------
    # Print quick stats to terminal
    # ------------------------------------------------------------------
    total_N = len(bk)
    print(f"[plot_decision_comparison] mismatch samples: "
          f"{mismatch_idx.size}/{total_N} ({100*mismatch_idx.size/total_N:.2f} %)")

    for m in method_names:
        n_local = int((decisions[m] == 0).sum())
        n_cloud = total_N - n_local
        print(f"  {m:<12}: local {100*n_local/total_N:5.2f}% | "
              f"cloud {100*n_cloud/total_N:5.2f}%")

    # ------------------------------------------------------------------
    # Build colour/marker maps
    # ------------------------------------------------------------------
    palette = {'Entropy':'mediumpurple', 'Deep-offload':'darkorange', 'Oracle':'seagreen'}
    y_map   = {m:i for i, m in enumerate(method_names)}

    # ------------------------------------------------------------------
    # Start plotting
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # (Optional) show agreed-upon samples in the background for context
    # if show_all:
    #     agree_mask = ~mismatch_mask
    #     if agree_mask.any():
    #         ax.scatter(bk[agree_mask],
    #                    np.full(agree_mask.sum(), -0.5),
    #                    s=30, c='lightgrey', alpha=0.6, marker='s',
    #                    label='agreed samples')

    # Plot each rule, differentiating local/cloud by fill
    for m, colour in palette.items():
        dec_arr = dec_plot[m]
        y_base  = y_map[m]
        # vertical jitter avoids identical-y stacking
        y_jit   = y_base + np.random.normal(0, jitter, size=dec_arr.size)

        for choice in (0, 1):          # 0 = local, 1 = cloud
            mask  = dec_arr == choice
            if not mask.any():
                continue
            face  = colour if choice else 'none'   # hollow for local
            edge  = colour
            ax.scatter(bk_plot[mask], y_jit[mask],
                       s=90, linewidths=1.2,
                       facecolors=face, edgecolors=edge, marker='o',
                       label=f"{m} {'cloud' if choice else 'local'}"
                             if choice == 0 else None)  # legend once per rule
            if annotate_labels and true_labels is not None:
                # find labels for the samples that are being plotted
                lbl_plot = np.asarray(true_labels)[keep_mask]
                for x, y, lab in zip(bk_plot[mask], y_jit[mask], lbl_plot[mask]):
                    txt = class_names[lab] if class_names is not None else str(lab)
                    ax.text(x, y+0.12, txt, fontsize=7, ha='center', va='bottom')

    # Draw oracle split
    ax.axvline(b_star, ls='--', c='grey', lw=1.2, label=f'b* = {b_star:.2f}')

    # Axis cosmetics
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(method_names, fontsize=12)
    ax.set_xlabel('bk', fontsize=12)
    ax.set_title("Decision comparison" +
                 ("" if show_all else " (mismatch samples)"),
                 fontsize=13, pad=10)
    ax.grid(axis='x', alpha=0.3)

    # Unique legend (avoid duplicates due to per-choice loop above)
    handles, labels = ax.get_legend_handles_labels()
    uniq            = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(),
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    # === NEW: Per-class agreement/mismatch stats plot ===
    if (true_labels is not None) and (class_names is not None):
        true_labels_arr = np.asarray(true_labels)
        oracle_dec = decisions['Oracle']
        offload_dec = decisions['Deep-offload']
        bk_arr = np.asarray(bk_vals)
        n_classes = len(class_names)

        agreed_mask = (oracle_dec == offload_dec)
        mismatch_mask = (oracle_dec != offload_dec)

        agreed_counts = []
        mismatch_counts = []
        agreed_bk_means = []
        mismatch_bk_means = []
        mismatch_percents = []
        total_percents = []
        total_counts = []

        N_total = len(true_labels_arr)
        for c in range(n_classes):
            idx_class = (true_labels_arr == c)
            n_class = idx_class.sum()
            total_counts.append(n_class)
            if n_class == 0:
                agreed_counts.append(0)
                mismatch_counts.append(0)
                agreed_bk_means.append(np.nan)
                mismatch_bk_means.append(np.nan)
                mismatch_percents.append(0.0)
                total_percents.append(0.0)
                continue
            agreed_idx = idx_class & agreed_mask
            mismatch_idx = idx_class & mismatch_mask
            n_agreed = agreed_idx.sum()
            n_mismatch = mismatch_idx.sum()
            agreed_counts.append(n_agreed)
            mismatch_counts.append(n_mismatch)
            agreed_bk_means.append(np.nanmean(bk_arr[agreed_idx]) if n_agreed > 0 else np.nan)
            mismatch_bk_means.append(np.nanmean(bk_arr[mismatch_idx]) if n_mismatch > 0 else np.nan)
            mismatch_percents.append(100.0 * n_mismatch / n_class)
            total_percents.append(100.0 * n_mismatch / N_total)

        x = np.arange(n_classes)
        width = 0.35
        fig2, ax1 = plt.subplots(figsize=(13, 5))
        rects1 = ax1.bar(x - width/2, agreed_counts, width, label='Agreed', color='seagreen', alpha=0.7)
        rects2 = ax1.bar(x + width/2, mismatch_counts, width, label='Mismatch', color='darkorange', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=30)
        ax1.set_ylabel('Number of samples')
        ax1.set_title('Agreement/Mismatch per class (Oracle vs Offload)')
        ax1.legend(loc='upper right')

        # Annotate mean bk above bars
        for i, (rect, mean_bk) in enumerate(zip(rects1, agreed_bk_means)):
            if not np.isnan(mean_bk):
                ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height()+2, f"bk={mean_bk:.2f}",
                         ha='center', va='bottom', fontsize=8, color='seagreen')
        for i, (rect, mean_bk) in enumerate(zip(rects2, mismatch_bk_means)):
            if not np.isnan(mean_bk):
                ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height()+2, f"bk={mean_bk:.2f}",
                         ha='center', va='bottom', fontsize=8, color='darkorange')

        # Add mismatch % as annotation above mismatch bars
        for i, rect in enumerate(rects2):
            ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height()+8,
                     f"{mismatch_percents[i]:.1f}%", ha='center', va='bottom', fontsize=9, color='black')

        # Optionally, show total % mismatches per class
        ax2 = ax1.twinx()
        ax2.plot(x, total_percents, 'o--', color='red', label='% mismatch (of total)')
        ax2.set_ylabel('% mismatch (of all samples)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper left')
        plt.tight_layout()
        # plt.show()

    return fig, mismatch_idx, fig2

@torch.no_grad()          # decorator to disable gradient computation
def gather_test_decisions(
        test_loader,
        local_feature_extractor,
        local_classifier,
        cloud_cnn,
        deep_offload_model,
        *,
        b_star: float,
        entropy_threshold: float,
        device: str = 'cuda',
        input_mode: str = 'logits',
        match_oracle_to: str | None = "Deep-offload",
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[int]]:
    """
    ------------------------------------------------------------------
    Runs **one** forward pass over the test_loader and collects:

      • bk values  (one per sample) – "difference in confidence" metric
        as used by the DDNN offloading papers.

      • Decisions from three rules:
        Oracle         – optimal rule relying on true labels.
        Deep-offload   – your trained offloading NN (logits → σ → 0/1).
        Entropy        – simple threshold on the local output entropy.

    Additionally, if *match_oracle_to* is set to the name of another rule
    (e.g. 'Deep-offload'), the Oracle decisions are post-processed so that
    they have **exactly the same number of local samples** as that rule.
    The least-confident (highest bk) Oracle-local samples are flipped
    to cloud to achieve the match.

    The function is careful to stay in *eval* mode and preserve current
    `torch.no_grad()` context so it adds zero overhead to your training
    script.

    Parameters
    ----------
    test_loader
        A PyTorch DataLoader yielding **(images, labels)**.
    local_feature_extractor
        Trunk CNN running on device for local inference (produces feature maps).
    local_classifier
        Small head that turns local features into class logits.
    cloud_cnn
        Remote CNN that also takes local features and outputs class logits.
    deep_offload_model
        The learned decision network (e.g. an MLP) which outputs a single
        logit indicating *cloud vs local*.
    b_star
        Threshold on bk used by the oracle rule.
    entropy_threshold
        If the *normalized entropy* of local logits exceeds this value
        ⇒ offload to cloud.
    device
        'cuda' or 'cpu' – ALL tensors are moved here before forward pass.
    input_mode
        What you feed into `deep_offload_model`:
        • 'logits'       : raw local logits  (default & simplest).
        • 'feat'         : local feature maps (the high-dim tensor).
        • 'logits_plus'  : concatenates logits + margin + entropy.

    match_oracle_to     : str or None.  If str, must be a key of *decisions*
                          (typically 'Deep-offload' or 'Entropy').

    Returns
    -------
    bk_vals
        NumPy array shape (N,) with bk for each test sample.
    decisions_dict
        Dict with keys {'Oracle','Deep-offload','Entropy'}, each value
        a NumPy array shape (N,) of 0/1 decisions.
    img_indices
        List of int indices (0…N-1) mapping directly back to your
        test_loader order.  Replace with filenames if you prefer.

    Notes
    -----
    • If you need the *same local/cloud ratio* across rules, use the
      resulting decisions to estimate the desired percentage and adjust
      `entropy_threshold` accordingly in a second pass.
    • The code assumes that *local_feature_extractor* is frozen and both
      classifiers operate on the SAME feature tensor to avoid redundant
      GPU computation.
    ------------------------------------------------------------------
    """
    # Put every sub-module in evaluation mode
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    deep_offload_model.eval()

    # Lists will be concatenated at the end (avoids expensive realloc)
    bk_vals, dec_or, dec_deep, dec_ent, img_ids = [], [], [], [], []
    all_labels = []  
    idx_global = 0  # running counter to assign unique indices

    for images, labels in test_loader:
        bs = images.size(0)                              # mini-batch size
        images, labels = images.to(device), labels.to(device)

        # --------------------------------------------------------------
        # Forward pass (local & cloud)
        # --------------------------------------------------------------
        features     = local_feature_extractor(images)   # shared trunk
        logits_local = local_classifier(features)        # local decision
        logits_cloud = cloud_cnn(features)               # remote decision

        # --------------------------------------------------------------
        # bk computation (lower bk → better local)
        # --------------------------------------------------------------
        p_loc = torch.softmax(logits_local, 1)[range(bs), labels]
        p_cld = torch.softmax(logits_cloud, 1)[range(bs), labels]
        bk    = (1 - p_loc) - (1 - p_cld)                # ∆ loss wrt ideal
        bk_vals.append(bk.cpu())

        # --------------------------------------------------------------
        # Oracle decision (needs true labels)
        # --------------------------------------------------------------
        dec_oracle_batch = my_oracle_decision_function(
                                logits_local, logits_cloud, labels,
                                b_star=b_star)
        dec_or.append(dec_oracle_batch.cpu())

        # --------------------------------------------------------------
        # Deep-offload decision (learned rule)
        # --------------------------------------------------------------
        if input_mode == 'logits':
            dom_input = logits_local.detach()
        elif input_mode == 'feat':
            dom_input = features.detach()
        elif input_mode == 'logits_plus':
            probs   = torch.softmax(logits_local, 1)
            top2    = torch.topk(probs, 2).values
            margin  = top2[:, 0] - top2[:, 1]            # confidence gap
            entropy_n = (-(probs*torch.log(probs + 1e-9)).sum(1) /
                         math.log(probs.size(1)))         # normalised entropy
            dom_input = torch.cat([logits_local,
                                   margin.unsqueeze(1),
                                   entropy_n.unsqueeze(1)], dim=1)
        else:
            raise ValueError("input_mode must be 'logits'|'feat'|'logits_plus'")

        logits_dec = deep_offload_model(dom_input)       # (bs, 1)
        dec_batch  = (torch.sigmoid(logits_dec).squeeze(1) > threshold).float()
        dec_deep.append(dec_batch.cpu())

        # --------------------------------------------------------------
        # Entropy decision (hand-crafted rule)
        # --------------------------------------------------------------
        probs_local = torch.softmax(logits_local, 1)
        entropy_raw = -(probs_local * torch.log(probs_local + 1e-9)).sum(1)
        entropy_n   = entropy_raw / math.log(probs_local.size(1))
        dec_ent_batch = (entropy_n > entropy_threshold).float()
        dec_ent.append(dec_ent_batch.cpu())

        # --------------------------------------------------------------
        # Book-keeping: global index or filename
        # --------------------------------------------------------------
        img_ids.extend(range(idx_global, idx_global + bs))
        idx_global += bs
        all_labels.append(labels.cpu())
    # ------------------------------------------------------------------
    # Concatenate lists into flat NumPy arrays
    # ------------------------------------------------------------------
# Concatenate lists ⇒ flat numpy arrays
    # ------------------------------------------------------------------
    bk_vals_np   = torch.cat(bk_vals).numpy()
    dec_or_np    = torch.cat(dec_or).numpy()
    dec_deep_np  = torch.cat(dec_deep).numpy()
    dec_ent_np   = torch.cat(dec_ent).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    # ------------------------------------------------------------------
    # OPTIONAL: balance Oracle so it matches another rule's local %
    # ------------------------------------------------------------------
    if match_oracle_to is not None:
        reference_dec = {
            "Deep-offload": dec_deep_np,
            "Entropy":      dec_ent_np
        }.get(match_oracle_to, None)

        if reference_dec is None:
            raise ValueError("match_oracle_to must be 'Deep-offload', 'Entropy' or None")

        target_local = int((reference_dec == 0).sum())          # desired count
        oracle_local_idx = np.where(dec_or_np == 0)[0]
        delta = len(oracle_local_idx) - target_local

        if delta > 0:  # need to flip some oracle-local → cloud
            # pick the delta local indices with *largest* bk (least sure)
            sort_idx = oracle_local_idx[np.argsort(bk_vals_np[oracle_local_idx])[::-1]]
            flip_idx = sort_idx[:delta]
            dec_or_np[flip_idx] = 1

        # if Oracle already has <= desired local, we leave it as-is
        # (could flip cloud→local with smallest bk if ever needed)

    # ------------------------------------------------------------------
    # Build final dictionary and return
    # ------------------------------------------------------------------
    decisions_dict = {
        "Oracle":       dec_or_np,
        "Deep-offload": dec_deep_np,
        "Entropy":      dec_ent_np
    }



    return bk_vals_np, decisions_dict, img_ids,all_labels_np
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

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(epochs_DDNN: int = 50,
         epochs_optimization: int = 70,
         batch_size: int = 128,
         local_weight: float = 0.7,
         input_mode: str = 'logits_plus'):
    """
    Train DDNN + offload rule for every dataset in DATASET_INFO
    and store a separate plot via test_difficulty_vs_L0.
    """

    # ➊ Dataset list & number-of-classes map
    DATASET_INFO = {
        'cifar10' : 10,   # baseline easy
        'cifar100': 100,  # harder 100-class variant
        'cinic10' : 10,   # CIFAR/ImageNet mix (32×32)
        'svhn'    : 10,   # street-view digits (32×32)
        'gtsrb32' : 43    # traffic signs (32×32)
    }

    # range of local percentages (L0) to test
    L0_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                 0.55, 0.585, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]

    # ➋ Loop over every dataset
    for dataset_name, n_classes in DATASET_INFO.items():
        print(f"\n================  {dataset_name.upper()}  =================")

        # -- update the global before model creation
        global NUM_CLASSES
        NUM_CLASSES = n_classes

        # (α) Data loaders --------------------------------------------------
        train_loader, val_loader, test_loader = load_data(
            batch_size=batch_size,
            dataset=dataset_name)

        # (β) Initialise models & optimisers -------------------------------
        (local_feature_extractor,
         local_classifier,
         cloud_cnn,
         offload_mechanism) = initialize_models()

        (cnn_optimizer,
         offload_optimizer) = initialize_optimizers(
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            offload_mechanism)

        # (γ) Train DDNN ----------------------------------------------------
        train_DDNN(train_loader,
                   local_feature_extractor,
                   local_classifier,
                   cloud_cnn,
                   cnn_optimizer,
                   local_weight,
                   epochs_DDNN)

        # (δ) Difficulty-vs-L0 curve + plot --------------------------------
        test_difficulty_vs_L0(
            L0_values,
            local_feature_extractor,
            local_classifier,
            cloud_cnn,
            train_loader,
            test_loader,
            val_loader,
            device=device,
            offload_epochs=epochs_optimization,
            batch_size=batch_size,
            input_mode=input_mode,
            dataset_name=dataset_name)   # used in plot filename/title


if __name__ == '__main__':
    main()

    
    