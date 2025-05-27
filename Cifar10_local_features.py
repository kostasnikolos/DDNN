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
# Set device to GPU for faster training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        
        self.fc2 = nn.Linear(64, 10)
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
        self.fc3 = nn.Linear(64, 10)

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
class OffloadMechanismCNN_v3(nn.Module):
    """
    A deeper offload mechanism implemented as a CNN.
    This network takes as input the local features with shape (batch, 32, 16, 16)
    (i.e., the output from the LocalFeatureExtractor) and processes them through 6 convolutional blocks.
    The architecture is as follows:
    
    Block 1: Conv2d (32->64), BatchNorm, LeakyReLU, MaxPool (output: (64, 8, 8))
    Block 2: Conv2d (64->128), BatchNorm, LeakyReLU, MaxPool (output: (128, 4, 4))
    Block 3: Conv2d (128->256), BatchNorm, LeakyReLU (output: (256, 4, 4))
    Block 4: Conv2d (256->256), BatchNorm, LeakyReLU, MaxPool (output: (256, 2, 2))
    Block 5: Conv2d (256->512), BatchNorm, LeakyReLU (output: (512, 2, 2))
    Block 6: Conv2d (512->512), BatchNorm, LeakyReLU, AdaptiveAvgPool to (512, 1, 1)
    
    Then the output is flattened (512) and passed through fully connected layers:
      FC1: 512 -> 256 (ReLU)
      FC2: 256 -> 128 (ReLU)
      FC3: 128 -> 1  (linear output, used with BCEWithLogitsLoss)
    """
    def __init__(self):
        super(OffloadMechanismCNN_v3, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 8, 8)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (128, 4, 4)
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # No pooling here
        
        # Block 4
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (256, 2, 2)
        
        # Block 5
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        # No pooling
        
        # Block 6: Use Adaptive Average Pooling to get fixed output (1x1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (512, 1, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        # x: shape (batch, 32, 16, 16)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.pool1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.pool2(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = self.pool4(x)
        
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01)
        
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.01)
        x = self.adapt_pool(x)  # x shape: (batch, 512, 1, 1)
        
        x = x.view(x.size(0), -1)  # Flatten to (batch, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
class OffloadMechanism(nn.Module):

    """
    The offload mechanism is an MLP that decides whether to send a sample
    to the cloud or to keep it locally.
    input_size = 8192 means we flatten the local features (32,16,16 => 8192).
    Output is a single logit for BCEWithLogitsLoss (binary classification).
    """
    def __init__(self, input_size=8192):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x.shape = (batch_size, 8192)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)  # BCEWithLogitsLoss
        return logits
    
        
def load_data(batch_size=32):    
    """
    Loads CIFAR-10 dataset, splits into train/val/test,
    and returns DataLoaders with given batch_size.
    """
    # Define a transform to normalize the data (same as dividing by 255.0)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL images to PyTorch tensors (scales the values from [0 , 255] to [0 ,1 ])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the values from [0 , 1] to [-1 , 1] for all three channels
    ])

    # Load the CIFAR10 dataset from torchvision and apply transformation
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split the training data into training and validation sets
    validation_split = 0.1
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # oeganize the data sets in batches and shuffle training data
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
    offload_mechanism = OffloadMechanism().to(device)
    return  offload_mechanism

# Initialize models
def initialize_models():
    """
        Initializes all models, the DDNN models (local_feature_extractor,local_classifier,cloud), the offload mechanism model and returns them
    
    """
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

def initialize_DDNN_optimizers(local_feature_extractor, local_classifier, cloud_cnn, lr=0.001):
    """
        Initializes DDNN optimizers
    """
    cnn_optimizer = optim.Adam(list(local_feature_extractor.parameters()) + list(local_classifier.parameters()) + list(cloud_cnn.parameters()), lr=lr)
    return  cnn_optimizer

def initialize_offload_optimizer(offload_mechanism, Ir=0.001):
    """ Initializes offload mechanism optimizer"""
    offload_optimizer = optim.Adam(offload_mechanism.parameters(),0.001 ) #καλο το 0.005

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
    local_weight= local_weight
    
   
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
    print(f' LO USED: {L0}')
    print(f"Mean of bks: {np.mean(all_bks)}")
    print(f"Median of bks: {np.median(all_bks)}")
    print(f"Standard deviation of bks: {np.std(all_bks)}")
    print(f"Max of bks: {np.max(all_bks)}, Min of bks: {np.min(all_bks)}")
    
    #  Plot histogram for visualization
    # plt.hist(all_bks, bins=50, alpha=0.75, color='blue')
    # plt.title('Distribution of bks')
    # plt.xlabel('bk values')
    # plt.ylabel('Frequency')
    # plt.show()
    
    # Calculate b* as the L0 percentile of all bks
    b_star = np.percentile(all_bks, L0 * 100)
    print(f"Calculated b*: {b_star}")
    
    return b_star
def test_DDNN_with_entropy(local_feature_extractor, local_classifier, cloud_cnn, data_loader, threshold ):
    """
    Tests the DDNN using an Entropy-based decision on local network.
    If the local classifier's normalized entropy < threshold => local
    Otherwise => cloud
    Returns local_acc, cloud_acc, overall_acc, #samples local, #samples cloud
    """
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    local_correct = 0
    cloud_correct = 0
    total_images = 0
    local_classified = 0  # To track how many samples were classified by Local CNN
    cloud_classified = 0  # To track how many samples were classified by Cloud CNN
    
    #  track the U values
    all_U_values = []  # List to store all U values
    all_entropy_values = []  # List to store all entropy values

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass through local CNN
            local_features = local_feature_extractor(images)
            

            # entropy= calculate_entropy(local_classifier(local_features))
            entropy= calculate_normalized_entropy(local_classifier(local_features))
            
            
            all_entropy_values.extend(entropy.cpu().numpy())
            

            confident= entropy< threshold
            


            # check if there is any confident sample to classify locally
            if confident.any():
                #  find the confident predictions in current batch
                confident_predictions = local_classifier(local_features[confident]) # No dropout its the final classification
                # find the condifent labels in current batch
                confident_labels= labels[confident]
                #  compare the predictions with the true ones and add the the correct ones from the badge
                local_correct += (confident_predictions.argmax(1) == confident_labels).sum().item()
                local_classified += confident.sum().item()  # Count samples classified by Local CNN
       

            # Cloud predictions (for uncertain samples)
            if (~confident).any():
                # fint the uncertain predicted samples in the current batch
                uncertain_features = local_features[~confident]
                # find the uncertain predicted samples labels in the current batch
                uncertain_labels = labels[~confident]
                # pass them trhough the cloud and classify
                cloud_predictions = cloud_cnn(uncertain_features)
                # save the correct predictions from  the batch
                cloud_correct += (cloud_predictions.argmax(1) == uncertain_labels).sum().item()
                
                cloud_classified += (~confident).sum().item()  # Count samples classified by Cloud CNN


            total_images += labels.size(0)
            

        
    local_accuracy = local_correct / local_classified * 100 if local_classified > 0 else 0
    cloud_accuracy = cloud_correct / cloud_classified * 100 if cloud_classified > 0 else 0
    overall_accuracy = (local_correct + cloud_correct) / total_images * 100

    # Print the number of samples classified by each network
    print(f"Samples classified by Local CNN: {local_classified}")
    print(f"Samples classified by Cloud CNN: {cloud_classified}")
    print(f"Entropy overall acuraccy: {overall_accuracy}")

      # Calculate and print the mean of all entropy values
    mean_entropy= np.mean(all_entropy_values)  # Use NumPy to compute the mean
    print(f"Mean Uncertainty (entropy) across all batches: {mean_entropy}")

    
    return local_accuracy, cloud_accuracy, overall_accuracy , local_classified , cloud_classified

    

def evaluate_with_offload_mechanism(offload_mechanism, 
                                    local_feature_extractor, 
                                    local_classifier, 
                                    cloud_cnn, 
                                    data_loader,  
                                    threshold=0.5):
    """
    Evaluate the optimization rule using the offloading mechanism,
    Pass every sample through the local feature extractor, decide where to classify with the offload mechanism 
    Calculate the accuracies
    Offload rule: offload_probs => if > threshold => Cloud, else Local.
    b_star isn't directly used here, unless you do something special.
    This function:
      1) get local_features => flatten => offload_mechanism => offload_probs
      2) if offload_probs> threshold => send to cloud, else local
    Then measure accuracy
    """
    offload_mechanism.eval()
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    local_correct, cloud_correct, total_samples = 0, 0, 0
    local_classified, cloud_classified = 0, 0

    #  Pass all samples through the local feature extractor, 
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # === 1) Υπολογισμός Local Features από την εικόνα ===
            local_feats = local_feature_extractor(images)  # shape: (batch, 32, 16, 16) π.χ.

            # === 2) Flatten για το OffloadMechanism αν περιμένει input_size=8192 ===
            local_feats_flat = local_feats.view(local_feats.size(0), -1)  # (batch, 8192)

            # === 3) Υπολογισμός offloading probabilities από τα local features ===
            logits = offload_mechanism(local_feats_flat)        # (batch, 1)
            offload_probs = torch.sigmoid(logits).squeeze(1)    # shape: (batch,)

            # === 4) Αποφάσισε Remote/Local με βάση threshold ===
            classify_remotely = (offload_probs > threshold)
            classify_locally = ~classify_remotely

            # === 5) Επεξεργασία δειγμάτων που πάνε Local ===
            if classify_locally.any():
                local_outputs = local_classifier(local_feats[classify_locally])
                local_correct += (local_outputs.argmax(dim=1) == labels[classify_locally]).sum().item()
                local_classified += classify_locally.sum().item()

            # === 6) Επεξεργασία δειγμάτων που πάνε Cloud ===
            if classify_remotely.any():
                cloud_outputs = cloud_cnn(local_feats[classify_remotely])
                cloud_correct += (cloud_outputs.argmax(dim=1) == labels[classify_remotely]).sum().item()
                cloud_classified += classify_remotely.sum().item()

            total_samples += labels.size(0)

    # === Υπολογισμός accuracies ===
    local_acc = (local_correct / local_classified * 100) if local_classified > 0 else 0
    cloud_acc = (cloud_correct / cloud_classified * 100) if cloud_classified > 0 else 0
    overall_acc = (local_correct + cloud_correct) / total_samples * 100

    print(f"Samples classified by Local CNN: {local_classified}")
    print(f"Samples classified by Cloud CNN: {cloud_classified}")
    print(f"Local Accuracy: {local_acc:.2f}%, "
          f"Cloud Accuracy: {cloud_acc:.2f}%, "
          f"Overall Accuracy: {overall_acc:.2f}%")

    return local_acc, cloud_acc, overall_acc, local_classified, cloud_classified


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

def calculate_bks(images, labels, local_classifier,local_feature_extractor,cloud_cnn):
    """
    Υπολογίζει τις τιμές b_k για τα δεδομένα (images, labels).
    """
    with torch.no_grad():
        local_probs = local_classifier(local_feature_extractor(images))
        cloud_probs = cloud_cnn(local_feature_extractor(images))
        
        local_cost = 1 - local_probs.gather(1, labels.view(-1, 1)).squeeze()
        cloud_cost = 1 - cloud_probs.gather(1, labels.view(-1, 1)).squeeze()
        
        bks = local_cost - cloud_cost
    return bks.cpu().numpy()

def create_3d_dataset(all_bks, all_local_features):
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



class OffloadDataset(Dataset):
    def __init__(self, combined_data, b_star):
        self.combined_data = combined_data
        self.b_star = b_star

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        global_idx, bk_value, figure_or_feat = self.combined_data[idx]
        label = 1.0 if bk_value > self.b_star else 0.0
        
        # Μετατροπή σε PyTorch tensor
        x_tensor = torch.tensor(figure_or_feat, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.float32)
        return x_tensor, y_tensor
    
def compare_entropy_with_offload_results(
    offload_model,
    local_feature_extractor,
    local_classifier,
    test_loader,
    threshold_entropy=0.2,
    device='cuda'
):
    """
    Evaluate the offload mechanism (binary decision) in comparison
    with the entropy-based local decision.

    Steps:
    1) For each batch, extract local features.
    2) Flatten those features and pass them to the offload_model to decide
       if the sample should go to the cloud (offload) or stay local.
    3) Simultaneously, compute the normalized entropy from the local classifier
       (not flattened, because the local classifier is directly on feature maps).
    4) Count how many samples each method (offload vs. entropy) sends locally
       or to the cloud.
    5) Compare the decision from the offload mechanism with the decision from entropy
       to see how often they agree/disagree.

    Args:
        offload_model: (nn.Module) The trained offload mechanism model (binary).
        local_feature_extractor: (nn.Module) The local feature extractor network.
        local_classifier: (nn.Module) The local classifier network.
        test_loader: (DataLoader) The test set loader.
        threshold_entropy (float): If local entropy < threshold_entropy => local.
        device (str): 'cuda' or 'cpu'.

    Returns:
        Prints out the counts of local/cloud decisions for both methods
        and their agreement percentage.
    """
    
    
    
    offload_model.eval()
    local_feature_extractor.eval()
    local_classifier.eval()

    total_samples = 0
    offload_local_count = 0
    offload_cloud_count = 0
    entropy_local_count = 0
    entropy_cloud_count = 0
    agreement_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1) Υπολογίζουμε local_features
            local_feats = local_feature_extractor(images)  # (batch, 32, 16, 16)
            # Flatten για το offload_model
            local_feats_flat = local_feats.view(local_feats.size(0), -1)  # (batch, 8192)

            # === Απόφαση βάσει offload_model ===
            logits = offload_model(local_feats_flat)  # (batch,1)
            offload_probs = torch.sigmoid(logits).squeeze(1)  # (batch,)
            offload_decision = (offload_probs > 0.5)          # True => Cloud

            offload_local_count += (~offload_decision).sum().item()
            offload_cloud_count += offload_decision.sum().item()

            # === Απόφαση βάσει Entropy ===
            # Εδώ περνάμε τα local_features (ΔΕΝ τις έχουμε flatten) στον local_classifier
            local_out = local_classifier(local_feats)  # (batch, 10)
            entropy = calculate_normalized_entropy(local_out)
            confident = (entropy < threshold_entropy)  # True => Local

            entropy_local_count += confident.sum().item()
            entropy_cloud_count += (~confident).sum().item()

            # === Agreement ===
            decision_entropy = (~confident).int()     # 0=>Local, 1=>Cloud
            decision_offload = offload_decision.int() # 0=>Local, 1=>Cloud
            same_decision = (decision_entropy == decision_offload)
            agreement_count += same_decision.sum().item()

            total_samples += images.size(0)

    agreement_percentage = 100.0 * agreement_count / total_samples
    print("===== Αποτελέσματα Σύγκρισης Offload vs Entropy =====")
    print(f"Συνολικά δείγματα       : {total_samples}")
    print(f"Offload -> Local        : {offload_local_count} ({100.0*offload_local_count/total_samples:.1f}%)")
    print(f"Offload -> Cloud        : {offload_cloud_count} ({100.0*offload_cloud_count/total_samples:.1f}%)")
    print(f"Entropy -> Local        : {entropy_local_count} ({100.0*entropy_local_count/total_samples:.1f}%)")
    print(f"Entropy -> Cloud        : {entropy_cloud_count} ({100.0*entropy_cloud_count/total_samples:.1f}%)")
    print(f"Συμφωνία Αποφάσεων      : {agreement_count} / {total_samples} ({agreement_percentage:.2f}%)")
    print("=====================================================")
def compute_bks_for_dataset_unified(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    data_loader,
    method=0,
    beta=0.2,
    device='cuda',
    return_features=True,
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
    import torch
    import torch.nn.functional as F
    import numpy as np

    # For method=3, we need a function to compute local entropy from raw logits
    def calculate_normalized_entropy_from_logits(logits):
        """
        Computes normalized entropy in [0..1] from raw logits:
            H_norm = - sum(p * log(p)) / log(#classes),
        where p = softmax(logits).
        """
        eps = 1e-9
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        return -torch.sum(probs * torch.log(probs + eps), dim=1) / np.log(num_classes)
    
    def get_top2_probs(prob_tensor):
        """Επιστρέφει p1, p2 (με descending σειρά) για κάθε δείγμα."""
        sorted_p, _ = torch.sort(prob_tensor, descending=True, dim=1)
        p1 = sorted_p[:, 0]
        p2 = sorted_p[:, 1]
        return p1, p2

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    all_features_list = []
    all_bks_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            # 1) Forward: local feats => local_out => cloud_out
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)  # raw logits
            cloud_out = cloud_cnn(local_feats)         # raw logits

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
                    cloud_cost[idx_tie] += beta * diff_probs

            elif method == 3:
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
                mask_l1_c1 = local_correct_mask & cloud_correct_mask
                idx = mask_l1_c1.nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    local_out_tie = local_out[idx]
                    local_entropy = calculate_normalized_entropy_from_logits(local_out_tie)
                    # If local_entropy < threshold => local=0,cloud=1
                    cond_mask = (local_entropy < entropy_threshold)
                    idx_true = idx[cond_mask]
                    local_cost[idx_true] = 0.0
                    cloud_cost[idx_true] = 1.0

                    # else => local=1, cloud=0
                    idx_false = idx[~cond_mask]
                    local_cost[idx_false] = 1.0
                    cloud_cost[idx_false] = 0.0
                    
            elif method == 4:
                # 1) Βρίσκουμε top2 για local και cloud
                p1_local, p2_local = get_top2_probs(local_probs)
                p1_cloud, p2_cloud = get_top2_probs(cloud_probs)

                marginLocal = p1_local - p2_local
                marginCloud = p1_cloud - p2_cloud

                # 2) cost_local = 1 - marginLocal
                #    cost_cloud = 1 - marginCloud
                cost_local = marginLocal
                cost_cloud = marginCloud

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

            # 5) bk = local_cost - cloud_cost
            bks = local_cost - cloud_cost
            all_bks_list.append(bks.cpu().numpy())

            # Optionally store features
            if return_features:
                all_features_list.append(local_feats.cpu().numpy())

    # 6) Convert to NumPy
    all_bks_array = np.concatenate(all_bks_list, axis=0)
    if return_features:
        all_features_array = np.concatenate(all_features_list, axis=0)
        return all_features_array, all_bks_array
    else:
        return None, all_bks_array


def compute_bks_for_dataset_original(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    data_loader,
    beta=0.2,      # Factor that modifies the cloud cost in case of tie
    device='cuda',
    return_features=True
):
    """
    Compute 'bk' values (local vs. cloud cost difference) for each sample
    in the data_loader. This method is similar to the binary variant but
    specifically handles the tie case (both local and cloud correct) by
    adjusting the cloud cost:

      if local and cloud both correct:
         cloud_cost <- (1 - p_cloud) + beta*(p_cloud - p_local)

    Steps:
      1) For each batch, extract features, compute local_out, cloud_out.
      2) Convert logits to probabilities via softmax.
      3) Compute local_cost = 1.0 - p_local.
         Compute cloud_cost = 1.0 - p_cloud.
      4) Check if local and cloud both are correct. If so, we add
         beta*(p_cloud - p_local) to the cloud_cost, effectively penalizing
         the cloud more if it is only slightly more confident.
      5) Compute bk = local_cost - cloud_cost for each sample.
      6) Optionally, if return_features=True, store the local_feats
         for later training of an offload mechanism.

    Args:
        local_feature_extractor (nn.Module): The local feature extractor network.
        local_classifier (nn.Module): The local classifier network.
        cloud_cnn (nn.Module): The cloud (more powerful) CNN network.
        data_loader (DataLoader): The dataset over which to compute these costs.
        beta (float): The penalty factor to add to the cloud cost in tie cases.
        device (str): 'cuda' or 'cpu' device.
        return_features (bool): Whether to return local feature maps as NumPy arrays.

    Returns:
        If return_features=True:
            (all_local_features, all_bks) as NumPy arrays.
          Otherwise:
            (None, all_bks).
          all_bks is shape (N,), one bk per sample.
    """
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()

    all_features_list = []
    all_bks_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # 1) Forward: extract local features, then local and cloud outputs.
            local_feats = local_feature_extractor(images)  # shape: (batch, C, H, W)
            local_out = local_classifier(local_feats)       # shape: (batch, num_classes)
            cloud_out = cloud_cnn(local_feats)

            # 2) Convert to probabilities via softmax
            local_probs = F.softmax(local_out, dim=1)       # shape: (batch, num_classes)
            cloud_probs = F.softmax(cloud_out, dim=1)

            batch_size = labels.size(0)
            # Probability that local/cloud picked the correct label
            local_prob_correct = local_probs[range(batch_size), labels]
            cloud_prob_correct = cloud_probs[range(batch_size), labels]

            # 3) Initial cost definitions
            local_cost = 1.0 - local_prob_correct
            cloud_cost = 1.0 - cloud_prob_correct

            # 4) Check if local/cloud are correct
            local_pred = local_out.argmax(dim=1)
            cloud_pred = cloud_out.argmax(dim=1)
            local_correct_mask = (local_pred == labels)
            cloud_correct_mask = (cloud_pred == labels)

            # Tie case: both local and cloud correct => adjust cloud cost
            both_correct_mask = local_correct_mask & cloud_correct_mask
            both_correct_indices = both_correct_mask.nonzero(as_tuple=True)[0]
            if len(both_correct_indices) > 0:
                # difference in probabilities
                diff_probs = cloud_prob_correct[both_correct_indices] - local_prob_correct[both_correct_indices]
                # Increase cloud_cost by beta*(cloud_prob - local_prob)
                # IF YOU WANT TO APPLY PENTALTY DIFFERENCE HERE YOU ARE
                # cloud_cost[both_correct_indices] += beta * diff_probs

            # 5) Calculate bk = local_cost - cloud_cost
            bks = local_cost - cloud_cost

            # 6) Store results
            all_bks_list.append(bks.cpu().numpy())
            if return_features:
                all_features_list.append(local_feats.cpu().numpy())

    # 7) Convert to NumPy arrays
    all_bks_array = np.concatenate(all_bks_list, axis=0)
    if return_features:
        all_features_array = np.concatenate(all_features_list, axis=0)
        return all_features_array, all_bks_array
    else:
        return None, all_bks_array

def train_offload_mechanism_local(offload_model, optimizer, offload_loader, epochs=10, lr=1e-3, device='cuda'):
    """
    Train a binary offload mechanism model (BCEWithLogitsLoss) that decides
    whether a sample goes local or cloud, given its (flattened) local features.

    Args:
        offload_model (nn.Module): The offload network (takes 8192-dim features).
        optimizer (torch.optim.Optimizer): The optimizer for offload_model parameters.
        offload_loader (DataLoader): Contains (local_feats, label=0/1).
        epochs (int): Number of training epochs.
        lr (float): Learning rate (if needed).
        device (str): 'cuda' or 'cpu'.

    Returns:
        None. (Prints epoch losses.)
    """
    criterion = nn.BCEWithLogitsLoss()
  
    for epoch in range(epochs):
        offload_model.train()
        total_loss = 0

        for local_feats, labels in offload_loader:
            # local_feats: shape (batch, 8192) if flattened
            local_feats = local_feats.to(device)
            # labels are 0/1 => shape (batch,)
            labels = labels.unsqueeze(1).to(device)  # shape: (batch,1) for BCE

            optimizer.zero_grad()
            logits = offload_model(local_feats)   # shape: (batch,1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(offload_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Offload Training Loss: {avg_loss:.4f}")

def train_offload_mechanism(offload_model, optimizer, train_loader, epochs, device='cuda', target_val_acc=80.0):
    """
    Train the offload mechanism with intermediate validation on the training data.
    After each training epoch, the model is evaluated on the training dataset.
    If the offload decision accuracy (computed by comparing the offload model's decision 
    to the ground truth computed via bk values) reaches or exceeds target_val_acc,
    the training stops early.
    
    Ground truth for each sample is determined by:
        if bk < b_star -> label 0 (local)
        if bk >= b_star -> label 1 (offload to cloud)
    (It is assumed that the training loader used here is the one generated for offload training)
    
    Args:
        offload_model (nn.Module): The offload network (expects input features, e.g., flattened features).
        optimizer (torch.optim.Optimizer): Optimizer for the offload model.
        train_loader (DataLoader): DataLoader for training offload mechanism.
        epochs (int): Maximum number of training epochs.
        device (str): 'cuda' or 'cpu'.
        target_val_acc (float): Target training accuracy (%) to reach before early stopping.
    
    Returns:
        None. (Prints epoch losses and training accuracy; stops training early if target is met.)
    """
    criterion = nn.BCEWithLogitsLoss()
    
    # We assume that train_loader provides samples where the labels correspond to the ground truth 
    # offload decision computed via bk values.
    for epoch in range(epochs):
        offload_model.train()
        total_loss = 0
        for features, labels in train_loader:
            # features: shape (batch, 32,16,16) OR (batch, flattened) depending on dataset construction
            # In our case, we expect the offload training dataset to have features already in the correct shape.
            features, labels = features.to(device), labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = offload_model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Offload Training Loss: {avg_loss:.4f}")
        
        # Evaluate on the training set (acting as validation)
        offload_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in train_loader:
                features, labels = features.to(device), labels.unsqueeze(1).to(device)
                logits = offload_model(features)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        train_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Training Offload Accuracy: {train_acc:.2f}%")
        
        # Early stopping if training accuracy reaches target_val_acc
        if train_acc >= target_val_acc:
            print(f"Early stopping: Training accuracy reached {train_acc:.2f}% (target {target_val_acc}%).")
            break
def analyze_local_cloud_probs_with_decisions(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    offload_model,
    data_loader,
    entropy_threshold=0.2,
    device='cuda'
):
    """
    Greek Description:
      Αναλύουμε πώς το τοπικό και το cloud μοντέλο συμπεριφέρονται (ορθότητα προβλέψεων) 
      και ταυτόχρονα βλέπουμε πόσα δείγματα θα πήγαιναν τοπικά ή στο cloud:
        1) Με βάση το entropy_threshold
        2) Με βάση το offload_model

      Στο τέλος, τυπώνει για καθεμιά από τις 4 κατηγορίες (local=σωστό,cloud=σωστό/λάθος/κλπ.):
        - Τον μέσο όρο πιθανότητας επιτυχίας του local
        - Τον μέσο όρο πιθανότητας επιτυχίας του cloud
        - Πόσα δείγματα (σε αυτήν την κατηγορία) πήγαν local / cloud 
          σύμφωνα με το offload μοντέλο
        - Πόσα δείγματα πήγαν local / cloud σύμφωνα με το entropy-based decision
    """

    # Switch to eval mode
    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    offload_model.eval()

    import numpy as np
    import torch.nn.functional as F

    # We'll accumulate data for the 4 categories:
    #   cat1 => local=correct & cloud=correct
    #   cat2 => local=correct & cloud=wrong
    #   cat3 => local=wrong   & cloud=correct
    #   cat4 => local=wrong   & cloud=wrong
    #
    # For each category, we store local_prob_correct and cloud_prob_correct
    cat1_local_probs = []
    cat1_cloud_probs = []
    cat2_local_probs = []
    cat2_cloud_probs = []
    cat3_local_probs = []
    cat3_cloud_probs = []
    cat4_local_probs = []
    cat4_cloud_probs = []

    # We'll also store how many samples in each category went local/cloud 
    # for offload decision and for entropy decision:
    cat1_offload_local = 0
    cat1_offload_cloud = 0
    cat2_offload_local = 0
    cat2_offload_cloud = 0
    cat3_offload_local = 0
    cat3_offload_cloud = 0
    cat4_offload_local = 0
    cat4_offload_cloud = 0

    cat1_entropy_local = 0
    cat1_entropy_cloud = 0
    cat2_entropy_local = 0
    cat2_entropy_cloud = 0
    cat3_entropy_local = 0
    cat3_entropy_cloud = 0
    cat4_entropy_local = 0
    cat4_entropy_cloud = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # === 1) Forward local feats => local_out => cloud_out
            local_feats = local_feature_extractor(images)
            local_out = local_classifier(local_feats)
            cloud_out = cloud_cnn(local_feats)

            # === 2) Probabilities
            local_softmax = F.softmax(local_out, dim=1)
            cloud_softmax = F.softmax(cloud_out, dim=1)

            batch_size = labels.size(0)
            local_probs_correct = local_softmax[range(batch_size), labels]
            cloud_probs_correct = cloud_softmax[range(batch_size), labels]

            # === 3) Check correctness (bool)
            local_pred = local_out.argmax(dim=1)
            cloud_pred = cloud_out.argmax(dim=1)
            local_correct_mask = (local_pred == labels)
            cloud_correct_mask = (cloud_pred == labels)

            # === 4) Offload decision
            # Flatten local_feats => pass to offload_model => offload_probs
            local_feats_flat = local_feats.view(local_feats.size(0), -1)
            logits_offload = offload_model(local_feats_flat)
            probs_offload = torch.sigmoid(logits_offload).squeeze(1)
            # if >0.5 => cloud
            offload_cloud_mask = (probs_offload > 0.5)
            offload_local_mask = ~offload_cloud_mask

            # === 5) Entropy-based decision
            # normalized entropy
            from math import log
            eps = 1e-5
            nclass = local_out.shape[1]
            log_c = log(float(nclass))
            local_entropy = -torch.sum(local_softmax * torch.log(local_softmax + eps), dim=1) / log_c
            # if local_entropy < threshold => local
            entropy_local_mask = (local_entropy < entropy_threshold)
            entropy_cloud_mask = ~entropy_local_mask

            # === 6) Build category masks
            cat1_mask = local_correct_mask & cloud_correct_mask
            cat2_mask = local_correct_mask & (~cloud_correct_mask)
            cat3_mask = (~local_correct_mask) & cloud_correct_mask
            cat4_mask = (~local_correct_mask) & (~cloud_correct_mask)

            # === 7) For each category mask, gather local/cloud probabilities
            def gather_probs(mask, arr):
                return arr[mask].cpu().tolist()

            cat1_local_probs.extend(gather_probs(cat1_mask, local_probs_correct))
            cat1_cloud_probs.extend(gather_probs(cat1_mask, cloud_probs_correct))
            cat2_local_probs.extend(gather_probs(cat2_mask, local_probs_correct))
            cat2_cloud_probs.extend(gather_probs(cat2_mask, cloud_probs_correct))
            cat3_local_probs.extend(gather_probs(cat3_mask, local_probs_correct))
            cat3_cloud_probs.extend(gather_probs(cat3_mask, cloud_probs_correct))
            cat4_local_probs.extend(gather_probs(cat4_mask, local_probs_correct))
            cat4_cloud_probs.extend(gather_probs(cat4_mask, cloud_probs_correct))

            # === 8) Count how many in each category => local/cloud by offload
            # We combine "catX_mask" with "offload_local_mask" => intersection
            cat1_idx = cat1_mask.nonzero(as_tuple=True)[0]
            cat2_idx = cat2_mask.nonzero(as_tuple=True)[0]
            cat3_idx = cat3_mask.nonzero(as_tuple=True)[0]
            cat4_idx = cat4_mask.nonzero(as_tuple=True)[0]

            # Offload local => intersection
            cat1_offload_local += (offload_local_mask[cat1_idx]).sum().item()
            cat1_offload_cloud += (offload_cloud_mask[cat1_idx]).sum().item()
            cat2_offload_local += (offload_local_mask[cat2_idx]).sum().item()
            cat2_offload_cloud += (offload_cloud_mask[cat2_idx]).sum().item()
            cat3_offload_local += (offload_local_mask[cat3_idx]).sum().item()
            cat3_offload_cloud += (offload_cloud_mask[cat3_idx]).sum().item()
            cat4_offload_local += (offload_local_mask[cat4_idx]).sum().item()
            cat4_offload_cloud += (offload_cloud_mask[cat4_idx]).sum().item()

            # === 9) Count how many in each category => local/cloud by entropy
            cat1_entropy_local += (entropy_local_mask[cat1_idx]).sum().item()
            cat1_entropy_cloud += (entropy_cloud_mask[cat1_idx]).sum().item()
            cat2_entropy_local += (entropy_local_mask[cat2_idx]).sum().item()
            cat2_entropy_cloud += (entropy_cloud_mask[cat2_idx]).sum().item()
            cat3_entropy_local += (entropy_local_mask[cat3_idx]).sum().item()
            cat3_entropy_cloud += (entropy_cloud_mask[cat3_idx]).sum().item()
            cat4_entropy_local += (entropy_local_mask[cat4_idx]).sum().item()
            cat4_entropy_cloud += (entropy_cloud_mask[cat4_idx]).sum().item()

    # === 10) Helper for printing means
    def mean_or_zero(arr):
        return np.mean(arr) if len(arr) > 0 else 0.0

    # compute means for local/cloud in each category
    c1_local_mean = mean_or_zero(cat1_local_probs)
    c1_cloud_mean = mean_or_zero(cat1_cloud_probs)
    c2_local_mean = mean_or_zero(cat2_local_probs)
    c2_cloud_mean = mean_or_zero(cat2_cloud_probs)
    c3_local_mean = mean_or_zero(cat3_local_probs)
    c3_cloud_mean = mean_or_zero(cat3_cloud_probs)
    c4_local_mean = mean_or_zero(cat4_local_probs)
    c4_cloud_mean = mean_or_zero(cat4_cloud_probs)

    cat1_count = len(cat1_local_probs)
    cat2_count = len(cat2_local_probs)
    cat3_count = len(cat3_local_probs)
    cat4_count = len(cat4_local_probs)

    print("=== Local/Cloud Probability + Decision Analysis ===")

    # function to print info for each category
    def print_category_info(cat_id, cat_count, local_mean, cloud_mean,
                            offload_local, offload_cloud, 
                            entropy_local, entropy_cloud):
        print(f"\nCategory {cat_id}, samples={cat_count}")
        print(f"  Avg local_prob: {local_mean:.3f}, Avg cloud_prob: {cloud_mean:.3f}")
        print(f"  Offload decision => local: {int(offload_local)}, cloud: {int(offload_cloud)}")
        print(f"  Entropy decision => local: {int(entropy_local)}, cloud: {int(entropy_cloud)}")

    print_category_info(
        "1 (L=correct, C=correct)", cat1_count, c1_local_mean, c1_cloud_mean,
        cat1_offload_local, cat1_offload_cloud,
        cat1_entropy_local, cat1_entropy_cloud
    )

    print_category_info(
        "2 (L=correct, C=wrong)", cat2_count, c2_local_mean, c2_cloud_mean,
        cat2_offload_local, cat2_offload_cloud,
        cat2_entropy_local, cat2_entropy_cloud
    )

    print_category_info(
        "3 (L=wrong, C=correct)", cat3_count, c3_local_mean, c3_cloud_mean,
        cat3_offload_local, cat3_offload_cloud,
        cat3_entropy_local, cat3_entropy_cloud
    )

    print_category_info(
        "4 (L=wrong, C=wrong)", cat4_count, c4_local_mean, c4_cloud_mean,
        cat4_offload_local, cat4_offload_cloud,
        cat4_entropy_local, cat4_entropy_cloud
    )


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


def evaluate_offload_mechanism_decision_accuracy(epochs_DDNN=15, epochs_offload=10, batch_size=32, L0=0.25, local_weight=0.7):
    """
    This function performs the following steps:
    1. Trains the Distributed Deep Neural Network (DDNN) using the training data.
    2. Computes the bk values for the entire training set and calculates the threshold b_star
       (the L0 percentile of bk values).
    3. Creates the offload dataset using the computed bk values and trains the offload mechanism.
    4. Evaluates the offload mechanism's decision accuracy on the training set and the test set.
    
    The ground truth offload decision for each sample is determined based on its bk value:
      - If bk < b_star, the sample should be processed locally (label 0).
      - If bk >= b_star, the sample should be offloaded to the cloud (label 1).
    
    The function prints the percentage of samples for which the offload mechanism's predicted
    decision matches the ground truth, separately for the training set and the test set.
    """
    # Step 1: Load data
    train_loader, val_loader, test_loader = load_data(batch_size)
    
    # Step 2: Initialize models (DDNN and offload mechanism)
    local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism = initialize_models()
    
    # Step 3: Initialize optimizers for DDNN and offload mechanism
    cnn_optimizer, offload_optimizer = initialize_optimizers(local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism)
    
    # Step 4: Train the DDNN network using training data
    train_DDNN(train_loader, local_feature_extractor, local_classifier, cloud_cnn, cnn_optimizer, local_weight, epochs_DDNN)
    
    # Step 5: Compute bk values and extract local features from the entire training set
    all_local_features, all_bks = compute_bks_for_dataset_unified(
        local_feature_extractor, local_classifier, cloud_cnn, train_loader,
        method=5, beta=0, device=device, return_features=True,
        entropy_threshold=0.2, tie_threshold=0.1, improvement_scale=0.12
    )
    
    # Step 6: Calculate b_star as the L0 percentile of all bk values
    b_star = calculate_b_star(all_bks, L0)
    
    # Step 7: Create the offload dataset using the computed bk values and local features
    combined_data = create_3d_dataset(all_bks, all_local_features)
    offload_dataset = OffloadDataset(combined_data, b_star)
    offload_loader = DataLoader(offload_dataset, batch_size=batch_size, shuffle=False)
    
    # Step 8: Train the offload mechanism using the offload dataset
    train_offload_mechanism(offload_mechanism, offload_optimizer, offload_loader, epochs=epochs_offload, device=device)
    
    # Step 9: Evaluate offload decision accuracy on the training set
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Compute local features from images
        local_feats = local_feature_extractor(images)
        
        # Compute bk values for the current batch using the DDNN
        batch_bks = calculate_bks(images, labels, local_classifier, local_feature_extractor, cloud_cnn)
        # Ground truth: 0 if bk < b_star (process locally), 1 if bk >= b_star (offload to cloud)
        gt_labels = (batch_bks >= b_star).astype(np.float32)
        
        # Get offload mechanism predictions: flatten local features and pass through the offload model
        local_feats_flat = local_feats.view(local_feats.size(0), -1)
        logits = offload_mechanism(local_feats_flat)
        pred_probs = torch.sigmoid(logits).squeeze(1)
        pred_labels = (pred_probs > 0.5).float()
        
        # Convert ground truth labels to a tensor
        gt_tensor = torch.tensor(gt_labels, device=device)
        
        # Count correct predictions
        correct_train += (pred_labels == gt_tensor).sum().item()
        total_train += labels.size(0)
    train_accuracy = correct_train / total_train * 100
    print(f"Offload decision accuracy on training set: {train_accuracy:.2f}%")
    
    # Step 10: Evaluate offload decision accuracy on the test set
    correct_test = 0
    total_test = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        local_feats = local_feature_extractor(images)
        
        # Compute bk values for the test batch
        batch_bks = calculate_bks(images, labels, local_classifier, local_feature_extractor, cloud_cnn)
        gt_labels = (batch_bks >= b_star).astype(np.float32)
        
        local_feats_flat = local_feats.view(local_feats.size(0), -1)
        logits = offload_mechanism(local_feats_flat)
        pred_probs = torch.sigmoid(logits).squeeze(1)
        pred_labels = (pred_probs > 0.5).float()
        
        gt_tensor = torch.tensor(gt_labels, device=device)
        correct_test += (pred_labels == gt_tensor).sum().item()
        total_test += labels.size(0)
    test_accuracy = correct_test / total_test * 100
    print(f"Offload decision accuracy on test set: {test_accuracy:.2f}%")

    
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
def evaluate_hybrid_inference(
    local_feature_extractor,
    local_classifier,
    cloud_cnn,
    offload_model,
    data_loader,
    entropy_threshold=0.2,
    offload_threshold=0.5,
    device='cuda'
):
    """
    Evaluate the DDNN using both the oflload mechanism and the entropy uncertainty to decide where to classify
    1) First check if the entropy of the local_output is lower than the given threshold and send to local
    2) Otherwise check the offload model decision to send to the cloud.
    """

    local_feature_extractor.eval()
    local_classifier.eval()
    cloud_cnn.eval()
    offload_model.eval()

    total_samples = 0
    local_correct = 0
    cloud_correct = 0
    local_count = 0
    cloud_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # === 1) Extract local features
            local_feats = local_feature_extractor(images)  # (batch, 32,16,16)
            # Compute local/classifier logits
            local_out = local_classifier(local_feats)
            # Compute cloud logits also!
            cloud_out = cloud_cnn(local_feats)

            # === 2) Compute local entropy
            entropy_vals = calculate_normalized_entropy(local_out)
            confident_mask = (entropy_vals < entropy_threshold)

            # A) Handle confident => local
            if confident_mask.any():
                idx_confident = confident_mask.nonzero(as_tuple=True)[0]
                local_preds = local_out[idx_confident].argmax(dim=1)
                correct_labels = labels[idx_confident]
                local_correct += (local_preds == correct_labels).sum().item()
                local_count += idx_confident.size(0)

            # B) Handle not confident => offload
            not_conf_mask = ~confident_mask
            if not_conf_mask.any():
                idx_not_conf = not_conf_mask.nonzero(as_tuple=True)[0]

                # flatten features for offload
                local_feats_flat = local_feats[idx_not_conf].view(-1, 8192)
                label_not_conf = labels[idx_not_conf]

                # pass to offload model
                offload_logits = offload_model(local_feats_flat)
                offload_probs = torch.sigmoid(offload_logits).squeeze(1)
                decision_cloud = (offload_probs > offload_threshold)
                decision_local = ~decision_cloud

                # --- local subset
                idx_local_off = decision_local.nonzero(as_tuple=True)[0]
                if idx_local_off.size(0) > 0:
                    local_idx_in_original = idx_not_conf[idx_local_off]
                    # we can use local_out for them
                    local_outputs_sub = local_out[local_idx_in_original]
                    local_preds_sub = local_outputs_sub.argmax(dim=1)
                    local_label_sub = labels[local_idx_in_original]
                    local_correct += (local_preds_sub == local_label_sub).sum().item()
                    local_count += local_idx_in_original.size(0)

                # --- cloud subset
                idx_cloud = decision_cloud.nonzero(as_tuple=True)[0]
                if idx_cloud.size(0) > 0:
                    cloud_idx_in_original = idx_not_conf[idx_cloud]
                    # now we can index cloud_out
                    cloud_outputs_sub = cloud_out[cloud_idx_in_original]
                    cloud_preds_sub = cloud_outputs_sub.argmax(dim=1)
                    cloud_label_sub = labels[cloud_idx_in_original]
                    cloud_correct += (cloud_preds_sub == cloud_label_sub).sum().item()
                    cloud_count += cloud_idx_in_original.size(0)

            total_samples += labels.size(0)

    overall_correct = local_correct + cloud_correct
    overall_acc = overall_correct / total_samples * 100
    local_acc = (local_correct / local_count) * 100 if local_count > 0 else 0
    cloud_acc = (cloud_correct / cloud_count) * 100 if cloud_count > 0 else 0

    print("\n=== Υβριδικό Inference Αποτελέσματα ===")
    print(f"Total samples: {total_samples}")
    print(f"Local samples: {local_count} ({100.0*local_count/total_samples:.1f}%)")
    print(f"Cloud samples: {cloud_count} ({100.0*cloud_count/total_samples:.1f}%)")
    print(f"Local Accuracy: {local_acc:.2f}%")
    print(f"Cloud Accuracy: {cloud_acc:.2f}%")
    print(f"Overall Accuracy: {overall_acc:.2f}%")

    return local_acc, cloud_acc, overall_acc, local_count, cloud_count

class OffloadDatasetCNN(Dataset):
    def __init__(self, combined_data, b_star):
        """
        combined_data: list of tuples (index, bk_value, feature)
        where feature is kept in its original shape (e.g., (32, 16, 16)).
        b_star: threshold computed from the bk values.
        """
        self.combined_data = combined_data
        self.b_star = b_star

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        global_idx, bk_value, feature = self.combined_data[idx]
        # Ground truth: label = 0 if bk < b_star (process locally), else 1 (offload to cloud)
        label = 1.0 if bk_value >= self.b_star else 0.0
        # Do NOT flatten the feature; keep its 3D shape.
        x_tensor = torch.tensor(feature, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.float32)
        return x_tensor, y_tensor

def create_3d_data_with_figures_cnn(all_bks, all_local_features):
    """
    Creates a combined data list of tuples (index, bk_value, feature)
    where each feature is kept in its original shape (e.g., (32,16,16)).
    """
    combined_data = []
    for i in range(len(all_bks)):
        bk_val = all_bks[i]
        feat = all_local_features[i]  # feature in shape (32,16,16)
        combined_data.append((i, bk_val, feat))
    return combined_data


# def evaluate_offload_mechanism_decision_accuracy(epochs_DDNN=15, epochs_offload=10, batch_size=32, L0=0.25, local_weight=0.7):
#     """
#     This function performs the following steps:
#     1. Trains the Distributed Deep Neural Network (DDNN) using the training data.
#     2. Computes the bk values for the entire training set and calculates the threshold b_star
#        (the L0 percentile of bk values).
#     3. Creates the offload dataset using the computed bk values and trains the offload mechanism.
#     4. Evaluates the offload mechanism's decision accuracy on the training set and the test set.
    
#     The ground truth offload decision for each sample is determined based on its bk value:
#       - If bk < b_star, the sample should be processed locally (label 0).
#       - If bk >= b_star, the sample should be offloaded to the cloud (label 1).
    
#     The function prints the percentage of samples for which the offload mechanism's predicted
#     decision matches the ground truth, separately for the training set and the test set.
#     """
#     # Step 1: Load data
#     train_loader, val_loader, test_loader = load_data(batch_size)
    
#     # Step 2: Initialize models (DDNN and offload mechanism)
#     local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism = initialize_models()
    
#     # Step 3: Initialize optimizers for DDNN and offload mechanism
#     cnn_optimizer, offload_optimizer = initialize_optimizers(local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism)
    
#     # Step 4: Train the DDNN network using training data
#     train_DDNN(train_loader, local_feature_extractor, local_classifier, cloud_cnn, cnn_optimizer, local_weight, epochs_DDNN)
    
#     # Step 5: Compute bk values and extract local features from the entire training set
#     all_local_features, all_bks = compute_bks_for_dataset_unified(
#         local_feature_extractor, local_classifier, cloud_cnn, train_loader,
#         method=5, beta=0, device=device, return_features=True,
#         entropy_threshold=0.2, tie_threshold=0.1, improvement_scale=0.12
#     )
    
#     # Step 6: Calculate b_star as the L0 percentile of all bk values
#     b_star = calculate_b_star(all_bks, L0)
    
#     # Step 7: Create the offload dataset using the computed bk values and local features
#     combined_data = create_3d_dataset(all_bks, all_local_features)
#     offload_dataset = OffloadDataset(combined_data, b_star)
#     offload_loader = DataLoader(offload_dataset, batch_size=batch_size, shuffle=False)
    
#     # Step 8: Train the offload mechanism using the offload dataset
#     train_offload_mechanism_local(offload_mechanism, offload_optimizer, offload_loader, epochs=epochs_offload, lr=0.001, device=device)
    
#     # Step 9: Evaluate offload decision accuracy on the training set
#     correct_train = 0
#     total_train = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         # Compute local features from images
#         local_feats = local_feature_extractor(images)
        
#         # Compute bk values for the current batch using the DDNN
#         batch_bks = calculate_bks(images, labels, local_classifier, local_feature_extractor, cloud_cnn)
#         # Ground truth: 0 if bk < b_star (process locally), 1 if bk >= b_star (offload to cloud)
#         gt_labels = (batch_bks >= b_star).astype(np.float32)
        
#         # Get offload mechanism predictions: flatten local features and pass through the offload model
#         local_feats_flat = local_feats.view(local_feats.size(0), -1)
#         logits = offload_mechanism(local_feats_flat)
#         pred_probs = torch.sigmoid(logits).squeeze(1)
#         pred_labels = (pred_probs > 0.5).float()
        
#         # Convert ground truth labels to a tensor
#         gt_tensor = torch.tensor(gt_labels, device=device)
        
#         # Count correct predictions
#         correct_train += (pred_labels == gt_tensor).sum().item()
#         total_train += labels.size(0)
#     train_accuracy = correct_train / total_train * 100
#     print(f"Offload decision accuracy on training set: {train_accuracy:.2f}%")
    
#     # Step 10: Evaluate offload decision accuracy on the test set
#     correct_test = 0
#     total_test = 0
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         local_feats = local_feature_extractor(images)
        
#         # Compute bk values for the test batch
#         batch_bks = calculate_bks(images, labels, local_classifier, local_feature_extractor, cloud_cnn)
#         gt_labels = (batch_bks >= b_star).astype(np.float32)
        
#         local_feats_flat = local_feats.view(local_feats.size(0), -1)
#         logits = offload_mechanism(local_feats_flat)
#         pred_probs = torch.sigmoid(logits).squeeze(1)
#         pred_labels = (pred_probs > 0.5).float()
        
#         gt_tensor = torch.tensor(gt_labels, device=device)
#         correct_test += (pred_labels == gt_tensor).sum().item()
#         total_test += labels.size(0)
#     test_accuracy = correct_test / total_test * 100
#     print(f"Offload decision accuracy on test set: {test_accuracy:.2f}%")

    
def main(epochs_DDNN=  15,epochs_optimization=10, batch_size=32 , L0=0.25 , local_weight=0.7):
    """ Initialze models,  train the 2 Networks(Local and Remote) , create  local features and bks from  the DDNNN 
        in order to train the Optimization Rule Network and test the DDNN with the optimized rule

    Args:
        epochs (int, optional): _description_. Defaults to 1.
        threshold (_type_, optional): _description_. Defaults to 1e-15.
        batch_size (int, optional): _description_. Defaults to 32.
        B (int, optional): _description_. Defaults to 10.
        L0 (float, optional): _description_. Defaults to 0.3.

    Returns:
        _type_: _description_
    """
    
    '''
    
    
    # initialize models  
    train_loader, val_loader, test_loader = load_data(batch_size)
    local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism = initialize_models()
    cnn_optimizer, offload_optimizer = initialize_optimizers(local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism)


    # Initialize scheduler after optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    

    # train the DDNN network
    train_DDNN(train_loader,local_feature_extractor,local_classifier,cloud_cnn,cnn_optimizer,local_weight,epochs_DDNN)

    # compute bk value and local feature map for each sample of the testing daaset
    all_local_features, all_bks = compute_bks_for_dataset_unified(local_feature_extractor,local_classifier,cloud_cnn,train_loader,method=5,beta=0,device=device,return_features=True,entropy_threshold=0.2,tie_threshold=0.1,improvement_scale=0.12)
    print("Finished computing Bk on the entire training set.")
    print("Shapes:", all_bks.shape)
    # Print the length of all_bks to check its size
    print(f"Length of all_bks: {len(all_bks)}")
    print(f"Length of local_features: {len(all_local_features)}")
    print(f"Expected length (number of training samples): {len(train_loader.dataset)}")
    
    # calculate b* value given the L0
    b_star = calculate_b_star(all_bks, L0)
    
    


    # combine local features and bks together in a 3D array with indexes
    combined_data = create_3d_dataset(all_bks, all_local_features)

    # create the offload the dataseet
    offload_dataset = OffloadDataset(combined_data, b_star=b_star)
    offload_loader = DataLoader(offload_dataset, batch_size=32, shuffle=False)

    # train the offload mechanism with local features as input
    # train_offload_mechanism_local(
    #     offload_mechanism, offload_optimizer, offload_loader ,epochs_optimization)
    train_offload_mechanism(offload_mechanism, offload_optimizer, offload_loader ,epochs_optimization)
  
    
    
    # compare the results o the entropy mechanism with the offload results
    # compare_entropy_with_offload_results(offload_mechanism,local_feature_extractor,local_classifier,test_loader,threshold_entropy=0.1,device=device)


    # plot_bk_distribution(all_bks, b_star)
    
    #  test the DDNN using the offload mechanism
    local_accuracy, cloud_accuracy, overall_accuracy, local_classified, cloud_classified = evaluate_with_offload_mechanism(offload_mechanism,local_feature_extractor, local_classifier, cloud_cnn,  test_loader, 0.5)



    
    # Εκτύπωση στατιστικων μεσων όρων κόστος
    analyze_local_cloud_probs_with_decisions(local_feature_extractor, local_classifier, cloud_cnn, offload_mechanism,data_loader=test_loader, entropy_threshold=0.2,device=device)
    
    
    
    
    oracle_res = oracle_rule(
    local_feature_extractor, local_classifier, cloud_cnn, test_loader, device=device
    )

    for p, acc in oracle_res.items():
        print(f"{p}% local => Oracle accuracy: {acc:.2f}%")
    '''
    
    #evaluate offload mechanism accuracy
    evaluate_offload_mechanism_decision_accuracy(15,10)

if __name__ == "__main__":
    main()
    
    
    