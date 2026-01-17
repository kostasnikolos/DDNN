"""
Offload Decision Mechanism
===========================

Models and datasets for the offload decision mechanism that determines
whether a sample should be processed locally (edge) or offloaded to the cloud.

- OffloadMechanism: Neural network that learns to make offload decisions
- OffloadDatasetCNN: PyTorch Dataset for training the offload mechanism
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class OffloadMechanism(nn.Module):
    """
    Flexible offload decision network supporting multiple input modes.
    
    This network learns to predict whether a sample should be processed locally (0)
    or offloaded to the cloud (1) based on various input representations.
    
    Supported Input Modes
    ---------------------
    - 'feat': Feature map tensor from LocalFeatureExtractor, shape (B, 32, 16, 16)
    - 'shallow_feat': Same as 'feat' but with shallow FC-only architecture
    - 'img': Raw image tensor, shape (B, 3, 32, 32)
    - 'logits': Local model logits only, shape (B, NUM_CLASSES)
    - 'logits_plus': Logits + margin + entropy, shape (B, NUM_CLASSES+2)
    
    Architecture
    ------------
    The network architecture adapts based on input_mode:
    
    **For 'feat' and 'img' modes (convolutional):**
    - Conv blocks with BatchNorm, LeakyReLU, optional Dropout
    - Optional skip connections (latent_in parameter)
    - Automatic channel adjustment for skip connections
    - FC tail for final decision
    
    **For 'logits' and 'logits_plus' modes (FC-only):**
    - Multi-layer perceptron with BatchNorm, ReLU, Dropout
    - Optional skip connections from input to intermediate layers
    
    **For 'shallow_feat' mode (FC-only, minimal):**
    - Lightweight MLP (2 hidden layers)
    - No skip connections
    - Lower dropout
    
    Default Architectures (Auto-configured)
    ---------------------------------------
    - 'feat': Conv[64,128,256] → FC[256,128,1], dropout=0.35, skip at block 1
    - 'shallow_feat': FC[64,32,1], dropout=0.1, no skips
    - 'img': Conv[64,128,256] → FC[256,128,1], dropout=0.35, skip at block 1
    - 'logits': FC[256,128,64,32,1], dropout=0.1, skip at FC layer 1
    - 'logits_plus': FC[256,128,64,32,1], dropout=0.1, skip at FC layer 1
    
    Parameters
    ----------
    input_shape : tuple, optional
        - For 'feat'/'img': (C, H, W), e.g., (32, 16, 16) or (3, 32, 32)
        - For 'logits': (D,), e.g., (10,) for CIFAR-10
        - For 'logits_plus': Auto-computed as (NUM_CLASSES + 2,)
        - If None, uses mode-specific default
    input_mode : str, default='feat'
        One of: 'feat', 'shallow_feat', 'img', 'logits', 'logits_plus'
    conv_dims : tuple/list of int, optional
        Output channels for each Conv block. Ignored for logits modes.
        Default for 'feat'/'img': (64, 128, 256)
    num_layers : int, optional
        Number of Conv-BN-LeakyReLU layers per block. Default=1
    fc_dims : tuple/list of int, optional
        Hidden sizes for FC layers. Last element MUST be 1 (binary output).
        Examples: [256, 128, 1] or [64, 32, 1]
    dropout_p : float, default=0.25
        Dropout probability. Mode-specific defaults apply if None.
    dropout_idx : tuple/list of int, optional
        Global Conv layer indices (0-based) where dropout is inserted.
        Default: () (no conv-level dropout, only FC dropout)
    latent_in : tuple/list of int, optional
        Block/layer indices where input is concatenated (skip connections).
        For Conv modes: block indices. For FC modes: layer indices.
        Channels/dimensions are auto-adjusted. Default: mode-specific.
    NUM_CLASSES : int, default=10
        Number of classification classes (for auto-sizing logits modes)
    
    Attributes
    ----------
    mode : str
        Active input mode
    conv_blocks : nn.ModuleList
        Convolutional blocks (empty for logits modes)
    fc_blocks : nn.ModuleList
        Fully connected blocks
    fc_out : nn.Linear
        Final output layer (→ 1 logit for binary decision)
    flat_dim : int
        Flattened dimension before FC layers
    
    Examples
    --------
    >>> # Auto-configured logits_plus mode for CIFAR-10
    >>> model = OffloadMechanism(input_mode='logits_plus', NUM_CLASSES=10)
    >>> 
    >>> # Custom architecture for feature mode
    >>> model = OffloadMechanism(
    ...     input_shape=(32, 16, 16),
    ...     input_mode='feat',
    ...     conv_dims=[128, 256],
    ...     fc_dims=[512, 256, 1],
    ...     dropout_p=0.3,
    ...     latent_in=(1,)  # Skip connection at block 1
    ... )
    >>> 
    >>> # Shallow FC-only mode
    >>> model = OffloadMechanism(input_mode='shallow_feat')
    
    Forward Pass
    ------------
    Input: Tensor matching input_mode specification
    Output: (batch,) tensor of logits (apply sigmoid for probabilities)
    
    >>> logits = model(input_tensor)  # shape: (B,)
    >>> probs = torch.sigmoid(logits)  # 0 = local, 1 = cloud
    >>> decisions = (probs > 0.5).float()
    
    Notes
    -----
    - Skip connections (latent_in) concatenate the original input at specified layers
    - Channel counts are automatically adjusted after concatenation
    - The network outputs raw logits; use BCEWithLogitsLoss for training
    - For inference, apply sigmoid then threshold (typically 0.5)
    """
    
    # ========================================================================
    # MODE-SPECIFIC DEFAULT ARCHITECTURES
    # ========================================================================
    _DEFAULTS = {
        'feat': {
            'input_shape': (32, 16, 16),
            'conv_dims': [64, 128, 256],
            'num_layers': 1,
            'fc_dims': [256, 128, 1],
            'dropout_p': 0.35,
            'latent_in': (1,)
        },
        'shallow_feat': {
            'input_shape': (32, 16, 16),
            'fc_dims': [64, 32, 1],
            'dropout_p': 0.1,
            'latent_in': ()
        },
        'img': {
            'input_shape': (3, 32, 32),
            'conv_dims': [64, 128, 256],
            'num_layers': 1,
            'fc_dims': [256, 128, 1],
            'dropout_p': 0.35,
            'latent_in': (1,)
        },
        'logits': {
            'fc_dims': [256, 128, 64, 32, 1],
            'dropout_p': 0.1,
            'latent_in': (1,)
        },
        'logits_plus': {
            'fc_dims': [256, 128, 64, 32, 1],
            'dropout_p': 0.1,
            'latent_in': (1,)
        },
        # ★ NEW: logits_plus + predicted bk from CloudLogitPredictor
        'logits_with_bk_pred': {
            'fc_dims': [256, 128, 64, 32, 1],
            'dropout_p': 0.15,
            'latent_in': (1,)
        },
        # ★ TESTING: logits_plus + REAL bk (using actual cloud logits) - for upper bound analysis
        'logits_with_real_bk': {
            'fc_dims': [256, 128, 64, 32, 1],
            'dropout_p': 0.15,
            'latent_in': (1,)
        },
        # ★ Hybrid mode - combines logits_plus with compressed features
        # Uses a mini CNN encoder to compress (32,16,16) features to compact vector
        'hybrid': {
            'fc_dims': [128, 64, 1],
            'dropout_p': 0.15,
            'latent_in': (),
            'feat_latent_dim': 32,
            # Hybrid encoder architecture (hardcoded for now)
            'encoder_conv_dims': [64, 128],  # (32→64→128)
            'encoder_pooling': [2, 2],  # MaxPool after each conv
            'encoder_fc_dim': 128  # GAP output before final projection
        },
        # ★ NEW: Regression mode with predicted cloud logits
        # Input: local_logits(10) + predicted_cloud_logits(10) = 20-dim
        # Output: predicted_bk (scalar)
        # Loss: MSE (target = actual bk)
        # Decision: predicted_bk ≥ b_star → offload
        'logits_predicted_regression': {
            'input_shape': (20,),  # local_logits(10) + cloud_logits(10)
            'fc_dims': [256, 128, 64, 32, 1],  # Deeper network for regression
            'dropout_p': 0.15,
            'latent_in': (1,)  # Skip connection at layer 1
        }
    }
    
    def __init__(self,
                 input_shape=None,
                 input_mode='feat',
                 conv_dims=(64, 128, 256),
                 num_layers=1,
                 fc_dims=None,  # Changed to None default
                 dropout_p=None,  # Changed to None default
                 num_classes=10,
                 dropout_idx=(),
                 latent_in=(),
                 feat_latent_dim=32):  # For hybrid mode
        super().__init__()
        
        self.num_classes = num_classes
        
        # Validate input mode
        valid_modes = list(self._DEFAULTS.keys())
        if input_mode not in valid_modes:
            raise ValueError(f"input_mode must be one of {valid_modes}")
        
        self.mode = input_mode
        defaults = self._DEFAULTS[input_mode]
        
        # Auto-configure input_shape
        if input_shape is None:
            if input_mode == 'logits':
                input_shape = (num_classes,)
            elif input_mode == 'logits_plus':
                input_shape = (num_classes + 2,)  # logits + margin + entropy
            elif input_mode in ('logits_with_bk_pred', 'logits_with_real_bk'):
                input_shape = (num_classes + 2 + num_classes,)  # logits_plus + bk (predicted or real)
            elif input_mode == 'logits_predicted_regression':
                input_shape = (num_classes * 2,)  # local_logits + cloud_logits = 20-dim
            elif input_mode == 'hybrid':
                input_shape = None  # Hybrid mode computes flat_dim differently
            else:
                input_shape = defaults['input_shape']
        
        # Apply mode-specific defaults
        if input_mode in ('feat', 'img'):
            self.conv_dims = conv_dims if conv_dims is not None else defaults['conv_dims']
            self.num_layers = num_layers if num_layers is not None else defaults['num_layers']
        elif input_mode == 'shallow_feat':
            self.conv_dims = None
            self.num_layers = None
        elif input_mode == 'hybrid':
            # Hybrid mode has its own mini-encoder
            self.conv_dims = None
            self.num_layers = None
            self.feat_latent_dim = feat_latent_dim if feat_latent_dim else defaults.get('feat_latent_dim', 32)
        else:  # logits modes
            self.conv_dims = None
            self.num_layers = None
        
        self.fc_dims = fc_dims if fc_dims is not None else defaults['fc_dims']
        self.dropout_p = dropout_p if dropout_p is not None else defaults['dropout_p']
        self.latent_in = set(latent_in if latent_in is not None else defaults['latent_in'])
        self.dropout_idx = set(dropout_idx if dropout_idx is not None else defaults.get('dropout_idx', ()))
        
        # ----------------------------------------------------------------
        # ★ NEW: Build feature encoder for HYBRID mode
        # ----------------------------------------------------------------
        self.feat_encoder = None
        if input_mode == 'hybrid':
            # Mini CNN encoder: (32, 16, 16) → (feat_latent_dim,)
            self.feat_encoder = nn.Sequential(
                # Block 1: (32, 16, 16) → (64, 8, 8)
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Block 2: (64, 8, 8) → (128, 4, 4)
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Global Average Pooling → (128,)
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                # FC to latent → (feat_latent_dim,)
                nn.Linear(128, self.feat_latent_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p)
            )
            # Input to FC: feat_latent_dim + logits_plus (num_classes + 2)
            self.flat_dim = self.feat_latent_dim + num_classes + 2
            print(f"  [Hybrid Mode] Feature encoder: (32,16,16) → ({self.feat_latent_dim},)")
            print(f"  [Hybrid Mode] Combined input dim: {self.flat_dim} (feat={self.feat_latent_dim} + logits_plus={num_classes + 2})")
        
        # ----------------------------------------------------------------
        # Build convolutional blocks (only for 'feat' and 'img' modes)
        # ----------------------------------------------------------------
        self.conv_blocks = nn.ModuleList()
        C0 = input_shape[0] if (self.mode not in ('logits', 'logits_plus', 'logits_with_bk_pred', 'logits_with_real_bk', 'logits_predicted_regression', 'hybrid')) else None
        in_ch = C0
        layer_counter = 0
        
        if input_mode not in ('logits', 'logits_plus', 'logits_with_bk_pred', 'logits_with_real_bk', 'logits_predicted_regression', 'shallow_feat', 'hybrid'):
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
                    in_ch = out_ch
                    layer_counter += 1
                
                self.conv_blocks.append(nn.Sequential(*layers))
                
                # MaxPool after every block except the last
                if blk_id < len(conv_dims) - 1:
                    self.conv_blocks.append(nn.MaxPool2d(2))
                    H, W = H // 2, W // 2
                
                # Automatic in_channel adjustment for skip connections
                if blk_id in self.latent_in:
                    in_ch += C0
            
            self.flat_dim = in_ch * H * W
        elif input_mode != 'hybrid':
            # FC-only modes (logits, logits_plus, shallow_feat) - NOT hybrid
            if input_mode == 'shallow_feat':
                self.flat_dim = input_shape[0] * input_shape[1] * input_shape[2]  # 32*16*16
            else:
                if len(input_shape) != 1:
                    raise ValueError("input_shape must be (D,) for 'logits'/'logits_plus'")
                self.flat_dim = input_shape[0]
        # Note: hybrid mode already set self.flat_dim in the feat_encoder section
        
        # ----------------------------------------------------------------
        # Build fully connected tail
        # ----------------------------------------------------------------
        self.fc_blocks = nn.ModuleList()
        last_in = self.flat_dim
        
        for i, hidden in enumerate(self.fc_dims[:-1]):
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(last_in, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p)
            ))
            
            # Skip connection handling for FC layers
            if i in self.latent_in and self.mode in ('logits', 'logits_plus', 'logits_with_bk_pred', 'logits_with_real_bk'):
                last_in = hidden + self.flat_dim
            else:
                last_in = hidden
        
        self.fc_out = nn.Linear(last_in, self.fc_dims[-1])
    
    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    def forward(self, x, feat=None):
        """
        Forward pass through offload mechanism.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor matching input_mode specification:
            - 'feat'/'shallow_feat': (B, 32, 16, 16)
            - 'img': (B, 3, 32, 32)
            - 'logits': (B, NUM_CLASSES)
            - 'logits_plus': (B, NUM_CLASSES+2)
            - 'hybrid': (B, NUM_CLASSES+2) - logits_plus component
        feat : torch.Tensor, optional
            For 'hybrid' mode only: feature tensor (B, 32, 16, 16)
        
        Returns
        -------
        torch.Tensor
            Offload decision logits of shape (B,)
            Apply sigmoid for probabilities: 0=local, 1=cloud
        """
        if self.mode == 'hybrid':
            # ★ HYBRID mode: combine compressed features with logits_plus
            if feat is None:
                raise ValueError("Hybrid mode requires 'feat' argument")
            
            # Encode features to latent vector
            feat_latent = self.feat_encoder(feat)  # (B, feat_latent_dim)
            
            # Concatenate with logits_plus
            x = torch.cat([feat_latent, x], dim=1)  # (B, feat_latent_dim + num_classes + 2)
            
            # FC path
            for block in self.fc_blocks:
                x = block(x)
        
        elif self.mode not in ('logits', 'logits_plus', 'shallow_feat'):
            # Convolutional path (feat/img modes)
            x0 = x  # Store original for skip connections
            blk_id = 0
            
            for module in self.conv_blocks:
                x = module(x)
                
                # Only count Sequential modules (Conv blocks) for blk_id
                if isinstance(module, nn.Sequential):
                    if blk_id in self.latent_in:
                        # Spatial alignment if needed
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
            # FC-only path (logits/logits_plus/shallow_feat modes)
            if self.mode == 'shallow_feat':
                x = torch.flatten(x, 1)  # (B, 32, 16, 16) → (B, 8192)
            else:
                x = x.view(x.size(0), -1)
            
            x0 = x  # Store original for skip connections
            
            for i, block in enumerate(self.fc_blocks):
                x = block(x)
                if i in self.latent_in and self.mode in ('logits', 'logits_plus', 'logits_with_bk_pred', 'logits_with_real_bk'):
                    x = torch.cat([x, x0], dim=1)
        

        return self.fc_out(x)


class OffloadDatasetCNN(Dataset):
    """
    PyTorch Dataset for training the offload decision mechanism.
    
    This dataset provides samples in the format required by OffloadMechanism,
    along with offload decision labels generated either by a simple bk-threshold
    rule or by an oracle function that considers actual classification outcomes.
    
    Data Format
    -----------
    Each sample in combined_data should be a tuple:
        (x_repr, bk_val, true_label, feat_tensor)
    
    where:
    - x_repr: Input representation (logits, features, or image) matching input_mode
    - bk_val: Scalar bk value (local_error - cloud_error)
    - true_label: Ground truth class label
    - feat_tensor: Local feature maps (always included for oracle mode)
    
    Label Generation Modes
    ----------------------
    **1. bk-threshold rule (use_oracle_labels=False, default):**
    - Label = 1 (cloud) if bk >= b_star, else 0 (local)
    - Simple threshold-based rule
    - Fast but noisy approximation
    
    **2. Oracle labels (use_oracle_labels=True):**
    - Requires local_clf and cloud_clf
    - For each sample:
      * If local correct AND cloud wrong → 0 (local)
      * If cloud correct AND local wrong → 1 (cloud)
      * If both correct or both wrong → fallback to bk-threshold
    - More accurate but requires forward passes through both models
    
    Parameters
    ----------
    combined_data : list of tuple
        Each tuple: (x_repr, bk_val, true_label, feat_tensor)
    b_star : float
        Threshold on bk value for the threshold-based rule
    input_mode : str, default='feat'
        How to represent input: 'feat', 'shallow_feat', 'img', 'logits', 'logits_plus'
    include_bk : bool, default=False
        If True, __getitem__ returns (x, y_offload, y_true, feat, bk_val)
        If False, returns (x, y_offload, y_true, feat)
    filter_mask : array-like of bool, optional
        Boolean mask to select a subset of combined_data
    use_oracle_labels : bool, default=False
        If True, generate labels using oracle function (requires local_clf, cloud_clf)
        If False, use simple bk-threshold rule
    local_clf : nn.Module, optional
        Local classifier (required if use_oracle_labels=True)
    cloud_clf : nn.Module, optional
        Cloud CNN (required if use_oracle_labels=True)
    device : str, default='cpu'
        Device for oracle label computation ('cpu' or 'cuda')
    use_soft_labels : bool, default=False
        If True, use soft labels based on distance from b_star instead of hard 0/1.
        Soft labels help the model learn that borderline samples are uncertain.
        Formula: soft_label = sigmoid((bk - b_star) / (bk_std / temperature))
    soft_label_temperature : float, default=3.0
        Controls softness of labels. Higher = softer (more labels near 0.5).
        - temperature=1-2: Almost hard labels
        - temperature=3-4: Balanced (recommended)
        - temperature=5-10: Very soft labels
    
    Returns (from __getitem__)
    --------------------------
    If include_bk=False (default):
        x_tensor : torch.Tensor
            Input matching input_mode specification
        y_offload : torch.Tensor
            Offload decision label (0=local, 1=cloud, or soft value in [0,1])
        y_true : torch.Tensor
            Ground truth class label
        feat_tensor : torch.Tensor
            Local feature maps (for potential downstream use)
    
    If include_bk=True:
        Same as above, plus:
        bk_val : torch.Tensor
            Scalar bk value
    
    Examples
    --------
    >>> # Simple threshold-based labels
    >>> dataset = OffloadDatasetCNN(combined_data, b_star, use_soft_labels=False)
    >>> 
    >>> # Soft labels for better borderline handling
    >>> dataset = OffloadDatasetCNN(combined_data, b_star, 
    ...                             use_soft_labels=True, 
    ...                             soft_label_temperature=3.0)
    >>>
    >>> # Hard threshold-based labels (original behavior)
    >>> dataset = OffloadDatasetCNN(
    ...     combined_data=data,
    ...     b_star=0.15,
    ...     input_mode='logits_plus',
    ...     use_oracle_labels=False
    ... )
    >>> 
    >>> # Oracle labels for cleaner training
    >>> dataset = OffloadDatasetCNN(
    ...     combined_data=data,
    ...     b_star=0.15,
    ...     input_mode='logits_plus',
    ...     use_oracle_labels=True,
    ...     local_clf=local_classifier,
    ...     cloud_clf=cloud_cnn,
    ...     device='cuda'
    ... )
    >>> 
    >>> # With filtering
    >>> clean_mask = (bk_values > -0.5) & (bk_values < 0.5)
    >>> dataset = OffloadDatasetCNN(
    ...     combined_data=data,
    ...     b_star=0.15,
    ...     filter_mask=clean_mask
    ... )
    
    Notes
    -----
    - Oracle mode precomputes all labels during __init__ (slower initialization)
    - Threshold mode computes labels on-the-fly (faster initialization)
    - For 'logits_plus' mode, x_repr should already include margin and entropy
    - feat_tensor is always returned (even in logits mode) for debugging/analysis
    """
    
    def __init__(
        self,
        combined_data,
        b_star,
        *,
        input_mode='feat',
        include_bk=False,
        filter_mask=None,
        use_oracle_labels=False,
        local_clf=None,
        cloud_clf=None,
        device='cpu',
        # ★ Soft Labels Parameters
        use_soft_labels=False,
        soft_label_temperature=3.0,
        # ★ NEW: Regression mode - return raw bk as target
        regression_target=False
    ):
        # Apply filter mask if provided
        if filter_mask is not None:
            self.combined_data = [combined_data[i] for i, f in enumerate(filter_mask) if f]
        else:
            self.combined_data = combined_data
        
        self.b_star = b_star
        self.mode = input_mode
        self.include_bk = include_bk
        self.use_oracle = use_oracle_labels
        self.local_clf = local_clf
        self.cloud_clf = cloud_clf
        self.device = device
        
        # ★ Soft labels configuration
        self.use_soft_labels = use_soft_labels
        self.soft_label_temperature = soft_label_temperature
        
        # ★ NEW: Regression mode - target is raw bk value instead of label
        self.regression_target = regression_target
        if self.regression_target:
            print(f"  [Regression Mode] Target = raw bk values (not labels)")
        
        # ★ NEW: Compute bk_std for soft labels normalization
        if self.use_soft_labels:
            all_bks = [sample[1] for sample in self.combined_data]  # bk is at index 1
            self.bk_std = float(np.std(all_bks))
            self.bk_mean = float(np.mean(all_bks))
            print(f"  [Soft Labels] Enabled with temperature={soft_label_temperature}")
            print(f"  [Soft Labels] bk_std={self.bk_std:.4f}, bk_mean={self.bk_mean:.4f}")
        else:
            self.bk_std = None
            self.bk_mean = None
        
        # Precompute oracle labels if requested
        if self.use_oracle:
            assert local_clf and cloud_clf, "Oracle mode requires both classifiers"
            local_clf.eval()
            cloud_clf.eval()
            
            oracle_lbls = []
            with torch.no_grad():
                for _, bk, y_true, feat in self.combined_data:
                    feat_t = torch.tensor(feat, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    
                    loc_ok = (local_clf(feat_t).argmax().item() == y_true)
                    cld_ok = (cloud_clf(feat_t).argmax().item() == y_true)
                    
                    if loc_ok and not cld_ok:
                        oracle_lbls.append(0.0)  # Local correct, cloud wrong
                    elif cld_ok and not loc_ok:
                        oracle_lbls.append(1.0)  # Cloud correct, local wrong
                    else:
                        # Both correct or both wrong: fallback to bk-threshold
                        oracle_lbls.append(1.0 if bk >= b_star else 0.0)
            
            self.oracle_labels = torch.tensor(oracle_lbls, dtype=torch.float32)
        else:
            self.oracle_labels = None
    
    def __len__(self):
        return len(self.combined_data)
    
    def _compute_soft_label(self, bk_val):
        """
        Compute soft label based on distance from b_star.
        
        Formula: soft_label = sigmoid((bk - b_star) / (bk_std / temperature))
        
        - bk >> b_star → label ≈ 1.0 (confident cloud)
        - bk << b_star → label ≈ 0.0 (confident local)
        - bk ≈ b_star  → label ≈ 0.5 (uncertain)
        """
        # Normalize by bk_std and scale by temperature
        margin = (bk_val - self.b_star) / (self.bk_std / self.soft_label_temperature)
        # Sigmoid function
        soft_label = 1.0 / (1.0 + math.exp(-margin))
        return soft_label
    
    def __getitem__(self, idx):
        x_repr, bk_val, true_lbl, feat_tensor = self.combined_data[idx]
        
        # -------------------- Target assignment --------------------
        if self.regression_target:
            # ★ Regression mode: target is raw bk value
            y_offload = torch.tensor(bk_val, dtype=torch.float32)
        elif self.oracle_labels is not None:
            label_offload = self.oracle_labels[idx]
            if isinstance(label_offload, torch.Tensor):
                y_offload = label_offload.clone().detach().float()
            else:
                y_offload = torch.tensor(label_offload, dtype=torch.float32)
        elif self.use_soft_labels:
            # Soft label based on distance from b_star
            label_offload = self._compute_soft_label(bk_val)
            y_offload = torch.tensor(label_offload, dtype=torch.float32)
        else:
            # Simple bk-threshold rule (hard label)
            label_offload = 1.0 if bk_val >= self.b_star else 0.0
            y_offload = torch.tensor(label_offload, dtype=torch.float32)
        
        y_cifar = torch.tensor(true_lbl, dtype=torch.long)
        
        # Input tensor based on mode
        if self.mode in ('feat', 'shallow_feat'):
            x_tensor = torch.as_tensor(feat_tensor, dtype=torch.float32)
        
        elif self.mode == 'img':
            if isinstance(x_repr, torch.Tensor):
                x_tensor = x_repr.clone().detach().float()
            else:
                x_tensor = torch.as_tensor(x_repr, dtype=torch.float32)
        
        elif self.mode in ('logits', 'logits_plus', 'hybrid', 'logits_with_bk_pred', 'logits_with_real_bk', 'logits_predicted_regression'):
            # For 'hybrid' mode, x_repr contains logits_plus (12 dims)
            # For 'logits_with_bk_pred'/'logits_with_real_bk' modes, x_repr contains logits_plus (12 dims), bk added during forward pass
            # For 'logits_predicted_regression' mode, x_repr contains local_logits (10 dims), cloud_logits added during forward pass
            # Features are in feat_tensor which is returned separately
            if isinstance(x_repr, torch.Tensor):
                x_tensor = x_repr.clone().detach().float()
            else:
                x_tensor = torch.tensor(x_repr, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unknown input_mode: {self.mode}")
        
        # -------------------- Return --------------------
        feat_out = torch.tensor(feat_tensor, dtype=torch.float32)
        
        if self.include_bk:
            bk_out = torch.tensor(bk_val, dtype=torch.float32)
            return x_tensor, y_offload, y_cifar, feat_out, bk_out
        else:
            return x_tensor, y_offload, y_cifar, feat_out
