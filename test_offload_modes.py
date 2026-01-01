"""
Quick test script to verify OffloadMechanism auto-initialization
across different input modes.
"""
import torch
from latest_deep_offload import OffloadMechanism

def test_offload_mechanism_modes():
    """
    Instantiate OffloadMechanism with 4 different modes and print their
    auto-initialized architecture details.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    modes_to_test = ['feat', 'img', 'logits', 'logits_plus']
    NUM_CLASSES = 10  # CIFAR-10
    
    print("="*80)
    print("TESTING OffloadMechanism AUTO-INITIALIZATION")
    print("="*80)
    
    for mode in modes_to_test:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*80}")
        
        # Create model with ONLY input_mode specified (all else uses defaults)
        # ⬅️ ΠΡΟΣΘΗΚΗ NUM_CLASSES
        model = OffloadMechanism(input_mode=mode, NUM_CLASSES=NUM_CLASSES).to(device)
        
        # Print architecture details
        print(f"\nAuto-initialized parameters:")
        print(f"  • Input mode:        {model.mode}")
        print(f"  • NUM_CLASSES:       {NUM_CLASSES}")
        
        # Get the default input_shape that was used (computed dynamically for logits modes)
        if mode == 'logits':
            expected_shape = (NUM_CLASSES,)
        elif mode == 'logits_plus':
            expected_shape = (NUM_CLASSES + 2,)
        else:
            expected_shape = OffloadMechanism._DEFAULTS[mode]['input_shape']
        
        print(f"  • Input shape (expected): {expected_shape}")
        
        # Check if conv mode or FC-only mode
        if model.mode in ('feat', 'img'):
            print(f"  • Conv dimensions:   {model.conv_dims}")
            print(f"  • Num layers/block:  {model.num_layers}")
            print(f"  • Flattened dim:     {model.flat_dim}")
        else:
            print(f"  • Input vector dim:  {model.flat_dim}")
        
        print(f"  • FC dimensions:     {model.fc_dims}")
        print(f"  • Dropout prob:      {model.dropout_p}")
        print(f"  • Latent skip-in:    {model.latent_in}")
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n  Total parameters:      {total_params:,}")
        print(f"  Trainable parameters:  {trainable_params:,}")
        
        # Test forward pass with dummy input
        print(f"\n  Testing forward pass...")
        try:
            if model.mode in ('feat', 'img'):
                # Use the expected input_shape
                dummy_input = torch.randn(8, *expected_shape).to(device)  # batch=8
            else:  # logits or logits_plus
                input_dim = expected_shape[0]
                dummy_input = torch.randn(8, input_dim).to(device)
            
            output = model(dummy_input)
            print(f"    ✓ Input shape:  {tuple(dummy_input.shape)}")
            print(f"    ✓ Output shape: {tuple(output.shape)}")
            print(f"    ✓ Forward pass successful!")
        except Exception as e:
            print(f"    ✗ Forward pass failed: {e}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_offload_mechanism_modes()