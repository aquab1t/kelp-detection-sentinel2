#!/usr/bin/env python3
"""Test binary kelp detection models on a sample scene."""

import sys
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kelp_detection import KelpPredictor


def test_1dcnn_binary():
    """Test 1D-CNN binary model."""
    print("=" * 60)
    print("Testing 1D-CNN Binary Model")
    print("=" * 60)
    
    model_path = Path(__file__).parent.parent / 'models' / '1dcnn_binary_int8.tflite'
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return False
    
    predictor = KelpPredictor(str(model_path))
    print(f"Model loaded: {predictor.input_shape} -> {predictor.output_shape}")
    print(f"Is binary: {predictor.is_binary}")
    
    # Test with random data
    test_data = np.random.randn(10, 9).astype(np.float32) * 0.5  # Scaled data
    predictions = predictor.predict(test_data, batch_size=10, show_progress=False)
    
    print(f"Test predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Sample predictions: {predictions[:5].flatten()}")
    
    # Test classify
    classified = predictor.predict_and_classify(test_data, batch_size=10)
    print(f"Classification unique values: {np.unique(classified)}")
    print(f"Kelp pixels: {np.sum(classified == 1)} / {len(classified)}")
    
    print("\n✓ 1D-CNN Binary test passed!\n")
    return True


def test_2dcnn_binary():
    """Test 2D-CNN binary model (PyTorch)."""
    print("=" * 60)
    print("Testing 2D-CNN Binary Model (PyTorch)")
    print("=" * 60)
    
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / '00_scripts'))
        from models import SimpleCNN
        
        model_path = Path(__file__).parent.parent.parent.parent / 'output' / 'models' / '2dcnn_binary_final.pt'
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return False
        
        model = SimpleCNN(in_channels=9, num_classes=1)
        model.load_state_dict(torch.load(str(model_path), map_location='cpu', weights_only=True))
        model.eval()
        
        print(f"Model loaded: 2D-CNN Binary")
        
        # Test with random data
        test_data = torch.randn(5, 9, 11, 11)  # batch=5, 9 bands, 11x11 patch
        with torch.no_grad():
            output = model(test_data).squeeze()
            if output.dim() == 0:  # Single value
                probs = torch.sigmoid(output).unsqueeze(0)
            else:
                probs = torch.sigmoid(output)
        
        print(f"Test output shape: {output.shape}")
        print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"Sample probabilities: {probs[:5]}")
        
        print("\n✓ 2D-CNN Binary test passed!\n")
        return True
        
    except Exception as e:
        print(f"Error testing 2D-CNN: {e}")
        return False


if __name__ == '__main__':
    print("\nTesting Binary Kelp Detection Models\n")
    
    success = True
    success &= test_1dcnn_binary()
    # success &= test_2dcnn_binary()  # Skip for now - requires PyTorch model conversion
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
