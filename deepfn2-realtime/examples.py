#!/usr/bin/env python3
"""
Simple usage examples for DeepFilterNet2 Real-Time Speech Denoiser
"""

import sys
import os
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api import DeepFilterNet2API, enhance_audio_simple


def example_simple_file_enhancement():
    """Example 1: Simple file enhancement"""
    print("Example 1: Simple File Enhancement")
    print("=" * 40)
    
    # Check if we have the test file
    test_file = Path("../noisy_fish.wav")
    if not test_file.exists():
        print("Test file not found. Creating synthetic example...")
        return example_synthetic_audio()
    
    # Simple one-line enhancement
    enhanced_audio = enhance_audio_simple(
        str(test_file), 
        "simple_enhanced.wav"
    )
    
    print(f"Enhanced audio saved to: simple_enhanced.wav")
    print(f"Enhanced audio shape: {enhanced_audio.shape}")
    return True


def example_api_usage():
    """Example 2: Using the full API"""
    print("\nExample 2: Full API Usage")
    print("=" * 40)
    
    # Create API instance
    api = DeepFilterNet2API(device="cpu")
    
    # Get model info
    info = api.get_model_info()
    print(f"Model info: {info}")
    
    # Check if we have test file
    test_file = Path("../noisy_fish.wav")
    if test_file.exists():
        # Enhance file with metrics
        success, metrics = api.enhance_file(
            test_file,
            "api_enhanced.wav",
            return_metrics=True
        )
        
        if success:
            print("Enhancement successful!")
            print(f"Quality metrics: {metrics}")
        else:
            print(f"Enhancement failed: {metrics}")
    else:
        print("Test file not found, skipping file enhancement example")
    
    return True


def example_synthetic_audio():
    """Example 3: Synthetic audio processing"""
    print("\nExample 3: Synthetic Audio Processing")
    print("=" * 40)
    
    import numpy as np
    import soundfile as sf
    
    # Create synthetic noisy speech
    duration = 3.0  # 3 seconds
    sr = 48000
    t = np.linspace(0, duration, int(duration * sr))
    
    # Synthetic speech-like signal (harmonic series)
    speech = np.zeros_like(t)
    for harmonic in [1, 2, 3, 4]:
        freq = 200 * harmonic  # Fundamental + harmonics
        speech += (1.0 / harmonic) * np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.3)
    
    # Add modulation for speech-like characteristics
    modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    speech *= modulation
    
    # Add background noise
    noise = np.random.normal(0, 0.4, len(speech))
    
    # Create noisy signal
    noisy_audio = speech + noise
    
    # Normalize
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio)) * 0.8
    speech = speech / np.max(np.abs(speech)) * 0.8
    
    print(f"Created synthetic audio: {duration}s at {sr}Hz")
    
    # Save original files
    sf.write("synthetic_clean.wav", speech, sr)
    sf.write("synthetic_noisy.wav", noisy_audio, sr)
    
    # Enhance using API
    api = DeepFilterNet2API(device="cpu")
    enhanced_audio, _ = api.enhance_audio(noisy_audio, sample_rate=sr)
    
    # Save enhanced
    sf.write("synthetic_enhanced.wav", enhanced_audio, sr)
    
    print("Files created:")
    print("  synthetic_clean.wav - Original clean speech")
    print("  synthetic_noisy.wav - Noisy input")
    print("  synthetic_enhanced.wav - Enhanced output")
    
    # Calculate simple metrics
    clean_rms = np.sqrt(np.mean(speech ** 2))
    noisy_rms = np.sqrt(np.mean(noisy_audio ** 2))
    enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
    
    print(f"\nSimple metrics:")
    print(f"  Clean RMS: {clean_rms:.3f}")
    print(f"  Noisy RMS: {noisy_rms:.3f}")
    print(f"  Enhanced RMS: {enhanced_rms:.3f}")
    print(f"  Enhancement ratio: {enhanced_rms/noisy_rms:.3f}")
    
    return True


def example_performance_test():
    """Example 4: Performance testing"""
    print("\nExample 4: Performance Testing")
    print("=" * 40)
    
    import time
    import numpy as np
    from deepfilternet2 import create_model
    
    # Create model
    model = create_model()
    
    # Test different chunk sizes for real-time processing
    chunk_sizes = [512, 1024, 2048, 4096]
    sample_rate = 48000
    
    print("Testing processing speed for different chunk sizes:")
    
    for chunk_size in chunk_sizes:
        # Create test chunk
        test_chunk = np.random.randn(chunk_size).astype(np.float32)
        
        # Warm up
        for _ in range(5):
            _ = model.process_chunk(test_chunk)
        
        # Benchmark
        num_iterations = 20
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = model.process_chunk(test_chunk)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        chunk_duration = chunk_size / sample_rate * 1000  # ms
        realtime_factor = chunk_duration / avg_time
        
        print(f"  Chunk size {chunk_size:4d}: {avg_time:5.1f}ms (RTF: {realtime_factor:5.1f}x)")
    
    return True


def main():
    """Run all examples"""
    print("DeepFilterNet2 Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_simple_file_enhancement()
        example_api_usage()
        example_synthetic_audio()
        example_performance_test()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        
        # List generated files
        for file_pattern in ["*enhanced.wav", "*synthetic*.wav", "*simple*.wav"]:
            import glob
            files = glob.glob(file_pattern)
            for f in files:
                print(f"  {f}")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()