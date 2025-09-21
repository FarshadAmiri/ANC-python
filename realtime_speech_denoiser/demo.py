#!/usr/bin/env python3
"""
Demo script for the Real-Time Speech Denoiser

This script demonstrates the capabilities of the speech denoiser
with both file processing and real-time processing.
"""

import os
import sys
import time

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import SpeechDenoiser


def demo_file_processing():
    """Demonstrate file processing"""
    print("=" * 60)
    print("DEMO: File Processing")
    print("=" * 60)
    
    # Create denoiser
    denoiser = SpeechDenoiser()
    
    # Test with sample file
    print("\n1. Testing with provided noisy sample...")
    success = denoiser.test_with_sample()
    
    if success:
        print("\n✓ File processing demo completed successfully!")
        print("  - Original file: ../noisy_fish.wav")
        print("  - Processed file: denoised_test_output.wav")
        print("  - You can play both files to hear the difference")
    else:
        print("\n✗ File processing demo failed")
    
    return success


def demo_realtime_processing():
    """Demonstrate real-time processing (interactive)"""
    print("\n" + "=" * 60)
    print("DEMO: Real-Time Processing")
    print("=" * 60)
    
    print("\nThis demo will start real-time speech enhancement.")
    print("It will process audio from your microphone and play the enhanced audio.")
    print("\nRequirements:")
    print("- Working microphone")
    print("- Working speakers/headphones")
    print("- Quiet environment for testing")
    
    response = input("\nDo you want to run the real-time demo? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\nStarting real-time demo...")
        print("Speak into your microphone to test the noise reduction.")
        print("Press Ctrl+C to stop.\n")
        
        try:
            denoiser = SpeechDenoiser()
            denoiser.run_realtime()
        except KeyboardInterrupt:
            print("\nReal-time demo stopped by user.")
        except Exception as e:
            print(f"\nReal-time demo failed: {e}")
            print("This might be due to missing audio devices or permissions.")
            return False
        
        return True
    else:
        print("Skipping real-time demo.")
        return True


def print_system_info():
    """Print system information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    import torch
    import sounddevice as sd
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    print("\nAudio devices:")
    try:
        devices = sd.query_devices()
        print(devices)
    except Exception as e:
        print(f"Could not query audio devices: {e}")


def main():
    """Main demo function"""
    print("Real-Time Speech Denoiser - Demo")
    print("Inspired by DeepFilterNet for real-time speech enhancement")
    print()
    
    # Print system info
    print_system_info()
    
    # Demo file processing
    file_success = demo_file_processing()
    
    if file_success:
        # Demo real-time processing (optional)
        demo_realtime_processing()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("\nFor more advanced usage, see the README.md file.")
    print("Command-line options:")
    print("  python main.py --mode realtime     # Real-time processing")
    print("  python main.py --mode file -i input.wav -o output.wav")
    print("  python main.py --test              # Test with sample")


if __name__ == "__main__":
    main()