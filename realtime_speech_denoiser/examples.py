#!/usr/bin/env python3
"""
Quick Start Guide for Real-Time Speech Denoiser

This script provides easy examples of how to use the speech denoiser
for common tasks.
"""

import os
import sys
import time

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import SpeechDenoiser


def example_file_processing():
    """Example: Process a noisy audio file"""
    print("Example 1: File Processing")
    print("-" * 30)
    
    # Create speech denoiser
    denoiser = SpeechDenoiser(device='auto')  # auto-detect GPU/CPU
    
    # Process a file (replace with your own file path)
    input_file = "../noisy_fish.wav"
    output_file = "cleaned_speech.wav"
    
    if os.path.exists(input_file):
        print(f"Processing: {input_file}")
        success = denoiser.process_file(input_file, output_file)
        
        if success:
            print(f"✅ Success! Clean audio saved to: {output_file}")
        else:
            print("❌ Processing failed")
    else:
        print(f"Input file not found: {input_file}")
        print("Please provide a valid audio file path")


def example_realtime_processing():
    """Example: Real-time microphone processing"""
    print("\nExample 2: Real-Time Processing")
    print("-" * 30)
    
    print("This will start real-time speech enhancement from your microphone.")
    print("Requirements:")
    print("- Working microphone")
    print("- Audio output device (speakers/headphones)")
    print()
    
    response = input("Start real-time processing? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        try:
            # Create speech denoiser
            denoiser = SpeechDenoiser(device='auto')
            
            print("Starting real-time processing...")
            print("Speak into your microphone - you'll hear the enhanced audio")
            print("Press Ctrl+C to stop")
            print()
            
            # Start real-time processing
            denoiser.run_realtime(
                sample_rate=48000,    # Audio sample rate
                chunk_size=1024       # Processing chunk size (affects latency)
            )
            
        except KeyboardInterrupt:
            print("\n✅ Real-time processing stopped")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure your microphone and speakers are working")
    else:
        print("Skipped real-time processing")


def example_batch_processing():
    """Example: Process multiple files"""
    print("\nExample 3: Batch Processing")
    print("-" * 30)
    
    # List of files to process (replace with your own files)
    input_files = [
        "../noisy_fish.wav",
        # Add more files here
    ]
    
    # Create speech denoiser once
    denoiser = SpeechDenoiser(device='auto')
    
    print(f"Processing {len(input_files)} file(s)...")
    
    for i, input_file in enumerate(input_files, 1):
        if os.path.exists(input_file):
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"clean_{base_name}_{i}.wav"
            
            print(f"\n[{i}/{len(input_files)}] Processing: {input_file}")
            success = denoiser.process_file(input_file, output_file)
            
            if success:
                print(f"✅ Saved: {output_file}")
            else:
                print(f"❌ Failed: {input_file}")
        else:
            print(f"❌ Not found: {input_file}")
    
    print("\nBatch processing completed!")


def example_custom_settings():
    """Example: Custom processing settings"""
    print("\nExample 4: Custom Settings")
    print("-" * 30)
    
    # Create denoiser with specific settings
    denoiser = SpeechDenoiser(
        device='cpu',  # Force CPU usage
        model_path=None  # No pretrained weights
    )
    
    # Custom real-time settings for lower latency
    print("Custom real-time settings for lower latency:")
    
    if input("Start real-time with custom settings? (y/N): ").strip().lower() in ['y', 'yes']:
        try:
            denoiser.run_realtime(
                sample_rate=16000,   # Lower sample rate for faster processing
                chunk_size=512       # Smaller chunks for lower latency
            )
        except KeyboardInterrupt:
            print("\n✅ Custom real-time processing stopped")
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function with examples"""
    print("Real-Time Speech Denoiser - Quick Start Examples")
    print("=" * 50)
    
    examples = [
        ("File Processing", example_file_processing),
        ("Real-Time Processing", example_realtime_processing),
        ("Batch Processing", example_batch_processing),
        ("Custom Settings", example_custom_settings),
    ]
    
    while True:
        print("\nChoose an example:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"  {i}. {name}")
        print("  0. Exit")
        
        try:
            choice = int(input("\nEnter your choice (0-4): "))
            
            if choice == 0:
                print("Goodbye!")
                break
            elif 1 <= choice <= len(examples):
                print(f"\n{'='*50}")
                examples[choice-1][1]()  # Call the example function
                print(f"{'='*50}")
                
                input("\nPress Enter to continue...")
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()