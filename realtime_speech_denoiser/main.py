#!/usr/bin/env python3
"""
Real-Time Speech Denoiser using Deep Learning
Inspired by DeepFilterNet for real-time speech enhancement

Usage:
    python main.py --mode realtime                    # Real-time microphone processing
    python main.py --mode file --input audio.wav     # Process audio file
    python main.py --test                             # Test with provided sample
"""

import argparse
import sys
import os
import signal
import time
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_processor import RealTimeProcessor, FileProcessor, process_audio_similarity
from deep_filter import create_model, load_pretrained_weights


class SpeechDenoiser:
    """Main application class for real-time speech denoising"""
    
    def __init__(self, device: str = 'auto', model_path: str = None):
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_model(device=self.device)
        
        # Load pretrained weights if available
        if model_path:
            load_pretrained_weights(self.model, model_path)
        
        # Create processors
        self.realtime_processor = None
        self.file_processor = FileProcessor(self.model, self.device)
        
        print("Speech denoiser initialized successfully!")
    
    def run_realtime(self, sample_rate: int = 48000, chunk_size: int = 1024):
        """Run real-time speech enhancement"""
        print("\n=== Real-Time Speech Enhancement ===")
        print("This will process microphone input in real-time.")
        print("Make sure you have a microphone connected.")
        print("The enhanced audio will be played through speakers.")
        print("Press Ctrl+C to stop.\n")
        
        # Create realtime processor
        self.realtime_processor = RealTimeProcessor(
            model=self.model,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            device=self.device
        )
        
        try:
            # Start processing
            self.realtime_processor.start()
            
            # Wait for user interrupt
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
        except Exception as e:
            print(f"Error in real-time processing: {e}")
        finally:
            if self.realtime_processor:
                self.realtime_processor.stop()
    
    def process_file(self, input_path: str, output_path: str = None):
        """Process an audio file"""
        print(f"\n=== File Processing ===")
        
        # Validate input
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            return False
        
        # Generate output path if not provided
        if output_path is None:
            input_pathlib = Path(input_path)
            output_path = str(input_pathlib.parent / f"{input_pathlib.stem}_denoised{input_pathlib.suffix}")
        
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        
        def progress_callback(progress):
            """Print progress"""
            percent = int(progress * 100)
            bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
            print(f"\rProcessing: [{bar}] {percent}%", end="", flush=True)
        
        try:
            # Process file
            enhanced_audio, sr = self.file_processor.process_file(
                input_path, output_path, progress_callback=progress_callback
            )
            print()  # New line after progress bar
            
            # Load original for comparison
            try:
                original_audio, _ = sf.read(input_path, dtype='float32')
                if original_audio.ndim > 1:
                    original_audio = original_audio[:, 0]
                
                # Calculate similarity metrics
                process_audio_similarity(original_audio, enhanced_audio, sr)
                
            except Exception as e:
                print(f"Could not calculate similarity metrics: {e}")
            
            print(f"\nFile processing completed successfully!")
            return True
            
        except Exception as e:
            print(f"\nError processing file: {e}")
            return False
    
    def test_with_sample(self):
        """Test with the provided noisy sample"""
        print("\n=== Testing with Provided Sample ===")
        
        # Look for the noisy sample
        sample_path = "../noisy_fish.wav"
        if not os.path.exists(sample_path):
            sample_path = "noisy_fish.wav"
        if not os.path.exists(sample_path):
            print("Error: Could not find noisy_fish.wav sample file")
            return False
        
        output_path = "denoised_test_output.wav"
        
        print(f"Testing with: {sample_path}")
        success = self.process_file(sample_path, output_path)
        
        if success:
            print(f"\nTest completed! Check the output file: {output_path}")
            print("You can compare the original and denoised audio files.")
        
        return success


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\nReceived interrupt signal. Shutting down...")
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-Time Speech Denoiser using Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode realtime                    # Real-time microphone processing
  %(prog)s --mode file --input audio.wav     # Process audio file
  %(prog)s --test                             # Test with provided sample
  %(prog)s --mode file --input noisy.wav --output clean.wav
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['realtime', 'file'], 
        help='Processing mode: realtime or file'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input audio file path (for file mode)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output audio file path (for file mode, optional)'
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for processing (default: auto)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to pretrained model weights (optional)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        help='Sample rate for real-time processing (default: 48000)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1024,
        help='Chunk size for real-time processing (default: 1024)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test with the provided noisy sample file'
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate arguments
    if not args.test and not args.mode:
        parser.error("Must specify either --mode or --test")
    
    if args.mode == 'file' and not args.input and not args.test:
        parser.error("File mode requires --input argument")
    
    try:
        # Create speech denoiser
        denoiser = SpeechDenoiser(device=args.device, model_path=args.model)
        
        # Run based on mode
        if args.test:
            denoiser.test_with_sample()
        elif args.mode == 'realtime':
            denoiser.run_realtime(
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size
            )
        elif args.mode == 'file':
            denoiser.process_file(args.input, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()