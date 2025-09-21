#!/usr/bin/env python3
"""
DeepFilterNet2 Real-Time Speech Denoiser

A Python-based real-time speech noise suppressor using DeepFilterNet2 architecture.
Supports both real-time microphone input and file processing.

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
from typing import Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepfilternet2 import DeepFilterNet2, create_model
from realtime_processor import RealTimeProcessor, FileProcessor, calculate_audio_metrics, test_model_performance


class DeepFilterNet2App:
    """Main application class for DeepFilterNet2 speech denoiser."""
    
    def __init__(self, device: str = 'auto', model_name: str = 'deepfilternet2_base'):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.realtime_processor = None
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\nReceived interrupt signal. Shutting down...")
        if self.realtime_processor:
            self.realtime_processor.stop()
        sys.exit(0)
    
    def load_model(self):
        """Load the DeepFilterNet2 model."""
        if self.model is None:
            print("Loading DeepFilterNet2 model...")
            try:
                self.model = create_model(
                    model_name=self.model_name,
                    device=self.device
                )
                # Update device to the resolved device
                self.device = self.model.device
                print(f"Model loaded successfully on device: {self.device}")
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)
        return self.model
    
    def run_realtime(self, sample_rate: int = 48000, chunk_size: int = 1024):
        """Run real-time speech enhancement."""
        model = self.load_model()
        
        print(f"\nStarting real-time speech enhancement...")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Chunk size: {chunk_size} samples")
        
        try:
            self.realtime_processor = RealTimeProcessor(
                model=model,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                device=self.device
            )
            
            self.realtime_processor.start()
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
                
        except Exception as e:
            print(f"Real-time processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.realtime_processor:
                self.realtime_processor.stop()
    
    def process_file(self, input_path: str, output_path: str = None):
        """Process an audio file."""
        model = self.load_model()
        
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return False
        
        if output_path is None:
            output_path = input_path.stem + "_enhanced" + input_path.suffix
        
        print(f"\nProcessing audio file...")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        
        try:
            processor = FileProcessor(model, device=self.device)
            
            def progress_callback(progress):
                percent = int(progress * 100)
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r  Progress: |{bar}| {percent}% Complete', end='', flush=True)
            
            # Process file
            start_time = time.time()
            enhanced_audio, sr = processor.process_file(
                str(input_path), 
                str(output_path),
                progress_callback=progress_callback
            )
            processing_time = time.time() - start_time
            
            print()  # New line after progress bar
            
            # Load original for comparison
            try:
                import librosa
                original_audio, _ = librosa.load(str(input_path), sr=sr, mono=True)
                
                # Calculate metrics
                metrics = calculate_audio_metrics(original_audio, enhanced_audio, sr)
                
                # Performance metrics
                audio_duration = len(original_audio) / sr
                realtime_factor = audio_duration / processing_time
                
                print(f"\nProcessing completed successfully!")
                print(f"  Audio duration: {audio_duration:.2f}s")
                print(f"  Processing time: {processing_time:.2f}s")
                print(f"  Real-time factor: {realtime_factor:.2f}x")
                
                if realtime_factor >= 1.0:
                    print("  ✅ Faster than real-time processing!")
                else:
                    print("  ⚠️  Processing slower than real-time")
                    
            except Exception as e:
                print(f"Could not calculate detailed metrics: {e}")
            
            return True
            
        except Exception as e:
            print(f"File processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_with_sample(self, test_file: str = None):
        """Test with the provided noisy sample or default test."""
        model = self.load_model()
        
        print(f"\nRunning performance test...")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        
        # Check for noisy_fish.wav in parent directory
        if test_file is None:
            parent_dir = Path(__file__).parent.parent
            noisy_fish = parent_dir / "noisy_fish.wav"
            if noisy_fish.exists():
                test_file = str(noisy_fish)
                print(f"  Found test file: {test_file}")
            else:
                print(f"  Test file not found at {noisy_fish}, using synthetic test")
        
        try:
            test_model_performance(model, test_file)
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'torch', 'torchaudio', 'numpy', 'sounddevice', 
        'soundfile', 'librosa', 'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Error: Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DeepFilterNet2 Real-Time Speech Denoiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode realtime                    # Real-time microphone processing
  %(prog)s --mode file --input audio.wav     # Process audio file
  %(prog)s --test                             # Test with provided sample
  %(prog)s --mode file --input noisy.wav --output clean.wav  # Custom output
  %(prog)s --mode realtime --device cuda     # Use GPU acceleration
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['realtime', 'file'], 
        help='Processing mode: realtime microphone or file processing'
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
        help='Device to run on (default: auto)'
    )
    parser.add_argument(
        '--model',
        default='deepfilternet2_base',
        help='Model name to use (default: deepfilternet2_base)'
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
        help='Run performance test with sample audio'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        help='Custom test file for performance testing'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='DeepFilterNet2 Real-Time Speech Denoiser v1.0'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Validate arguments
    if args.test:
        # Test mode
        pass
    elif args.mode == 'file':
        if not args.input:
            parser.error("File mode requires --input argument")
    elif args.mode == 'realtime':
        # Real-time mode - check audio system
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                print("Warning: No audio input devices found")
        except Exception as e:
            print(f"Warning: Could not check audio devices: {e}")
    elif args.mode is None and not args.test:
        parser.error("Must specify --mode or --test")
    
    # Create application
    app = DeepFilterNet2App(
        device=args.device,
        model_name=args.model
    )
    
    try:
        if args.test:
            success = app.test_with_sample(args.test_file)
        elif args.mode == 'realtime':
            app.run_realtime(
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size
            )
            success = True
        elif args.mode == 'file':
            success = app.process_file(args.input, args.output)
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()