"""
Real-time audio processing for DeepFilterNet2.
Handles both real-time microphone input and file processing.
"""

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import threading
import queue
import time
from typing import Optional, Callable, Tuple
from collections import deque
import librosa
from pathlib import Path

# Optional import for real-time processing
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    print("Warning: sounddevice not available. Real-time processing disabled.")
    SOUNDDEVICE_AVAILABLE = False
    sd = None

from deepfilternet2 import DeepFilterNet2


class RealTimeProcessor:
    """Real-time audio processor for DeepFilterNet2."""
    
    def __init__(
        self,
        model: DeepFilterNet2,
        sample_rate: int = 48000,
        chunk_size: int = 1024,
        overlap: float = 0.5,
        device: str = 'cpu',
        max_latency_ms: int = 100
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = device
        self.max_latency_ms = max_latency_ms
        
        # Calculate processing parameters
        self.hop_size = int(chunk_size * (1 - overlap))
        self.buffer_size = chunk_size * 4  # Buffer for smooth processing
        
        # Audio buffers
        self.input_buffer = deque(maxlen=self.buffer_size * 10)
        self.output_buffer = deque(maxlen=self.buffer_size * 10)
        
        # Processing queues
        self.processing_queue = queue.Queue(maxsize=10)
        
        # Threading
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.processed_chunks = 0
        self.processing_times = deque(maxlen=100)
        self.total_samples_processed = 0
        
        print(f"RealTimeProcessor initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Chunk size: {chunk_size} samples ({chunk_size/sample_rate*1000:.1f} ms)")
        print(f"  Hop size: {self.hop_size} samples")
        print(f"  Overlap: {overlap*100:.0f}%")
        print(f"  Device: {device}")
        print(f"  Max latency: {max_latency_ms} ms")
    
    def _audio_callback(self, indata, outdata, frames, time, status):
        """Audio callback for sounddevice stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_in = indata[:, 0].copy()
        else:
            audio_in = indata[:, 0].copy()
        
        # Add to input buffer
        self.input_buffer.extend(audio_in)
        
        # Get processed audio from output buffer
        output_samples = []
        for _ in range(frames):
            if self.output_buffer:
                output_samples.append(self.output_buffer.popleft())
            else:
                output_samples.append(0.0)  # Silence if no processed audio available
        
        # Output processed audio
        outdata[:, 0] = np.array(output_samples, dtype=np.float32)
        
        # Trigger processing if we have enough samples
        if len(self.input_buffer) >= self.chunk_size:
            chunk = np.array(list(self.input_buffer)[:self.chunk_size])
            # Remove processed samples (with overlap)
            for _ in range(self.hop_size):
                if self.input_buffer:
                    self.input_buffer.popleft()
            
            # Queue for processing
            try:
                self.processing_queue.put_nowait(chunk)
            except queue.Full:
                print("Processing queue full, dropping chunk")
    
    def _processing_worker(self):
        """Worker thread for audio processing."""
        while self.is_running:
            try:
                # Get chunk to process
                chunk = self.processing_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process with model
                with torch.no_grad():
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(chunk).float().to(self.device)
                    
                    # Ensure proper sample rate
                    if len(audio_tensor) != self.chunk_size:
                        # Pad or trim to expected size
                        if len(audio_tensor) < self.chunk_size:
                            audio_tensor = F.pad(audio_tensor, (0, self.chunk_size - len(audio_tensor)))
                        else:
                            audio_tensor = audio_tensor[:self.chunk_size]
                    
                    # Process
                    enhanced = self.model.process_chunk(audio_tensor)
                    
                    # Convert back to numpy
                    enhanced_audio = enhanced.cpu().numpy()
                
                # Add to output buffer (only hop_size samples to maintain overlap)
                output_chunk = enhanced_audio[:self.hop_size]
                self.output_buffer.extend(output_chunk)
                
                # Update statistics
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                self.processed_chunks += 1
                self.total_samples_processed += len(chunk)
                
                # Check latency warning
                if processing_time > self.max_latency_ms:
                    print(f"Warning: Processing time {processing_time:.1f}ms exceeds target {self.max_latency_ms}ms")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def start(self):
        """Start real-time processing."""
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not available. Install with: pip install sounddevice")
        
        if self.is_running:
            print("Already running")
            return
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.start()
        
        # Check available devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0 and device['max_output_channels'] > 0:
                print(f"  {i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")
        
        # Start audio stream
        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.hop_size,
                latency='low'
            )
            
            self.stream.start()
            print(f"\nReal-time processing started!")
            print("Speak into your microphone. Press Ctrl+C to stop.")
            
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop real-time processing."""
        if not self.is_running:
            return
        
        print("\nStopping real-time processing...")
        self.is_running = False
        
        # Stop audio stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        print("Real-time processing stopped.")
        self.print_statistics()
    
    def print_statistics(self):
        """Print processing statistics."""
        if self.processed_chunks > 0:
            avg_time = np.mean(self.processing_times)
            max_time = np.max(self.processing_times)
            min_time = np.min(self.processing_times)
            
            total_duration = self.total_samples_processed / self.sample_rate
            realtime_factor = total_duration / (self.processed_chunks * avg_time / 1000)
            
            print(f"\nProcessing Statistics:")
            print(f"  Processed chunks: {self.processed_chunks}")
            print(f"  Total audio duration: {total_duration:.1f}s")
            print(f"  Average processing time: {avg_time:.1f}ms")
            print(f"  Min/Max processing time: {min_time:.1f}/{max_time:.1f}ms")
            print(f"  Real-time factor: {realtime_factor:.2f}x")
            
            if realtime_factor < 1.0:
                print("  ⚠️  Warning: Processing slower than real-time!")
            else:
                print("  ✅ Real-time processing achieved!")


class FileProcessor:
    """Process audio files with DeepFilterNet2."""
    
    def __init__(self, model: DeepFilterNet2, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def process_file(
        self, 
        input_path: str, 
        output_path: str = None,
        chunk_size: int = 8192,  # Larger chunks for file processing
        progress_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Process an audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save output (optional)
            chunk_size: Processing chunk size
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (enhanced_audio, sample_rate)
        """
        print(f"Processing file: {input_path}")
        
        # Load audio file
        try:
            audio, sr = librosa.load(input_path, sr=self.model.sr, mono=True)
            print(f"Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz")
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")
        
        # Process in chunks
        enhanced_chunks = []
        total_chunks = int(np.ceil(len(audio) / chunk_size))
        
        print(f"Processing {total_chunks} chunks...")
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Process chunk
            with torch.no_grad():
                chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
                enhanced_chunk = self.model.process_chunk(chunk_tensor)
                enhanced_chunks.append(enhanced_chunk.cpu().numpy())
            
            # Progress callback
            if progress_callback:
                progress = (i // chunk_size + 1) / total_chunks
                progress_callback(progress)
            
            if (i // chunk_size + 1) % 10 == 0:
                print(f"  Processed {i // chunk_size + 1}/{total_chunks} chunks")
        
        # Concatenate chunks
        enhanced_audio = np.concatenate(enhanced_chunks)
        
        # Trim to original length
        enhanced_audio = enhanced_audio[:len(audio)]
        
        # Save output file
        if output_path:
            try:
                sf.write(output_path, enhanced_audio, sr)
                print(f"Saved enhanced audio: {output_path}")
            except Exception as e:
                print(f"Warning: Failed to save output file: {e}")
        
        return enhanced_audio, sr


def calculate_audio_metrics(original: np.ndarray, enhanced: np.ndarray, sr: int) -> dict:
    """
    Calculate audio quality metrics.
    
    Args:
        original: Original audio
        enhanced: Enhanced audio
        sr: Sample rate
        
    Returns:
        Dictionary of metrics
    """
    try:
        from scipy.signal import correlate
        from scipy.stats import pearsonr
        
        # Ensure same length
        min_len = min(len(original), len(enhanced))
        original = original[:min_len]
        enhanced = enhanced[:min_len]
        
        # Pearson correlation
        correlation, _ = pearsonr(original, enhanced)
        
        # Cross-correlation peak
        cross_corr = correlate(enhanced, original, mode='full')
        max_cross_corr = np.max(np.abs(cross_corr)) / (np.linalg.norm(original) * np.linalg.norm(enhanced) + 1e-10)
        
        # SNR improvement estimate
        original_power = np.mean(original ** 2)
        noise_estimate = np.mean((original - enhanced) ** 2)
        snr_improvement = 10 * np.log10(original_power / (noise_estimate + 1e-10))
        
        # RMS levels
        original_rms = np.sqrt(np.mean(original ** 2))
        enhanced_rms = np.sqrt(np.mean(enhanced ** 2))
        
        # Spectral metrics
        original_fft = np.fft.rfft(original)
        enhanced_fft = np.fft.rfft(enhanced)
        spectral_correlation = np.corrcoef(np.abs(original_fft), np.abs(enhanced_fft))[0, 1]
        
        metrics = {
            'correlation': correlation,
            'cross_correlation': max_cross_corr,
            'snr_improvement': snr_improvement,
            'original_rms': original_rms,
            'enhanced_rms': enhanced_rms,
            'rms_ratio': enhanced_rms / (original_rms + 1e-10),
            'spectral_correlation': spectral_correlation
        }
        
        print(f"\nAudio Quality Metrics:")
        print(f"  Pearson correlation: {correlation:.3f}")
        print(f"  Max cross-correlation: {max_cross_corr:.3f}")
        print(f"  Estimated SNR improvement: {snr_improvement:.1f} dB")
        print(f"  RMS ratio (enhanced/original): {metrics['rms_ratio']:.3f}")
        print(f"  Spectral correlation: {spectral_correlation:.3f}")
        
        return metrics
        
    except ImportError:
        print("scipy not available for detailed metrics")
        return {'correlation': None}
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'error': str(e)}


def test_model_performance(model: DeepFilterNet2, test_file: str = None):
    """
    Test model performance with sample audio.
    
    Args:
        model: DeepFilterNet2 model to test
        test_file: Optional test file path
    """
    print("\n" + "="*50)
    print("DEEPFILTERNET2 PERFORMANCE TEST")
    print("="*50)
    
    if test_file and Path(test_file).exists():
        # Test with provided file
        processor = FileProcessor(model)
        
        print(f"\nTesting with file: {test_file}")
        
        # Load original
        original_audio, sr = librosa.load(test_file, sr=model.sr, mono=True)
        print(f"Original audio: {len(original_audio)/sr:.2f}s at {sr}Hz")
        
        # Process
        start_time = time.time()
        enhanced_audio, _ = processor.process_file(test_file)
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_audio_metrics(original_audio, enhanced_audio, sr)
        
        # Performance metrics
        audio_duration = len(original_audio) / sr
        realtime_factor = audio_duration / processing_time
        
        print(f"\nPerformance Metrics:")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Real-time factor: {realtime_factor:.2f}x")
        
        if realtime_factor >= 1.0:
            print("  ✅ Real-time processing achieved!")
        else:
            print("  ⚠️  Processing slower than real-time")
        
        # Save enhanced file for comparison
        output_file = Path(test_file).stem + "_enhanced.wav"
        sf.write(output_file, enhanced_audio, sr)
        print(f"\nEnhanced audio saved: {output_file}")
        
    else:
        # Test with synthetic signal
        print("\nTesting with synthetic noisy signal...")
        
        # Create test signal: clean speech + noise
        duration = 5.0  # 5 seconds
        sr = model.sr
        t = np.linspace(0, duration, int(duration * sr))
        
        # Synthetic speech-like signal
        speech = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)  # Decaying tone
        speech += 0.3 * np.sin(2 * np.pi * 880 * t) * np.exp(-t * 0.3)  # Harmonic
        
        # Add noise
        noise = np.random.normal(0, 0.3, len(speech))
        noisy = speech + noise
        
        # Normalize
        noisy = noisy / np.max(np.abs(noisy)) * 0.8
        
        print(f"Synthetic test signal: {duration}s at {sr}Hz")
        
        # Process
        start_time = time.time()
        with torch.no_grad():
            noisy_tensor = torch.from_numpy(noisy).float()
            enhanced_tensor = model(noisy_tensor)
            enhanced = enhanced_tensor.numpy()
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_audio_metrics(speech, enhanced, sr)
        
        # Performance
        realtime_factor = duration / processing_time
        print(f"\nPerformance Metrics:")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Real-time factor: {realtime_factor:.2f}x")
        
        # Save test files
        sf.write("test_noisy.wav", noisy, sr)
        sf.write("test_enhanced.wav", enhanced, sr)
        sf.write("test_clean.wav", speech, sr)
        print("\nTest files saved: test_noisy.wav, test_enhanced.wav, test_clean.wav")
    
    print("\nTest completed!")


if __name__ == "__main__":
    # Test real-time processor
    from deepfilternet2 import create_model
    
    print("Creating DeepFilterNet2 model...")
    model = create_model()
    
    print("\nTesting file processing...")
    test_model_performance(model)