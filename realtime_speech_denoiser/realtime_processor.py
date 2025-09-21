import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
from typing import Optional, Callable
from collections import deque

from deep_filter import DeepFilter, create_model


class RealTimeProcessor:
    """Real-time audio processor for speech enhancement"""
    
    def __init__(
        self,
        model: DeepFilter,
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
        self.buffer_size = chunk_size + self.hop_size  # Extra buffer for overlap
        
        # Audio buffers
        self.input_buffer = deque(maxlen=self.buffer_size * 10)  # Large input buffer
        self.output_buffer = deque(maxlen=self.buffer_size * 10)  # Large output buffer
        
        # Processing queue
        self.processing_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Threading
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.processed_chunks = 0
        self.processing_times = deque(maxlen=100)
        
        print(f"RealTimeProcessor initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Chunk size: {chunk_size} samples ({chunk_size/sample_rate*1000:.1f} ms)")
        print(f"  Hop size: {self.hop_size} samples")
        print(f"  Overlap: {overlap*100:.0f}%")
        print(f"  Device: {device}")
    
    def _audio_callback(self, indata, outdata, frames, time, status):
        """Audio callback for sounddevice stream"""
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
        """Worker thread for audio processing"""
        while self.is_running:
            try:
                # Get chunk to process
                chunk = self.processing_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process with model
                with torch.no_grad():
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(chunk).float().to(self.device)
                    
                    # Process
                    enhanced = self.model(audio_tensor)
                    
                    # Convert back to numpy
                    if enhanced.dim() > 1:
                        enhanced = enhanced.squeeze(0)
                    enhanced_audio = enhanced.cpu().numpy()
                
                # Add to output buffer
                self.output_buffer.extend(enhanced_audio[:self.hop_size])
                
                # Update statistics
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                self.processed_chunks += 1
                
                # Check latency
                if processing_time > self.max_latency_ms:
                    print(f"Warning: Processing time {processing_time:.1f}ms exceeds max latency {self.max_latency_ms}ms")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def start(self):
        """Start real-time processing"""
        if self.is_running:
            print("Already running")
            return
        
        print("Starting real-time speech enhancement...")
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.start()
        
        # Start audio stream
        self.stream = sd.Stream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.hop_size,
            dtype=np.float32,
            callback=self._audio_callback,
            latency='low'
        )
        
        self.stream.start()
        print("Real-time processing started. Press Ctrl+C to stop.")
    
    def stop(self):
        """Stop real-time processing"""
        if not self.is_running:
            return
        
        print("Stopping real-time processing...")
        
        self.is_running = False
        
        # Stop audio stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join()
        
        self.print_statistics()
        print("Real-time processing stopped.")
    
    def print_statistics(self):
        """Print processing statistics"""
        if not self.processing_times:
            return
        
        avg_time = np.mean(self.processing_times)
        max_time = np.max(self.processing_times)
        min_time = np.min(self.processing_times)
        
        print(f"\nProcessing Statistics:")
        print(f"  Processed chunks: {self.processed_chunks}")
        print(f"  Average processing time: {avg_time:.1f} ms")
        print(f"  Min processing time: {min_time:.1f} ms")
        print(f"  Max processing time: {max_time:.1f} ms")
        print(f"  Real-time factor: {(self.chunk_size/self.sample_rate*1000)/avg_time:.2f}x")


class FileProcessor:
    """Process audio files with the same model used for real-time"""
    
    def __init__(self, model: DeepFilter, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def process_file(
        self, 
        input_path: str, 
        output_path: str,
        chunk_size: int = 8192,  # Larger chunks for file processing
        progress_callback: Optional[Callable] = None
    ):
        """Process an audio file"""
        print(f"Processing file: {input_path}")
        
        # Load audio
        audio, sr = sf.read(input_path, dtype='float32')
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio[:, 0]
        
        # Resample if needed
        if sr != self.model.sr:
            print(f"Resampling from {sr} Hz to {self.model.sr} Hz")
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.model.sr)
            sr = self.model.sr
        
        print(f"Audio length: {len(audio)/sr:.2f} seconds")
        
        # Process in chunks
        enhanced_audio = []
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        
        with torch.no_grad():
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                
                # Pad if necessary
                if len(chunk) < chunk_size and i + chunk_size >= len(audio):
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                # Convert to tensor
                chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
                
                # Process
                enhanced_chunk = self.model(chunk_tensor)
                
                # Convert back to numpy
                if enhanced_chunk.dim() > 1:
                    enhanced_chunk = enhanced_chunk.squeeze(0)
                enhanced_chunk = enhanced_chunk.cpu().numpy()
                
                # Trim to original length for last chunk
                if i + chunk_size >= len(audio):
                    original_chunk_len = len(audio) - i
                    enhanced_chunk = enhanced_chunk[:original_chunk_len]
                
                enhanced_audio.append(enhanced_chunk)
                
                # Progress callback
                if progress_callback:
                    progress = (i // chunk_size + 1) / num_chunks
                    progress_callback(progress)
        
        # Concatenate results
        enhanced_audio = np.concatenate(enhanced_audio)
        
        # Save output
        sf.write(output_path, enhanced_audio, sr)
        print(f"Enhanced audio saved to: {output_path}")
        
        return enhanced_audio, sr


def process_audio_similarity(original: np.ndarray, enhanced: np.ndarray, sr: int):
    """Calculate similarity metrics between original and enhanced audio"""
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
        max_cross_corr = np.max(np.abs(cross_corr)) / (np.linalg.norm(original) * np.linalg.norm(enhanced))
        
        # SNR improvement (rough estimate)
        original_energy = np.mean(original ** 2)
        noise_estimate = np.mean((original - enhanced) ** 2)
        snr_improvement = 10 * np.log10(original_energy / (noise_estimate + 1e-10))
        
        print(f"\nSimilarity Metrics:")
        print(f"  Pearson correlation: {correlation:.3f}")
        print(f"  Max cross-correlation: {max_cross_corr:.3f}")
        print(f"  Estimated SNR improvement: {snr_improvement:.1f} dB")
        
        return {
            'correlation': correlation,
            'cross_correlation': max_cross_corr,
            'snr_improvement': snr_improvement
        }
        
    except ImportError:
        print("scipy not available for similarity metrics")
        return None