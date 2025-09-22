"""
DeepFilterNet2 Python API

Easy-to-use Python API for integrating DeepFilterNet2 speech enhancement
into other projects.
"""

from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

from deepfilternet2 import DeepFilterNet2, create_model
from realtime_processor import RealTimeProcessor, FileProcessor, calculate_audio_metrics


class DeepFilterNet2API:
    """
    High-level API for DeepFilterNet2 speech enhancement.
    
    This class provides simple methods for both real-time and batch processing
    of audio data for noise suppression.
    """
    
    def __init__(
        self, 
        model_name: str = "deepfilternet2_base",
        device: str = "auto"
    ):
        """
        Initialize the DeepFilterNet2 API.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        # Resolve device immediately
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None
        self._realtime_processor = None
    
    @property
    def model(self) -> DeepFilterNet2:
        """Get the loaded model, loading it if necessary."""
        if self._model is None:
            self._model = create_model(self.model_name, self.device)
        return self._model
    
    def enhance_audio(
        self, 
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Enhance audio by removing noise.
        
        Args:
            audio: Audio data as numpy array or path to audio file
            sample_rate: Sample rate of the audio (required if audio is numpy array)
            
        Returns:
            Tuple of (enhanced_audio, sample_rate)
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio_data, sr = librosa.load(audio, sr=self.model.sr, mono=True)
        else:
            audio_data = audio
            sr = sample_rate or self.model.sr
            
            # Resample if necessary
            if sr != self.model.sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.model.sr)
                sr = self.model.sr
        
        # Ensure numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()
        
        # Process with model
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
            enhanced_tensor = self.model(audio_tensor)
            enhanced_audio = enhanced_tensor.cpu().numpy()
        
        return enhanced_audio, sr
    
    def enhance_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        return_metrics: bool = False
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
        """
        Enhance an audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio (optional)
            return_metrics: Whether to return quality metrics
            
        Returns:
            Success status, optionally with metrics dictionary
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = input_path.stem + "_enhanced" + input_path.suffix
        
        try:
            # Load original audio
            original_audio, sr = librosa.load(input_path, sr=self.model.sr, mono=True)
            
            # Enhance audio
            enhanced_audio, _ = self.enhance_audio(original_audio, sr)
            
            # Save enhanced audio
            sf.write(output_path, enhanced_audio, sr)
            
            # Calculate metrics if requested
            if return_metrics:
                metrics = calculate_audio_metrics(original_audio, enhanced_audio, sr)
                return True, metrics
            
            return True
            
        except Exception as e:
            if return_metrics:
                return False, {"error": str(e)}
            return False
    
    def start_realtime(
        self,
        sample_rate: int = 48000,
        chunk_size: int = 1024,
        **kwargs
    ) -> RealTimeProcessor:
        """
        Start real-time audio processing.
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Processing chunk size
            **kwargs: Additional arguments for RealTimeProcessor
            
        Returns:
            RealTimeProcessor instance
        """
        if self._realtime_processor and self._realtime_processor.is_running:
            raise RuntimeError("Real-time processing already active")
        
        self._realtime_processor = RealTimeProcessor(
            model=self.model,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            device=self.device,
            **kwargs
        )
        
        self._realtime_processor.start()
        return self._realtime_processor
    
    def stop_realtime(self):
        """Stop real-time audio processing."""
        if self._realtime_processor:
            self._realtime_processor.stop()
            self._realtime_processor = None
    
    def is_realtime_active(self) -> bool:
        """Check if real-time processing is active."""
        return (self._realtime_processor is not None and 
                self._realtime_processor.is_running)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.model.sr,
            "n_fft": self.model.n_fft,
            "hop_length": self.model.hop_length,
            "n_erb": self.model.n_erb,
            "hidden_size": self.model.hidden_size
        }


# Convenience functions for simple use cases
def enhance_audio_simple(
    audio_input: Union[np.ndarray, str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "auto"
) -> np.ndarray:
    """
    Simple function to enhance audio with default settings.
    
    Args:
        audio_input: Audio data (numpy array) or path to audio file
        output_path: Optional path to save enhanced audio
        device: Device to run on
        
    Returns:
        Enhanced audio as numpy array
    """
    api = DeepFilterNet2API(device=device)
    
    if isinstance(audio_input, (str, Path)):
        enhanced_audio, sr = api.enhance_audio(audio_input)
        if output_path:
            sf.write(output_path, enhanced_audio, sr)
    else:
        enhanced_audio, sr = api.enhance_audio(audio_input, sample_rate=48000)
        if output_path:
            sf.write(output_path, enhanced_audio, sr)
    
    return enhanced_audio


def create_realtime_processor(
    device: str = "auto",
    sample_rate: int = 48000,
    chunk_size: int = 1024
) -> DeepFilterNet2API:
    """
    Create and configure a real-time processor.
    
    Args:
        device: Device to run on
        sample_rate: Audio sample rate
        chunk_size: Processing chunk size
        
    Returns:
        Configured DeepFilterNet2API instance
    """
    api = DeepFilterNet2API(device=device)
    return api


# Context manager for real-time processing
class RealTimeContext:
    """Context manager for real-time audio processing."""
    
    def __init__(self, api: DeepFilterNet2API, **kwargs):
        self.api = api
        self.kwargs = kwargs
        self.processor = None
    
    def __enter__(self):
        self.processor = self.api.start_realtime(**self.kwargs)
        return self.processor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api.stop_realtime()


# Example usage functions
def example_file_processing():
    """Example of file processing."""
    print("DeepFilterNet2 File Processing Example")
    print("=" * 40)
    
    # Create API instance
    api = DeepFilterNet2API()
    
    # Enhance a file
    input_file = "noisy_audio.wav"  # Replace with your file
    output_file = "enhanced_audio.wav"
    
    success, metrics = api.enhance_file(
        input_file, 
        output_file, 
        return_metrics=True
    )
    
    if success:
        print(f"Successfully enhanced {input_file} -> {output_file}")
        print(f"Quality metrics: {metrics}")
    else:
        print(f"Enhancement failed: {metrics.get('error', 'Unknown error')}")


def example_realtime_processing():
    """Example of real-time processing."""
    print("DeepFilterNet2 Real-Time Processing Example")
    print("=" * 40)
    
    # Create API instance
    api = DeepFilterNet2API()
    
    # Start real-time processing
    print("Starting real-time processing...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    
    try:
        with RealTimeContext(api) as processor:
            # Keep running until interrupted
            import time
            while True:
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("\nStopping real-time processing...")


def example_numpy_processing():
    """Example of processing numpy audio data."""
    print("DeepFilterNet2 Numpy Processing Example")
    print("=" * 40)
    
    # Create API instance
    api = DeepFilterNet2API()
    
    # Create synthetic noisy audio
    import numpy as np
    duration = 3.0  # 3 seconds
    sr = 48000
    t = np.linspace(0, duration, int(duration * sr))
    
    # Synthetic speech + noise
    speech = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)
    noise = np.random.normal(0, 0.3, len(speech))
    noisy_audio = speech + noise
    
    # Enhance audio
    enhanced_audio, sr = api.enhance_audio(noisy_audio, sample_rate=sr)
    
    # Save results
    sf.write("example_noisy.wav", noisy_audio, sr)
    sf.write("example_enhanced.wav", enhanced_audio, sr)
    
    print("Created example_noisy.wav and example_enhanced.wav")
    print(f"Original shape: {noisy_audio.shape}")
    print(f"Enhanced shape: {enhanced_audio.shape}")


if __name__ == "__main__":
    # Run examples
    print("DeepFilterNet2 API Examples")
    print("=" * 50)
    
    # Example with numpy arrays
    example_numpy_processing()
    
    print("\nTo run real-time example, call example_realtime_processing()")
    print("To run file processing example, call example_file_processing()")