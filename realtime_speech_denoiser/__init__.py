"""
Real-Time Speech Denoiser using Deep Learning

A Python-based real-time speech denoiser inspired by DeepFilterNet principles.
Supports both real-time microphone processing and file-based audio enhancement.

Key Features:
- Real-time speech enhancement with low latency
- File-based audio processing
- Neural network based on ERB filterbank and LSTM
- Cross-platform compatibility
- CLI interface for easy use

Usage:
    from realtime_speech_denoiser import SpeechDenoiser
    
    denoiser = SpeechDenoiser()
    denoiser.run_realtime()  # For real-time processing
    denoiser.process_file("input.wav", "output.wav")  # For file processing
"""

from .deep_filter import DeepFilter, create_model, load_pretrained_weights
from .realtime_processor import RealTimeProcessor, FileProcessor, process_audio_similarity

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Real-Time Speech Denoiser using Deep Learning"

__all__ = [
    'DeepFilter',
    'create_model', 
    'load_pretrained_weights',
    'RealTimeProcessor',
    'FileProcessor',
    'process_audio_similarity'
]