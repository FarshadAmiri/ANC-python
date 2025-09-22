"""
DeepFilterNet2 Real-Time Speech Denoiser

A Python-based real-time speech noise suppressor using DeepFilterNet2 architecture
with pre-trained models for high-quality noise suppression.

Main components:
- DeepFilterNet2: Core neural network model
- RealTimeProcessor: Real-time audio processing
- FileProcessor: Batch file processing
- DeepFilterNet2API: High-level Python API

Usage:
    # Simple file processing
    from deepfn2_realtime import enhance_audio_simple
    enhanced = enhance_audio_simple("noisy.wav", "clean.wav")
    
    # Real-time processing
    from deepfn2_realtime import DeepFilterNet2API
    api = DeepFilterNet2API()
    api.start_realtime()
    
    # Custom processing
    from deepfn2_realtime import DeepFilterNet2, RealTimeProcessor
    model = DeepFilterNet2()
    processor = RealTimeProcessor(model)
"""

__version__ = "1.0.0"
__author__ = "DeepFilterNet2 Implementation"
__email__ = "support@example.com"

# Core model and components
from .deepfilternet2 import (
    DeepFilterNet2,
    ERBFilterBank,
    ComplexLinear,
    ComplexLSTM,
    create_model,
    download_pretrained_model
)

# Processing components  
from .realtime_processor import (
    RealTimeProcessor,
    FileProcessor,
    calculate_audio_metrics,
    test_model_performance
)

# High-level API
from .api import (
    DeepFilterNet2API,
    enhance_audio_simple,
    create_realtime_processor,
    RealTimeContext
)

__all__ = [
    # Core model
    "DeepFilterNet2",
    "ERBFilterBank", 
    "ComplexLinear",
    "ComplexLSTM",
    "create_model",
    "download_pretrained_model",
    
    # Processing
    "RealTimeProcessor",
    "FileProcessor", 
    "calculate_audio_metrics",
    "test_model_performance",
    
    # API
    "DeepFilterNet2API",
    "enhance_audio_simple",
    "create_realtime_processor", 
    "RealTimeContext",
    
    # Version info
    "__version__"
]