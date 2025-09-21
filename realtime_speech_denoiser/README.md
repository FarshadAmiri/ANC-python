# Real-Time Speech Denoiser

A Python-based real-time speech denoiser using deep learning techniques inspired by DeepFilterNet. This system provides low-latency noise suppression for both real-time microphone input and pre-recorded audio files.

## Features

- **Real-time processing**: Low-latency speech enhancement from microphone input
- **File processing**: Batch processing of audio files
- **Deep learning based**: Neural network using ERB filterbank and LSTM layers
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Offline operation**: No API calls required, completely offline
- **CLI interface**: Easy-to-use command-line interface
- **Performance metrics**: Built-in similarity measurements and processing statistics

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## Installation

1. **Install system dependencies** (Linux/macOS):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install portaudio19-dev python3-dev build-essential
   
   # macOS
   brew install portaudio
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Real-time Processing

Process microphone input in real-time:

```bash
python main.py --mode realtime
```

Options:
- `--sample-rate`: Sample rate (default: 48000)
- `--chunk-size`: Processing chunk size (default: 1024)
- `--device`: Device to use (auto/cpu/cuda)

### File Processing

Process a single audio file:

```bash
python main.py --mode file --input noisy_audio.wav --output clean_audio.wav
```

### Test with Sample

Test the system with the provided noisy sample:

```bash
python main.py --test
```

This will process the `noisy_fish.wav` file and create a denoised version for comparison.

## Architecture

The speech denoiser consists of several key components:

### 1. ERB Filterbank
- Converts linear frequency spectrum to perceptually relevant ERB (Equivalent Rectangular Bandwidth) scale
- Provides better frequency resolution for speech processing

### 2. Feature Extraction
- Combines ERB features with linear frequency features
- Creates rich representation for the neural network

### 3. LSTM Network
- Temporal modeling for speech enhancement
- Learns to distinguish between speech and noise patterns

### 4. Magnitude Mask Generation
- Generates frequency-dependent masks
- Applied to input spectrum to suppress noise

### 5. Real-time Processing Pipeline
- Overlap-add processing for smooth audio output
- Multi-threaded architecture for low latency
- Adaptive buffering to handle processing variations

## Performance

The system is designed for real-time operation with the following characteristics:

- **Latency**: Typically < 50ms on modern hardware
- **Processing**: Real-time factor > 1.0x on CPU
- **Memory**: < 500MB RAM usage
- **Compatibility**: Works with standard audio interfaces

## Technical Details

### Audio Processing
- **Sample Rate**: 48 kHz (configurable)
- **Frame Size**: 1024 samples (≈21ms at 48kHz)
- **Overlap**: 50% for smooth reconstruction
- **Bit Depth**: 32-bit float processing

### Model Architecture
- **Input**: ERB + linear frequency features
- **Hidden**: 256-unit LSTM with 3 layers
- **Output**: Frequency-domain magnitude mask
- **Activation**: Sigmoid for mask generation

### Real-time Optimization
- **Threading**: Separate audio I/O and processing threads
- **Buffering**: Adaptive buffering for consistent latency
- **Memory**: Efficient tensor operations with PyTorch
- **Device**: Automatic GPU acceleration when available

## File Structure

```
realtime_speech_denoiser/
├── __init__.py              # Package initialization
├── main.py                  # CLI interface
├── deep_filter.py           # Neural network model
├── realtime_processor.py    # Real-time processing engine
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Examples

### Basic Real-time Usage

```python
from realtime_speech_denoiser import RealTimeProcessor, create_model

# Create model
model = create_model()

# Create processor
processor = RealTimeProcessor(model)

# Start real-time processing
processor.start()

# ... processing happens in background ...

# Stop when done
processor.stop()
```

### File Processing

```python
from realtime_speech_denoiser import FileProcessor, create_model

# Create model and processor
model = create_model()
processor = FileProcessor(model)

# Process file
enhanced_audio, sr = processor.process_file("input.wav", "output.wav")
```

## Troubleshooting

### Audio Issues
- **No audio input**: Check microphone permissions and device connections
- **High latency**: Reduce chunk size or use GPU acceleration
- **Audio dropouts**: Increase buffer sizes or reduce processing complexity

### Performance Issues
- **Slow processing**: Enable GPU acceleration with CUDA
- **Memory errors**: Reduce batch size or use CPU processing
- **Installation issues**: Install system audio dependencies

### Common Solutions
1. **Microphone not detected**: 
   ```bash
   python -c "import sounddevice; print(sounddevice.query_devices())"
   ```

2. **CUDA not available**:
   - Install PyTorch with CUDA support
   - Verify GPU drivers are installed

3. **Audio crackling**:
   - Increase audio buffer size
   - Use lower sample rate (e.g., 16kHz)

## Contributing

This project is designed to be modular and extensible. Key areas for improvement:

- **Model training**: Train on larger speech datasets
- **Architecture**: Experiment with transformer-based models
- **Optimization**: Further latency reductions
- **Features**: Additional noise types and environments

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Inspired by the DeepFilterNet paper and implementation:
- "DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio" 
- "DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices"

The implementation uses principles from these works while providing a simplified, real-time focused solution.