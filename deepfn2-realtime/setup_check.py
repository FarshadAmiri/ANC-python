#!/usr/bin/env python3
"""
Setup and system check for DeepFilterNet2 Real-Time Speech Denoiser
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        ('torch', '2.0.0'),
        ('torchaudio', '2.0.0'), 
        ('numpy', '1.21.0'),
        ('soundfile', '0.12.1'),
        ('librosa', '0.10.0'),
        ('scipy', '1.9.0'),
    ]
    
    optional_packages = [
        ('sounddevice', '0.4.6'),  # For real-time processing
        ('h5py', '3.7.0'),        # For model storage
    ]
    
    print("\nChecking required dependencies...")
    all_ok = True
    
    for package, min_version in required_packages:
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"✅ {package}: {version}")
            else:
                print(f"✅ {package}: installed (version unknown)")
        except ImportError:
            print(f"❌ {package}: not installed")
            all_ok = False
    
    print("\nChecking optional dependencies...")
    for package, min_version in optional_packages:
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"✅ {package}: {version}")
            else:
                print(f"✅ {package}: installed (version unknown)")
        except ImportError:
            print(f"⚠️  {package}: not installed (optional)")
    
    return all_ok


def check_audio_system():
    """Check if audio system is working."""
    print("\nChecking audio system...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Check for input devices
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        print(f"✅ Found {len(input_devices)} input device(s)")
        print(f"✅ Found {len(output_devices)} output device(s)")
        
        if len(input_devices) == 0:
            print("⚠️  No microphone detected")
            return False
        
        if len(output_devices) == 0:
            print("⚠️  No audio output device detected")
            return False
        
        # Show a few devices
        print("\nAvailable input devices:")
        for i, device in enumerate(input_devices[:3]):
            print(f"  {device['name']}")
        
        return True
        
    except ImportError:
        print("⚠️  sounddevice not available (real-time processing disabled)")
        print("   Install with: pip install sounddevice")
        return True  # Not critical for file processing
    except Exception as e:
        print(f"❌ Audio system check failed: {e}")
        return False


def check_torch_functionality():
    """Check if PyTorch is working correctly."""
    print("\nChecking PyTorch functionality...")
    
    try:
        import torch
        
        # Check basic tensor operations
        x = torch.randn(10, 10)
        y = torch.mm(x, x)
        print("✅ Basic tensor operations working")
        
        # Check CUDA if available
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            # Test CUDA tensor
            x_cuda = torch.randn(10, 10).cuda()
            y_cuda = torch.mm(x_cuda, x_cuda)
            print("✅ CUDA tensor operations working")
        else:
            print("ℹ️  CUDA not available (using CPU)")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch functionality check failed: {e}")
        return False


def test_model_creation():
    """Test if DeepFilterNet2 model can be created."""
    print("\nTesting model creation...")
    
    try:
        from deepfilternet2 import create_model
        
        model = create_model(device="cpu")
        print("✅ Model creation successful")
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1024)  # 1024 samples
        output = model.process_chunk(dummy_input)
        print(f"✅ Model inference working (input: {dummy_input.shape}, output: {output.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def install_dependencies():
    """Install missing dependencies."""
    print("\nAttempting to install missing dependencies...")
    
    try:
        # Install from requirements.txt if available
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            print(f"Installing from {requirements_file}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
        else:
            # Install manually
            packages = [
                "torch>=2.0.0",
                "torchaudio>=2.0.0", 
                "numpy>=1.21.0",
                "soundfile>=0.12.1",
                "librosa>=0.10.0",
                "scipy>=1.9.0",
                "sounddevice>=0.4.6",
                "h5py>=3.7.0"
            ]
            
            for package in packages:
                print(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
        
        print("✅ Dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def system_info():
    """Print system information."""
    import platform
    
    print(f"\nSystem Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {sys.version}")
    
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError:
        print(f"  PyTorch: not installed")


def main():
    """Main setup check."""
    print("DeepFilterNet2 Real-Time Speech Denoiser - System Check")
    print("=" * 60)
    
    system_info()
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Audio System", check_audio_system), 
        ("PyTorch", check_torch_functionality),
        ("Model Creation", test_model_creation),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SYSTEM CHECK SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All checks passed! DeepFilterNet2 is ready to use.")
        print("\nQuick start:")
        print("  python main.py --test                  # Test with sample")
        print("  python main.py --mode realtime         # Real-time processing")
        print("  python main.py --mode file -i in.wav   # File processing")
        print("  python examples.py                     # Run usage examples")
    else:
        print("\n❌ Some checks failed. Please resolve the issues above.")
        
        if not results.get("Dependencies", True):
            choice = input("\nWould you like to try installing missing dependencies? (y/n): ")
            if choice.lower() == 'y':
                if install_dependencies():
                    print("\nDependencies installed. Please run the setup check again.")
                else:
                    print("\nFailed to install dependencies automatically.")
        
        print("\nTroubleshooting:")
        print("  - Run: pip install -r requirements.txt")
        print("  - For audio issues: sudo apt-get install portaudio19-dev (Linux)")
        print("  - For Windows: Install Visual C++ redistributable")
        print("  - Check microphone and speaker connections")
        print("  - Verify audio permissions")


if __name__ == "__main__":
    main()