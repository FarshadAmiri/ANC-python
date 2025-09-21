#!/usr/bin/env python3
"""
Setup and installation verification for Real-Time Speech Denoiser
"""

import sys
import subprocess
import importlib
import platform


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('torch', '2.0.0'),
        ('torchaudio', '2.0.0'),
        ('numpy', '1.21.0'),
        ('sounddevice', '0.4.6'),
        ('soundfile', '0.12.1'),
        ('librosa', '0.10.0'),
        ('scipy', '1.9.0'),
    ]
    
    print("\nChecking dependencies...")
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
    
    return all_ok


def check_audio_system():
    """Check if audio system is working"""
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
        
        return True
        
    except Exception as e:
        print(f"❌ Audio system check failed: {e}")
        return False


def check_torch_functionality():
    """Check if PyTorch is working correctly"""
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
        else:
            print("ℹ️  CUDA not available (using CPU)")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch functionality check failed: {e}")
        return False


def install_dependencies():
    """Install missing dependencies"""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def test_model_creation():
    """Test if the model can be created successfully"""
    print("\nTesting model creation...")
    
    try:
        sys.path.append('.')
        from deep_filter import create_model
        
        model = create_model()
        print("✅ Model creation successful")
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1024)
        with torch.no_grad():
            output = model(dummy_input)
        print("✅ Model forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def run_system_check():
    """Run complete system check"""
    print("Real-Time Speech Denoiser - System Check")
    print("=" * 50)
    
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print()
    
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
            print(f"❌ {name} check failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("SYSTEM CHECK SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<20} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All checks passed! System is ready for speech denoising.")
        print("\nNext steps:")
        print("  python main.py --test              # Test with sample file")
        print("  python main.py --mode realtime     # Real-time processing")
        print("  python demo.py                     # Run interactive demo")
    else:
        print("❌ Some checks failed. Please resolve the issues above.")
        print("\nTroubleshooting:")
        if not results.get("Dependencies", True):
            print("  - Run: pip install -r requirements.txt")
        if not results.get("Audio System", True):
            print("  - Check microphone and speaker connections")
            print("  - Check audio permissions")
        
    return all_passed


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup and check Real-Time Speech Denoiser")
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    parser.add_argument('--check', action='store_true', help='Run system check only')
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
    
    if args.check or not args.install:
        run_system_check()


if __name__ == "__main__":
    main()