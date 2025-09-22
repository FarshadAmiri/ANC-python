#!/usr/bin/env python3
"""
DeepFilterNet2 Simple Demo
Demonstrates effective noise reduction capabilities.
"""

import torch
import numpy as np
import soundfile as sf
from deepfilternet2 import create_model
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def create_demo_audio():
    """Create demonstration audio with speech + noise."""
    print("Creating demonstration audio...")
    
    sr = 48000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create synthetic speech-like signal
    speech = np.zeros_like(t)
    
    # Multiple voices/tones
    fundamentals = [120, 180, 250]  # Different voice pitches
    
    for i, f0 in enumerate(fundamentals):
        # Each voice speaks for part of the duration
        start_time = i * duration / len(fundamentals)
        end_time = (i + 1) * duration / len(fundamentals)
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        
        # Create harmonic series for this voice
        for harmonic in range(1, 6):
            freq = f0 * harmonic
            if freq < 4000:
                amplitude = 1.0 / harmonic * (0.5 + 0.5 * np.sin(2 * np.pi * t * 2))  # Modulation
                voice_segment = amplitude * np.sin(2 * np.pi * freq * t)
                speech[start_idx:end_idx] += voice_segment[start_idx:end_idx]
    
    # Add formant resonances (vowel-like sounds)
    formants = [800, 1200, 2600]
    for formant in formants:
        formant_energy = 0.3 * np.sin(2 * np.pi * formant * t) * np.exp(-np.abs(t - duration/2))
        speech += formant_energy
    
    # Normalize speech
    speech = speech / np.max(np.abs(speech)) * 0.7
    
    # Add realistic noise (combination of white and colored noise)
    np.random.seed(42)
    
    # White noise
    white_noise = np.random.randn(len(t)) * 0.3
    
    # Low-frequency rumble
    rumble = np.random.randn(len(t)) * 0.2
    rumble = np.convolve(rumble, np.ones(100)/100, mode='same')  # Low-pass filter
    
    # High-frequency hiss
    hiss = np.random.randn(len(t)) * 0.15
    b, a = [1, -0.95], [1]  # High-pass filter coefficients (simple)
    for _ in range(3):
        hiss = np.convolve(hiss, b, mode='same')
    
    # Combine all noise sources
    total_noise = white_noise + rumble + hiss
    
    # Create noisy speech
    noisy_speech = speech + total_noise
    
    return speech, noisy_speech, total_noise, sr

def calculate_metrics(clean, noisy, enhanced):
    """Calculate audio quality metrics."""
    # SNR calculations
    noise = noisy - clean
    enhanced_noise = enhanced - clean
    
    original_snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    enhanced_snr = 10 * np.log10(np.mean(clean**2) / np.mean(enhanced_noise**2))
    snr_improvement = enhanced_snr - original_snr
    
    # Correlation
    correlation = np.corrcoef(clean, enhanced)[0, 1]
    
    # Energy reduction
    energy_reduction = (np.mean(noisy**2) - np.mean(enhanced**2)) / np.mean(noisy**2) * 100
    
    return {
        'original_snr': original_snr,
        'enhanced_snr': enhanced_snr,
        'snr_improvement': snr_improvement,
        'correlation': correlation,
        'energy_reduction': energy_reduction
    }

def main():
    print("🎙️  DeepFilterNet2 Noise Reduction Demo")
    print("=" * 50)
    
    # Create demonstration audio
    clean_speech, noisy_speech, noise, sr = create_demo_audio()
    
    print(f"✅ Created demo audio:")
    print(f"   Duration: {len(clean_speech)/sr:.1f} seconds")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Clean speech RMS: {np.sqrt(np.mean(clean_speech**2)):.4f}")
    print(f"   Noise RMS: {np.sqrt(np.mean(noise**2)):.4f}")
    
    # Save input files
    sf.write('demo_clean.wav', clean_speech, sr)
    sf.write('demo_noisy.wav', noisy_speech, sr)
    
    # Load and apply DeepFilterNet2
    print("\n🧠 Loading DeepFilterNet2 model...")
    model = create_model()
    
    print("🔄 Processing audio...")
    noisy_tensor = torch.from_numpy(noisy_speech).float()
    
    with torch.no_grad():
        enhanced_speech = model(noisy_tensor).numpy()
    
    # Save enhanced audio
    sf.write('demo_enhanced.wav', enhanced_speech, sr)
    
    # Calculate and display metrics
    print("\n📊 Results:")
    metrics = calculate_metrics(clean_speech, noisy_speech, enhanced_speech)
    
    print(f"   Original SNR:     {metrics['original_snr']:6.1f} dB")
    print(f"   Enhanced SNR:     {metrics['enhanced_snr']:6.1f} dB")
    print(f"   SNR Improvement:  {metrics['snr_improvement']:6.1f} dB  {'✅' if metrics['snr_improvement'] > 0 else '❌'}")
    print(f"   Correlation:      {metrics['correlation']:6.3f}      {'✅' if metrics['correlation'] > 0.7 else '❌'}")
    print(f"   Energy Reduction: {metrics['energy_reduction']:6.1f}%     {'✅' if metrics['energy_reduction'] > 5 else '❌'}")
    
    # Overall assessment
    print(f"\n🎯 Overall Performance:")
    if (metrics['snr_improvement'] > 2 and 
        metrics['correlation'] > 0.7 and 
        metrics['energy_reduction'] > 10):
        print("   🌟 EXCELLENT - Significant noise reduction with good speech preservation")
    elif (metrics['snr_improvement'] > 1 and 
          metrics['correlation'] > 0.6 and 
          metrics['energy_reduction'] > 5):
        print("   ✅ GOOD - Effective noise reduction")
    elif metrics['snr_improvement'] > 0:
        print("   ⚠️  FAIR - Some noise reduction achieved")
    else:
        print("   ❌ POOR - Needs improvement")
    
    print(f"\n📁 Files created:")
    print(f"   demo_clean.wav    - Original clean speech")
    print(f"   demo_noisy.wav    - Noisy input")
    print(f"   demo_enhanced.wav - DeepFilterNet2 output")
    print(f"\n🎧 Listen to the files to hear the difference!")
    
    # Optional: Create simple visualization
    if HAS_MATPLOTLIB:
        try:
            print(f"\n📈 Creating visualization...")
            
            # Time domain comparison
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            time_axis = np.linspace(0, len(clean_speech)/sr, len(clean_speech))
            
            axes[0].plot(time_axis, clean_speech, alpha=0.8, color='green')
            axes[0].set_title('Clean Speech')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_axis, noisy_speech, alpha=0.8, color='red')
            axes[1].set_title('Noisy Speech')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(time_axis, enhanced_speech, alpha=0.8, color='blue')
            axes[2].set_title('Enhanced Speech (DeepFilterNet2)')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_ylabel('Amplitude')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('demo_waveforms.png', dpi=150, bbox_inches='tight')
            print(f"   demo_waveforms.png - Waveform comparison")
            
            # Spectrum comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            freq_clean = np.abs(np.fft.fft(clean_speech))[:len(clean_speech)//2]
            freq_noisy = np.abs(np.fft.fft(noisy_speech))[:len(noisy_speech)//2]
            freq_enhanced = np.abs(np.fft.fft(enhanced_speech))[:len(enhanced_speech)//2]
            freqs = np.linspace(0, sr/2, len(freq_clean))
            
            axes[0].semilogx(freqs, 20*np.log10(freq_clean + 1e-10), color='green')
            axes[0].set_title('Clean Spectrum')
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Magnitude (dB)')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].semilogx(freqs, 20*np.log10(freq_noisy + 1e-10), color='red')
            axes[1].set_title('Noisy Spectrum')
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Magnitude (dB)')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].semilogx(freqs, 20*np.log10(freq_enhanced + 1e-10), color='blue')
            axes[2].set_title('Enhanced Spectrum')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude (dB)')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('demo_spectra.png', dpi=150, bbox_inches='tight')
            print(f"   demo_spectra.png - Frequency spectrum comparison")
            
        except Exception as e:
            print(f"   (Visualization error: {e})")
    else:
        print(f"   (Install matplotlib to generate visualizations: pip install matplotlib)")

if __name__ == "__main__":
    main()