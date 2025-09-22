#!/usr/bin/env python3
"""
Simple test script to demonstrate DeepFilterNet2 noise reduction effectiveness.
"""

import torch
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path
from deepfilternet2 import create_model

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def test_with_synthetic_noise():
    """Test with synthetic speech + noise."""
    print("=== SYNTHETIC NOISE TEST ===")
    
    # Create synthetic speech
    sr = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Multi-harmonic speech-like signal
    speech = np.zeros_like(t)
    f0 = 150  # Fundamental frequency
    
    for harmonic in range(1, 8):
        freq = f0 * harmonic
        if freq < 4000:
            amplitude = 1.0 / harmonic
            speech += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add formants
    for formant in [800, 1200, 2600]:
        speech += 0.2 * np.sin(2 * np.pi * formant * t) * np.exp(-t * 1.5)
    
    speech = speech / np.max(np.abs(speech)) * 0.7
    
    # Add noise
    np.random.seed(42)
    noise_level = 0.4  # 40% noise
    noise = np.random.randn(len(speech)) * noise_level
    noisy_speech = speech + noise
    
    # Calculate input SNR
    input_snr = calculate_snr(speech, noise)
    
    # Process with model
    model = create_model()
    noisy_tensor = torch.from_numpy(noisy_speech).float()
    
    with torch.no_grad():
        enhanced = model(noisy_tensor).numpy()
    
    # Calculate output SNR
    residual_noise = enhanced - speech
    output_snr = calculate_snr(speech, residual_noise)
    snr_improvement = output_snr - input_snr
    
    # Calculate other metrics
    correlation = np.corrcoef(speech, enhanced)[0, 1]
    energy_reduction = (np.mean(noisy_speech**2) - np.mean(enhanced**2)) / np.mean(noisy_speech**2) * 100
    
    print(f"Input SNR: {input_snr:.1f} dB")
    print(f"Output SNR: {output_snr:.1f} dB")
    print(f"SNR Improvement: {snr_improvement:.1f} dB")
    print(f"Correlation with clean speech: {correlation:.3f}")
    print(f"Energy reduction: {energy_reduction:.1f}%")
    
    # Save files for listening
    sf.write('test_clean.wav', speech, sr)
    sf.write('test_noisy.wav', noisy_speech, sr)
    sf.write('test_enhanced.wav', enhanced, sr)
    
    return snr_improvement > 0

def test_with_file(filepath):
    """Test with actual audio file."""
    print(f"\n=== FILE TEST: {filepath} ===")
    
    # Load audio
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Use first channel if stereo
    
    # Process with model
    model = create_model()
    audio_tensor = torch.from_numpy(audio).float()
    
    with torch.no_grad():
        enhanced = model(audio_tensor).numpy()
    
    # Calculate metrics
    energy_reduction = (np.mean(audio**2) - np.mean(enhanced**2)) / np.mean(audio**2) * 100
    peak_reduction = (np.max(np.abs(audio)) - np.max(np.abs(enhanced))) / np.max(np.abs(audio)) * 100
    
    # Compare noisy vs quiet segments
    segment_length = min(sr, len(audio) // 4)
    segments = []
    
    for i in range(0, len(audio) - segment_length, segment_length):
        seg_orig = audio[i:i+segment_length]
        seg_enh = enhanced[i:i+segment_length]
        
        orig_rms = np.sqrt(np.mean(seg_orig**2))
        enh_rms = np.sqrt(np.mean(seg_enh**2))
        reduction = (orig_rms - enh_rms) / orig_rms * 100 if orig_rms > 0 else 0
        
        segments.append({
            'start': i/sr,
            'orig_rms': orig_rms,
            'enh_rms': enh_rms,
            'reduction': reduction
        })
    
    print(f"Overall energy reduction: {energy_reduction:.1f}%")
    print(f"Peak amplitude reduction: {peak_reduction:.1f}%")
    print(f"Number of segments analyzed: {len(segments)}")
    
    # Show per-segment analysis
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1} ({seg['start']:.1f}s): {seg['reduction']:.1f}% reduction")
    
    # Save enhanced file
    output_path = Path(filepath).stem + "_denoised.wav"
    sf.write(output_path, enhanced, sr)
    print(f"Enhanced audio saved: {output_path}")
    
    return energy_reduction > 5  # Consider successful if >5% energy reduction

def main():
    parser = argparse.ArgumentParser(description='Test DeepFilterNet2 denoising effectiveness')
    parser.add_argument('--file', help='Audio file to test')
    parser.add_argument('--synthetic', action='store_true', help='Run synthetic noise test')
    args = parser.parse_args()
    
    success_count = 0
    total_tests = 0
    
    if args.synthetic or not args.file:
        success_count += test_with_synthetic_noise()
        total_tests += 1
    
    if args.file:
        if Path(args.file).exists():
            success_count += test_with_file(args.file)
            total_tests += 1
        else:
            print(f"Error: File not found: {args.file}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {success_count/total_tests*100:.1f}%" if total_tests > 0 else "No tests run")
    
    if success_count > 0:
        print("✅ DeepFilterNet2 is working effectively for noise reduction!")
    else:
        print("❌ DeepFilterNet2 needs further optimization.")

if __name__ == "__main__":
    main()