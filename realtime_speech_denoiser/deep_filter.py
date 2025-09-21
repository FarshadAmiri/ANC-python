import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ERBFilter(nn.Module):
    """ERB (Equivalent Rectangular Bandwidth) filterbank for frequency domain processing"""
    
    def __init__(self, sr: int = 48000, n_fft: int = 1024, n_erb: int = 32):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.n_erb = n_erb
        self.freq_bins = n_fft // 2 + 1
        
        # Create ERB filterbank
        erb_filters = self._create_erb_filters()
        self.register_buffer('erb_filters', erb_filters)
        
    def _hz_to_erb(self, hz):
        """Convert Hz to ERB scale"""
        return 24.7 * (4.37e-3 * hz + 1.0)
    
    def _erb_to_hz(self, erb):
        """Convert ERB scale to Hz"""
        return (erb / 24.7 - 1.0) / 4.37e-3
    
    def _create_erb_filters(self):
        """Create ERB filterbank matrix"""
        # Frequency range
        freq_max = self.sr / 2
        freq_bins = torch.linspace(0, freq_max, self.freq_bins)
        
        # ERB boundaries
        erb_min = self._hz_to_erb(50)  # Start from 50 Hz
        erb_max = self._hz_to_erb(freq_max)
        erb_centers = torch.linspace(erb_min, erb_max, self.n_erb)
        erb_freqs = torch.tensor([self._erb_to_hz(erb) for erb in erb_centers])
        
        # Create triangular filters
        filters = torch.zeros(self.n_erb, self.freq_bins)
        
        for i in range(self.n_erb):
            # Define triangular filter boundaries
            if i == 0:
                left_freq = 0
            else:
                left_freq = erb_freqs[i-1]
                
            center_freq = erb_freqs[i]
            
            if i == self.n_erb - 1:
                right_freq = freq_max
            else:
                right_freq = erb_freqs[i+1]
            
            # Create triangular filter
            left_slope = (freq_bins >= left_freq) & (freq_bins <= center_freq)
            right_slope = (freq_bins >= center_freq) & (freq_bins <= right_freq)
            
            if torch.any(left_slope):
                filters[i, left_slope] = (freq_bins[left_slope] - left_freq) / (center_freq - left_freq)
            if torch.any(right_slope):
                filters[i, right_slope] = (right_freq - freq_bins[right_slope]) / (right_freq - center_freq)
        
        # Normalize filters
        filters = F.normalize(filters, p=1, dim=1)
        return filters
    
    def forward(self, x):
        """Apply ERB filterbank to frequency domain signal"""
        # x shape: (batch, freq_bins, time)
        return torch.matmul(self.erb_filters, x)


class DeepFilter(nn.Module):
    """
    Simplified neural network inspired by DeepFilterNet for real-time speech enhancement
    """
    
    def __init__(
        self, 
        sr: int = 48000, 
        n_fft: int = 1024, 
        hop_length: int = 256,
        n_erb: int = 32,
        hidden_size: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_erb = n_erb
        self.freq_bins = n_fft // 2 + 1
        
        # ERB filterbank
        self.erb_filter = ERBFilter(sr, n_fft, n_erb)
        
        # Feature extraction
        self.feature_dim = n_erb + self.freq_bins  # ERB + linear frequency features
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Output layers for magnitude mask
        self.mask_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.freq_bins),
            nn.Sigmoid()  # Mask should be between 0 and 1
        )
        
        # Window for STFT
        self.register_buffer('window', torch.hann_window(n_fft))
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process audio waveform
        Args:
            waveform: (batch, time) or (time,)
        Returns:
            enhanced_waveform: same shape as input
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        batch_size = waveform.shape[0]
        
        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
        # Get magnitude and phase
        magnitude = torch.abs(stft)  # (batch, freq, time)
        phase = torch.angle(stft)
        
        # Feature extraction
        erb_features = self.erb_filter(magnitude)  # (batch, n_erb, time)
        
        # Concatenate ERB and linear frequency features
        features = torch.cat([
            erb_features,
            magnitude
        ], dim=1)  # (batch, feature_dim, time)
        
        # Transpose for LSTM: (batch, time, features)
        features = features.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # (batch, time, hidden)
        
        # Generate mask
        mask = self.mask_net(lstm_out)  # (batch, time, freq_bins)
        mask = mask.transpose(1, 2)  # (batch, freq_bins, time)
        
        # Apply mask to magnitude
        enhanced_magnitude = magnitude * mask
        
        # Add simple spectral subtraction for better initial performance
        enhanced_magnitude = self._apply_spectral_subtraction(magnitude, enhanced_magnitude)
        
        # Reconstruct complex spectrum
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        
        # ISTFT
        enhanced_waveform = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=waveform.shape[-1]
        )
        
        return enhanced_waveform
    
    def _apply_spectral_subtraction(self, original_magnitude, enhanced_magnitude):
        """Apply simple spectral subtraction for additional noise reduction"""
        # Estimate noise floor from quiet regions
        noise_floor = torch.quantile(original_magnitude, 0.1, dim=-1, keepdim=True)
        
        # Apply over-subtraction with floor
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Noise floor factor
        
        # Spectral subtraction
        subtracted = original_magnitude - alpha * noise_floor
        
        # Apply floor
        floor = beta * original_magnitude
        enhanced_ss = torch.maximum(subtracted, floor)
        
        # Blend with neural network output
        blend_factor = 0.3  # 30% spectral subtraction, 70% neural network
        final_magnitude = blend_factor * enhanced_ss + (1 - blend_factor) * enhanced_magnitude
        
        return final_magnitude


def create_model(
    sr: int = 48000,
    frame_size: int = 1024,
    hop_length: int = 256,
    device: str = 'cpu'
) -> DeepFilter:
    """Create and initialize a DeepFilter model"""
    
    model = DeepFilter(
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_length,
        n_erb=32,
        hidden_size=256,
        num_layers=3
    )
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Initialize with better weights for noise reduction
    _initialize_for_speech_enhancement(model)
    
    return model


def _initialize_for_speech_enhancement(model: DeepFilter):
    """Initialize model weights for better speech enhancement"""
    with torch.no_grad():
        # Initialize LSTM weights for better convergence
        for name, param in model.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                if 'bias_hh' in name or 'bias_ih' in name:
                    hidden_size = param.size(0) // 4
                    param[hidden_size:2*hidden_size].fill_(1.0)
        
        # Initialize mask network to output moderate suppression initially
        for module in model.mask_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)
        
        # Initialize the final layer to output around 0.7 (moderate suppression)
        if hasattr(model.mask_net[-2], 'bias'):
            nn.init.constant_(model.mask_net[-2].bias, 0.8)


def load_pretrained_weights(model: DeepFilter, weights_path: Optional[str] = None):
    """Load pretrained weights if available"""
    if weights_path and torch.cuda.is_available():
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            print(f"Could not load weights: {e}")
            print("Using randomly initialized weights")
    else:
        print("Using randomly initialized weights")
    
    return model