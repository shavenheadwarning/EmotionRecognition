# -*- coding: utf-8 -*-
"""
AST Feature Extractor for RAVDESS Emotion Classification
Generates mel spectrograms compatible with Audio Spectrogram Transformer
"""

import torch
import torchaudio
import torchaudio.compliance.kaldi
import numpy as np
from typing import Dict, Tuple, Optional
import logging


class ASTFeatureExtractor:
    """
    Feature extractor for AST model
    Generates mel spectrograms using Kaldi-compatible features
    """

    def __init__(self, config: Dict):
        """
        Initialize AST feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sample_rate = config['features']['sample_rate']
        self.max_length = config['features']['max_length']

        # AST-specific parameters
        self.num_mel_bins = 128  # Standard for AST
        self.target_length = int(self.max_length * 1000 / 10)  # Convert to frames (10ms frame shift)
        self.frame_shift = 10  # 10ms frame shift

        # Normalization statistics (can be computed from dataset)
        # These are approximate values, should be computed from RAVDESS for best results
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

        logging.info(f"AST Feature Extractor initialized:")
        logging.info(f"  Mel bins: {self.num_mel_bins}")
        logging.info(f"  Target length: {self.target_length} frames")
        logging.info(f"  Sample rate: {self.sample_rate}")
        logging.info(f"  Audio length: {self.max_length}s")

    def extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract FilterBank features using Kaldi-compatible method
        
        Args:
            waveform: Audio waveform tensor of shape (channels, samples) or (samples,)
            
        Returns:
            FilterBank features of shape (time_frames, mel_bins)
        """
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Remove DC offset
        waveform = waveform - waveform.mean()

        # Extract FilterBank features using Kaldi compliance
        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.num_mel_bins,
                dither=0.0,
                frame_shift=self.frame_shift
            )
        except Exception as e:
            logging.warning(f"Kaldi fbank failed, using torchaudio MelSpectrogram: {e}")
            # Fallback to torchaudio MelSpectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=400,  # 25ms window at 16kHz
                hop_length=int(self.sample_rate * self.frame_shift / 1000),  # 10ms hop
                n_mels=self.num_mel_bins,
                f_min=0,
                f_max=self.sample_rate // 2
            )
            fbank = mel_transform(waveform).squeeze(0).transpose(0, 1)
            # Convert to log scale
            fbank = torch.log(fbank + 1e-6)

        return fbank

    def pad_or_truncate(self, fbank: torch.Tensor) -> torch.Tensor:
        """
        Pad or truncate features to target length
        
        Args:
            fbank: FilterBank features of shape (time_frames, mel_bins)
            
        Returns:
            Padded/truncated features of shape (target_length, mel_bins)
        """
        n_frames = fbank.shape[0]

        if n_frames < self.target_length:
            # Pad with zeros
            padding = self.target_length - n_frames
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, padding), mode='constant', value=0)
        elif n_frames > self.target_length:
            # Truncate to target length
            fbank = fbank[:self.target_length, :]

        return fbank

    def normalize_features(self, fbank: torch.Tensor) -> torch.Tensor:
        """
        Normalize features using dataset statistics
        
        Args:
            fbank: FilterBank features
            
        Returns:
            Normalized features
        """
        # Normalize using dataset mean and std
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        return fbank

    def apply_spec_augment(self, fbank: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply SpecAugment for data augmentation during training
        
        Args:
            fbank: FilterBank features of shape (time_frames, mel_bins)
            training: Whether to apply augmentation
            
        Returns:
            Augmented features
        """
        if not training:
            return fbank

        # SpecAugment parameters
        freq_mask_param = 15  # Maximum frequency mask size
        time_mask_param = 35  # Maximum time mask size

        # Apply frequency masking
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        # Apply time masking
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param)

        # Add batch dimension for transforms
        fbank = fbank.transpose(0, 1).unsqueeze(0)  # (1, mel_bins, time_frames)

        # Apply masks
        if np.random.random() < 0.5:  # 50% chance to apply freq mask
            fbank = freq_masking(fbank)
        if np.random.random() < 0.5:  # 50% chance to apply time mask
            fbank = time_masking(fbank)

        # Remove batch dimension and transpose back
        fbank = fbank.squeeze(0).transpose(0, 1)  # (time_frames, mel_bins)

        return fbank

    def extract_ast_features(self, waveform: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Extract complete AST-compatible features
        
        Args:
            waveform: Audio waveform tensor
            training: Whether in training mode (for augmentation)
            
        Returns:
            AST-compatible mel spectrogram of shape (time_frames, mel_bins)
        """
        # Extract FilterBank features
        fbank = self.extract_fbank_features(waveform)

        # Pad or truncate to target length
        fbank = self.pad_or_truncate(fbank)

        # Apply SpecAugment during training
        if training:
            fbank = self.apply_spec_augment(fbank, training=True)

        # Normalize features
        fbank = self.normalize_features(fbank)

        return fbank

    def extract_features(self, waveform: torch.Tensor, feature_type: str) -> torch.Tensor:
        """
        Extract features based on specified type
        
        Args:
            waveform: Audio waveform tensor
            feature_type: Feature type ('ast' for AST features)
            
        Returns:
            Extracted features
        """
        if feature_type == 'ast':
            return self.extract_ast_features(waveform, training=True)
        else:
            raise ValueError(f"Unsupported feature type for AST: {feature_type}")


class ASTFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies AST feature extraction on-the-fly
    """

    def __init__(
        self,
        base_dataset,
        feature_extractor: ASTFeatureExtractor,
        feature_type: str,
        waveform_augmentor: Optional[object] = None,
    ):
        """
        Initialize AST feature dataset
        
        Args:
            base_dataset: Base dataset that returns (waveform, label)
            feature_extractor: ASTFeatureExtractor instance
            feature_type: Type of features to extract ('ast')
        """
        self.base_dataset = base_dataset
        self.feature_extractor = feature_extractor
        self.feature_type = feature_type
        self.waveform_augmentor = waveform_augmentor

        # Determine if dataset is in training mode
        self.training = hasattr(base_dataset, 'mode') and base_dataset.mode == 'train'

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        waveform, label = self.base_dataset[idx]
        if self.training and self.waveform_augmentor is not None:
            waveform = self.waveform_augmentor(waveform)

        # Extract AST features
        features = self.feature_extractor.extract_ast_features(waveform, training=self.training)

        return features, label

    def train(self):
        """Set dataset to training mode"""
        self.training = True
        return self

    def eval(self):
        """Set dataset to evaluation mode"""
        self.training = False
        return self


def create_ast_data_loaders(config: Dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders with AST feature extraction
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader with AST features
    """
    from data_processing.data_loader import create_data_loaders

    # Create base data loaders (returning raw waveforms)
    train_loader, val_loader = create_data_loaders(config)

    # Create AST feature extractor
    ast_feature_extractor = ASTFeatureExtractor(config)

    # Optional waveform-level augmentation for training set
    waveform_augmentor = None
    noise_cfg = config.get('noise_augmentation', {})
    if noise_cfg.get('enabled', False):
        try:
            aug_type = str(noise_cfg.get('type', 'white')).lower()
            if aug_type == 'white':
                from augmentations.noise import WhiteNoiseAugmentor
                waveform_augmentor = WhiteNoiseAugmentor(
                    p_apply=float(noise_cfg.get('p_apply', 0.7)),
                    snr_db_choices=list(noise_cfg.get('snr_db_choices', [20.0])),
                    target_peak_dbfs=float(noise_cfg.get('target_peak_dbfs', -1.0)),
                )
            elif aug_type == 'esc50':
                from augmentations.esc50_noise import ESC50NoiseAugmentor
                esc50_cfg = noise_cfg.get('esc50', {})
                waveform_augmentor = ESC50NoiseAugmentor(
                    categories=list(esc50_cfg.get('categories', [])) or None,
                    p_apply=float(noise_cfg.get('p_apply', 0.7)),
                    snr_db_choices=list(noise_cfg.get('snr_db_choices', [0.0, 5.0, 10.0, 20.0])),
                    target_peak_dbfs=float(noise_cfg.get('target_peak_dbfs', -1.0)),
                    resample_sr=int(esc50_cfg.get('resample_sr', config['features']['sample_rate'])),
                    audio_root=esc50_cfg.get('audio_root'),
                    meta_csv=esc50_cfg.get('meta_csv'),
                    groups=esc50_cfg.get('groups'),
                )
            else:
                logging.warning(f"Unknown noise augmentation type for AST: {aug_type}")
        except Exception as e:
            logging.warning(f"Failed to initialize noise augmentor for AST: {e}")

    # Wrap datasets with AST feature extraction
    train_ast_dataset = ASTFeatureDataset(
        train_loader.dataset,
        ast_feature_extractor,
        'ast',
        waveform_augmentor=waveform_augmentor,
    ).train()  # Set to training mode for SpecAugment

    val_ast_dataset = ASTFeatureDataset(
        val_loader.dataset,
        ast_feature_extractor,
        'ast',
        waveform_augmentor=None,
    ).eval()  # Set to eval mode (no augmentation)

    # Create new data loaders with AST features
    train_ast_loader = torch.utils.data.DataLoader(
        train_ast_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,  # Reduce workers for complex feature extraction
        pin_memory=True if config['device'] == 'cuda' else False
    )

    val_ast_loader = torch.utils.data.DataLoader(
        val_ast_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    return train_ast_loader, val_ast_loader


if __name__ == "__main__":
    # Test AST feature extraction
    from data_processing.data_loader import load_config

    config = load_config()
    ast_extractor = ASTFeatureExtractor(config)

    # Create dummy waveform (3 seconds at 22050 Hz)
    sample_rate = config['features']['sample_rate']
    duration = config['features']['max_length']
    dummy_waveform = torch.randn(int(sample_rate * duration))

    print(f"Input waveform shape: {dummy_waveform.shape}")

    # Test AST feature extraction
    ast_features = ast_extractor.extract_ast_features(dummy_waveform)
    print(f"AST features shape: {ast_features.shape}")

    expected_time_frames = int(duration * 1000 / 10)  # 10ms frame shift
    expected_shape = (expected_time_frames, 128)
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {ast_features.shape}")
    print(f"Shape match: {ast_features.shape == expected_shape}")

    # Test with batch
    batch_size = 4
    batch_waveforms = torch.randn(batch_size, int(sample_rate * duration))
    print(f"\nBatch test - Input shape: {batch_waveforms.shape}")

    batch_features = []
    for i in range(batch_size):
        features = ast_extractor.extract_ast_features(batch_waveforms[i])
        batch_features.append(features)

    batch_features = torch.stack(batch_features)
    print(f"Batch AST features shape: {batch_features.shape}")
    print(f"Expected batch shape: ({batch_size}, {expected_time_frames}, 128)")

    print("AST feature extraction test completed!")