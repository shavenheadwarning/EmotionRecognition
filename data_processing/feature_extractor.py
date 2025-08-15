import torch
import torchaudio
import numpy as np
from typing import Dict, Tuple, Optional
import logging


class FeatureExtractor:
    """
    Feature extraction for audio signals
    Supports MFCC and Mel Spectrogram features
    """

    def __init__(self, config: Dict):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sample_rate = config['features']['sample_rate']

        # Initialize MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=config['features']['mfcc']['n_mfcc'],
            melkwargs={
                'n_fft': config['features']['mfcc']['n_fft'],
                'hop_length': config['features']['mfcc']['hop_length'],
                'n_mels': config['features']['mfcc']['n_mels'],
                'center': False
            }
        )

        # Initialize Mel Spectrogram transform
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=config['features']['mel_spectrogram']['n_fft'],
            hop_length=config['features']['mel_spectrogram']['hop_length'],
            win_length=config['features']['mel_spectrogram']['win_length'],
            n_mels=config['features']['mel_spectrogram']['n_mels'],
            center=False
        )

        # Log mel spectrogram transform (for better neural network training)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from waveform
        
        Args:
            waveform: Audio waveform tensor of shape (batch_size, seq_length) or (seq_length,)
            
        Returns:
            MFCC features of shape (batch_size, 2*n_mfcc) or (2*n_mfcc,)
        """
        squeeze_output = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            squeeze_output = True

        # Extract MFCC
        mfcc = self.mfcc_transform(waveform)

        # Statistical features: mean and std across time dimension
        mfcc_mean = torch.mean(mfcc, dim=2)  # Shape: (batch_size, n_mfcc)
        mfcc_std = torch.std(mfcc, dim=2)    # Shape: (batch_size, n_mfcc)

        # Concatenate mean and std
        mfcc_features = torch.cat([mfcc_mean, mfcc_std], dim=1)  # Shape: (batch_size, 2*n_mfcc)

        if squeeze_output:
            mfcc_features = mfcc_features.squeeze(0)  # Remove batch dimension for single sample

        return mfcc_features

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Mel Spectrogram features from waveform
        
        Args:
            waveform: Audio waveform tensor of shape (batch_size, seq_length) or (seq_length,)
            
        Returns:
            Mel spectrogram of shape (batch_size, 1, n_mels, time_frames) or (1, n_mels, time_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        # Extract mel spectrogram
        mel_spec = self.mel_spectrogram_transform(waveform)

        # Convert to log scale (dB)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # Normalize to [0, 1] range
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

        # Add channel dimension for CNN (batch_size, channels=1, n_mels, time_frames)
        log_mel_spec = log_mel_spec.unsqueeze(1)

        if squeeze_output:
            log_mel_spec = log_mel_spec.squeeze(0)  # Remove batch dimension

        return log_mel_spec

    def extract_features(self, waveform: torch.Tensor, feature_type: str) -> torch.Tensor:
        """
        Extract features based on specified type
        
        Args:
            waveform: Audio waveform tensor
            feature_type: 'mfcc' or 'mel_spectrogram'
            
        Returns:
            Extracted features
        """
        if feature_type == 'mfcc':
            return self.extract_mfcc(waveform)
        elif feature_type == 'mel_spectrogram':
            return self.extract_mel_spectrogram(waveform)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")


class FeatureDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies feature extraction on-the-fly
    """

    def __init__(
        self,
        base_dataset,
        feature_extractor: "FeatureExtractor",
        feature_type: str,
        waveform_augmentor: Optional[object] = None,
    ):
        """
        Initialize feature dataset
        
        Args:
            base_dataset: Base dataset that returns (waveform, label)
            feature_extractor: FeatureExtractor instance
            feature_type: Type of features to extract ('mfcc' or 'mel_spectrogram')
        """
        self.base_dataset = base_dataset
        self.feature_extractor = feature_extractor
        self.feature_type = feature_type
        self.waveform_augmentor = waveform_augmentor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        waveform, label = self.base_dataset[idx]
        if self.waveform_augmentor is not None:
            waveform = self.waveform_augmentor(waveform)
        features = self.feature_extractor.extract_features(waveform, self.feature_type)
        return features, label


def create_feature_data_loaders(config: Dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders with feature extraction
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader with features
    """
    from data_processing.data_loader import create_data_loaders

    # Get model configuration to determine feature type
    model_name = config['training']['model_name']
    model_config = config['models'][model_name]
    feature_type = model_config['feature_type']

    # For AST features, use dedicated AST data loaders
    if feature_type == 'ast':
        from data_processing.ast_feature_extractor import create_ast_data_loaders
        return create_ast_data_loaders(config)

    # Create base data loaders (returning raw waveforms)
    train_loader, val_loader = create_data_loaders(config)


    feature_extractor = FeatureExtractor(config)

    # Optional waveform-level augmentation (train only)
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
                logging.info(f"Using white noise augmentor p_apply: {noise_cfg.get('p_apply', 0.7)}")
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
                logging.info(f"Using ESC-50 noise augmentor p_apply: {noise_cfg.get('p_apply', 0.7)}")
            else:
                logging.warning(f"Unknown noise augmentation type: {aug_type}")
        except Exception as e:
            logging.warning(f"Failed to initialize noise augmentor: {e}")
            waveform_augmentor = None

    # Wrap datasets with feature extraction
    train_feature_dataset = FeatureDataset(
        train_loader.dataset,
        feature_extractor,
        feature_type,
        waveform_augmentor=waveform_augmentor,
    )
    val_feature_dataset = FeatureDataset(
        val_loader.dataset,
        feature_extractor,
        feature_type,
        waveform_augmentor=None,
    )

    # Create new data loaders with features
    train_feature_loader = torch.utils.data.DataLoader(
        train_feature_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    val_feature_loader = torch.utils.data.DataLoader(
        val_feature_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    return train_feature_loader, val_feature_loader


if __name__ == "__main__":
    # Test feature extraction
    from data_processing.data_loader import load_config

    config = load_config()
    feature_extractor = FeatureExtractor(config)

    # Create dummy waveform
    sample_rate = config['features']['sample_rate']
    duration = config['features']['max_length']
    dummy_waveform = torch.randn(int(sample_rate * duration))

    # Test MFCC extraction
    mfcc_features = feature_extractor.extract_mfcc(dummy_waveform)
    print(f"MFCC features shape: {mfcc_features.shape}")

    # Test Mel Spectrogram extraction
    mel_features = feature_extractor.extract_mel_spectrogram(dummy_waveform)
    print(f"Mel Spectrogram features shape: {mel_features.shape}")

    # Test with batch
    batch_waveform = torch.randn(4, int(sample_rate * duration))
    batch_mfcc = feature_extractor.extract_mfcc(batch_waveform)
    batch_mel = feature_extractor.extract_mel_spectrogram(batch_waveform)
    print(f"Batch MFCC shape: {batch_mfcc.shape}")
    print(f"Batch Mel Spectrogram shape: {batch_mel.shape}")