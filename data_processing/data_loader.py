import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path
from typing import Tuple, List, Dict


class RAVDESSDataset(Dataset):
    """
    RAVDESS Dataset Loader
    
    RAVDESS filename format: Modality-Vocal channel-Emotion-Emotional intensity-Statement-Repetition-Actor.wav
    Example: 03-01-05-02-01-01-12.wav
    - 03: Modality (01=full-AV, 02=video-only, 03=audio-only)
    - 01: Vocal channel (01=speech, 02=song)  
    - 05: Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
    - 02: Emotional intensity (01=normal, 02=strong)
    - 01: Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
    - 01: Repetition (01=1st repetition, 02=2nd repetition)
    - 12: Actor (01 to 24)
    """

    def __init__(self, config: Dict, mode: str = 'train'):
        """
        Initialize RAVDESS dataset
        
        Args:
            config: Configuration dictionary
            mode: 'train' or 'val'
        """
        self.config = config
        self.mode = mode
        self.sample_rate = config['features']['sample_rate']
        self.max_length = config['features']['max_length']

        # Emotion mapping (RAVDESS uses 1-8, we convert to 0-7 for model)
        self.emotion_map = {i: i-1 for i in range(1, 9)}
        self.emotion_labels = config['emotions']

        # Load file paths and labels
        self.file_paths, self.labels = self._load_data()

        logging.info(f"Loaded {len(self.file_paths)} {mode} samples")
        logging.info(f"Emotion distribution: {self._get_emotion_distribution()}")

    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load file paths and extract emotion labels"""
        file_paths = []
        labels = []

        # Construct data path
        base_path = Path(self.config['dataset']['data_path'])

        # Choose speech or song data
        if self.config['dataset']['use_speech']:
            data_path = base_path / self.config['dataset']['speech_path']
        if self.config['dataset']['use_song']:
            song_path = base_path / self.config['dataset']['song_path']
            if not self.config['dataset']['use_speech']:
                data_path = song_path

        # Collect all wav files
        all_files = []
        all_labels = []

        for actor_dir in sorted(data_path.glob('Actor_*')):
            if actor_dir.is_dir():
                for wav_file in actor_dir.glob('*.wav'):
                    emotion_label = self._extract_emotion_from_filename(wav_file.name)
                    if emotion_label is not None:
                        all_files.append(str(wav_file))
                        all_labels.append(emotion_label)

        # Split data
        if len(all_files) == 0:
            raise ValueError(f"No audio files found in {data_path}")

        # Use stratified split to maintain emotion distribution
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, all_labels,
            test_size=self.config['dataset']['val_split'],
            stratify=all_labels,
            random_state=self.config['dataset']['random_seed']
        )

        if self.mode == 'train':
            return train_files, train_labels
        else:
            return val_files, val_labels

    def _extract_emotion_from_filename(self, filename: str) -> int:
        """
        Extract emotion label from RAVDESS filename
        
        Args:
            filename: RAVDESS filename (e.g., '03-01-05-02-01-01-12.wav')
            
        Returns:
            Emotion label (0-7) or None if invalid
        """
        try:
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = int(parts[2])  # Third element is emotion
                if 1 <= emotion_code <= 8:
                    return self.emotion_map[emotion_code]  # Convert to 0-7
        except (ValueError, IndexError):
            logging.warning(f"Invalid filename format: {filename}")
        return None

    def _get_emotion_distribution(self) -> Dict[str, int]:
        """Get distribution of emotions in the dataset"""
        distribution = {}
        for label in self.labels:
            emotion_name = self.emotion_labels[label + 1]  # Convert back to 1-8 for lookup
            distribution[emotion_name] = distribution.get(emotion_name, 0) + 1
        return distribution

    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Pad or trim to fixed length
            target_length = int(self.sample_rate * self.max_length)
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            elif waveform.shape[1] < target_length:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Normalize
            if self.config['features']['normalize']:
                waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

            return waveform.squeeze(0)  # Remove channel dimension

        except Exception as e:
            logging.error(f"Error loading audio file {file_path}: {e}")
            # Return zero tensor as fallback
            target_length = int(self.sample_rate * self.max_length)
            return torch.zeros(target_length)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get audio waveform and emotion label
        
        Returns:
            waveform: Audio tensor of shape (seq_length,)
            label: Emotion label (0-7)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform = self._load_audio(file_path)

        return waveform, int(label)


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders dispatching by dataset name."""
    dataset_name = config['dataset']['name'].lower()
    if dataset_name == 'ravdess':
        train_dataset = RAVDESSDataset(config, mode='train')
        val_dataset = RAVDESSDataset(config, mode='val')
    elif dataset_name == 'iemocap':
        from .iemocap_loader import create_iemocap_data_loaders
        return create_iemocap_data_loaders(config)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']['name']}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False,
    )
    return train_loader, val_loader


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file and merge optional noise augmentation config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Merge noise augmentation config if exists
    try:
        noise_cfg_path = Path('config') / 'noise.yaml'
        if noise_cfg_path.exists():
            with open(noise_cfg_path, 'r', encoding='utf-8') as nf:
                noise_cfg = yaml.safe_load(nf) or {}
            if isinstance(noise_cfg, dict):
                config.update(noise_cfg)
    except Exception as e:
        logging.warning(f"Failed to load noise augmentation config: {e}")

    return config


if __name__ == "__main__":
    # Test data loader
    config = load_config()
    train_loader, val_loader = create_data_loaders(config)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: waveforms shape = {waveforms.shape}, labels shape = {labels.shape}")
        print(f"Labels: {labels}")
        break