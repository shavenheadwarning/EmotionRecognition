import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import logging


RAW_EMO_TOKENS = [
    # common short codes
    'ang', 'hap', 'exc', 'sad', 'neu', 'fru', 'fea', 'sur', 'dis', 'oth', 'xxx',
    # full words (lowercase)
    'angry', 'happy', 'excited', 'sad', 'neutral', 'frustrated', 'fear', 'fearful', 'surprise', 'surprised', 'disgust', 'other'
]


def _lower_no_space(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


class IEMOCAPDataset(Dataset):
    """
    IEMOCAP utterance-level dataset with robust parsing of labels from dialog/EmoEvaluation.
    Splitting is controlled via config to avoid speaker/session leakage.
    """

    def __init__(self, config: Dict, mode: str = 'train') -> None:
        self.config = config
        self.mode = mode
        self.sample_rate = config['features']['sample_rate']
        self.max_length = config['features']['max_length']

        dataset_cfg = config['dataset']
        self.base_path = Path(dataset_cfg['data_path'])

        # Build mapping from utterance ID -> raw emotion label
        self.utt_to_raw_label = self._parse_emoeval_labels()

        # Map raw labels to target labels using config mapping
        self.raw_to_target = dataset_cfg.get('emotion_mapping', {})

        # Collect utterance wav files and labels
        all_utts, all_labels = self._collect_wav_and_labels()

        # Apply split strategy
        split_strategy = str(dataset_cfg.get('split_strategy', 'session')).lower()
        if split_strategy == 'session':
            train_sessions = set(dataset_cfg.get('train_sessions', []))
            val_sessions = set(dataset_cfg.get('val_sessions', []))
            selected = []
            for wav_path, label in zip(all_utts, all_labels):
                session_name = self._extract_session_from_path(wav_path)
                if self.mode == 'train' and session_name in train_sessions:
                    selected.append((wav_path, label))
                if self.mode == 'val' and session_name in val_sessions:
                    selected.append((wav_path, label))
        else:
            # Fallback: simple split by ratio (not recommended for final experiments)
            ratio = dataset_cfg.get('val_split', 0.2)
            split_idx = int(len(all_utts) * (1 - ratio))
            pairs = list(zip(all_utts, all_labels))
            if self.mode == 'train':
                selected = pairs[:split_idx]
            else:
                selected = pairs[split_idx:]

        self.file_paths = [p for p, _ in selected]
        self.labels = [int(_) for _, _ in selected]

        # Emotion labels dictionary (index -> name)
        self.emotion_labels = config['emotions']

        logging.info(f"IEMOCAP: loaded {len(self.file_paths)} {self.mode} samples (split={split_strategy}).")

    def _extract_session_from_path(self, wav_path: Path) -> str:
        # Return 'SessionX' from path if present
        for part in wav_path.parts:
            if part.lower().startswith('session'):
                return part if part.startswith('Session') else 'Session' + part[len('session'):]
        return 'SessionUnknown'

    def _parse_emoeval_labels(self) -> Dict[str, str]:
        """
        Parse labels from dialog/EmoEvaluation (or Emotions) files across all sessions.
        Returns: dict of utterance_id -> raw_label (lowercase, short code when possible)
        """
        utt_to_label: Dict[str, str] = {}

        # Possible directory name variations
        ses_globs = [
            self.base_path.glob('Session*/dialog/EmoEvaluation/**/*.txt'),
            self.base_path.glob('Session*/Session*/dialog/EmoEvaluation/**/*.txt'),
            self.base_path.glob('Session*/dialog/Emotions/**/*.txt'),
            self.base_path.glob('Session*/Session*/dialog/Emotions/**/*.txt'),
        ]

        def _normalize_emotion(token: str) -> str:
            t = token.lower()
            if t in ['angry', 'anger', 'ang']:
                return 'ang'
            if t in ['happy', 'hap', 'excited', 'exc']:
                # Keep 'exc' to allow merging later via config
                return 'exc' if 'exc' in t else 'hap'
            if t in ['sad', 'sadness']:
                return 'sad'
            if t in ['neutral', 'neu']:
                return 'neu'
            if t in ['frustrated', 'fru']:
                return 'fru'
            if t in ['fear', 'fearful', 'fea']:
                return 'fea'
            if t in ['surprise', 'surprised', 'sur']:
                return 'sur'
            if t in ['disgust', 'dis']:
                return 'dis'
            if t in ['other', 'oth', 'xxx']:
                return 'oth'
            return t

        emo_token_set = set(RAW_EMO_TOKENS)
        utt_regex = re.compile(r"(Ses\d{2}[FM]_(?:impro\d+|script\d+(?:_\d+)?)_[FM]\d{3})", re.IGNORECASE)

        for g in ses_globs:
            for eval_file in g:
                try:
                    with open(eval_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line_l = line.strip().lower()
                            if not line_l or not line_l.startswith('ses'):
                                continue
                            m = utt_regex.search(line.strip())
                            if not m:
                                continue
                            utt_id = m.group(1)
                            found = None
                            for tok in emo_token_set:
                                if tok in line_l:
                                    found = tok
                                    break
                            if found is None:
                                continue
                            utt_to_label[utt_id] = _normalize_emotion(found)
                except Exception as e:
                    logging.warning(f"Failed to parse {eval_file}: {e}")

        return utt_to_label

    def _collect_wav_and_labels(self) -> Tuple[List[Path], List[int]]:
        """
        Collect utterance-level WAV files and map them to label indices using config mapping.
        """
        wav_paths: List[Path] = []
        labels: List[int] = []

        # Search common utterance directories (case-insensitive variants)
        wav_globs = [
            self.base_path.glob('Session*/Sentences/wav/*/*.wav'),
            self.base_path.glob('Session*/Session*/Sentences/wav/*/*.wav'),
            self.base_path.glob('Session*/sentences/wav/*/*.wav'),
            self.base_path.glob('Session*/Session*/sentences/wav/*/*.wav'),
        ]

        # Target label indices mapping from config['emotions'] (keys are 1-based)
        # Build name -> index mapping with 0-based contiguous indices in ascending id order
        id_to_name = self.config['emotions']
        sorted_ids = sorted(id_to_name.keys())
        name_to_idx = {id_to_name[k]: idx for idx, k in enumerate(sorted_ids)}

        for g in wav_globs:
            for wav in g:
                utt_id = wav.stem  # e.g., Ses01F_impro01_F000
                raw_label = self.utt_to_raw_label.get(utt_id)
                if raw_label is None:
                    continue
                target_name = self.raw_to_target.get(raw_label)
                if target_name is None or target_name not in name_to_idx:
                    # Skip labels that are not mapped or not used
                    continue
                wav_paths.append(wav)
                labels.append(name_to_idx[target_name])

        if not wav_paths:
            logging.error(f"No utterance WAV files found or labels not parsed in {self.base_path}")

        # Debug: log label index coverage
        try:
            if labels:
                unique_labels = sorted(set(labels))
                logging.info(f"IEMOCAP: label indices present: {unique_labels}")
        except Exception:
            pass

        return wav_paths, labels

    def _load_audio(self, file_path: Path) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(str(file_path))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            target_length = int(self.sample_rate * self.max_length)
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            elif waveform.shape[1] < target_length:
                pad = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))

            if self.config['features']['normalize']:
                waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

            return waveform.squeeze(0)
        except Exception as e:
            logging.error(f"Error loading audio {file_path}: {e}")
            target_length = int(self.sample_rate * self.max_length)
            return torch.zeros(target_length)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        wav_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform = self._load_audio(wav_path)
        return waveform, label


def create_iemocap_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train/val data loaders for IEMOCAP."""
    train_dataset = IEMOCAPDataset(config, mode='train')
    val_dataset = IEMOCAPDataset(config, mode='val')

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


