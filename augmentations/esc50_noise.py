import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio


class NoiseClipPool:
    """
    A simple pool that collects .wav files from given directories and provides
    random segments aligned to a target waveform length.
    """

    def __init__(self, root_dirs: List[Path], resample_sr: int) -> None:
        self.resample_sr = int(resample_sr)
        self.wav_paths: List[Path] = []
        for rd in root_dirs:
            if rd is None:
                continue
            if not rd.exists():
                continue
            for p in rd.rglob('*.wav'):
                self.wav_paths.append(p)

    def is_empty(self) -> bool:
        return len(self.wav_paths) == 0

    def _load_resampled(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.resample_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_sr)(wav)
        return wav.squeeze(0)

    def get_random_segment(self, num_samples: int) -> torch.Tensor:
        if self.is_empty():
            return torch.zeros(num_samples)
        path = random.choice(self.wav_paths)
        noise = self._load_resampled(path)
        if noise.numel() == 0:
            return torch.zeros(num_samples)
        if noise.numel() >= num_samples:
            start = random.randint(0, noise.numel() - num_samples)
            return noise[start:start + num_samples]
        # If noise is shorter, tile and crop
        reps = (num_samples + noise.numel() - 1) // noise.numel()
        tiled = noise.repeat(reps)[:num_samples]
        return tiled


class ESC50NoiseAugmentor:
    """
    ESC-50 environmental noise augmentor with SNR control and category sampling.

    categories: List of dicts with keys {name, path}
    """

    def __init__(
        self,
        categories: Optional[List[Dict[str, str]]] = None,
        p_apply: float = 0.7,
        snr_db_choices: Optional[List[float]] = None,
        target_peak_dbfs: float = -1.0,
        resample_sr: int = 16000,
        eps: float = 1e-8,
        # ESC-50 meta mode
        audio_root: Optional[str] = None,
        meta_csv: Optional[str] = None,
        groups: Optional[List[Dict]] = None,
    ) -> None:
        self.p_apply = float(p_apply)
        self.snr_db_choices = snr_db_choices or [0.0, 5.0, 10.0, 20.0]
        self.target_peak_linear = 10.0 ** (target_peak_dbfs / 20.0)
        self.eps = float(eps)
        self.resample_sr = int(resample_sr)

        # Build per-category sources
        self.category_names: List[str] = []
        self.pools: List[NoiseClipPool] = []
        self.filelists: List[List[Path]] = []  # used in meta mode

        # Mode A: explicit directories
        if categories:
            for cat in categories:
                name = cat.get('name', 'unknown')
                path = Path(cat.get('path', ''))
                pool = NoiseClipPool([path], resample_sr=self.resample_sr)
                if not pool.is_empty():
                    self.category_names.append(name)
                    self.pools.append(pool)

        # Mode B: ESC-50 meta csv with groups of category names
        if audio_root and meta_csv and (groups or not self.pools):
            try:
                import csv
                audio_root_p = Path(audio_root)
                meta_p = Path(meta_csv)
                if meta_p.exists() and audio_root_p.exists():
                    # Load meta index: category -> [file paths]
                    cat_to_files: Dict[str, List[Path]] = {}
                    with open(meta_p, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            fname = row.get('filename') or row.get('file')
                            category = (row.get('category') or '').strip().lower()
                            if not fname or not category:
                                continue
                            full = audio_root_p / fname
                            cat_to_files.setdefault(category, []).append(full)

                    # Default groups if none provided
                    if not groups:
                        groups = [
                            {
                                'name': 'natural_soundscapes',
                                'categories': ['rain', 'sea_waves', 'crickets', 'chirping_birds', 'wind', 'frog', 'thunderstorm', 'water_drops', 'crackling_fire', 'insects']
                            },
                            {
                                'name': 'human_non_speech',
                                'categories': ['coughing', 'sneezing', 'breathing', 'laughing', 'footsteps', 'clapping']
                            },
                        ]

                    for grp in groups:
                        name = grp.get('name', 'group')
                        cats = [c.strip().lower() for c in grp.get('categories', [])]
                        files: List[Path] = []
                        for c in cats:
                            files.extend(cat_to_files.get(c, []))
                        if files:
                            self.category_names.append(name)
                            self.filelists.append(files)
            except Exception:
                pass

    def _compute_rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x ** 2) + self.eps)

    def _mix_with_snr(self, speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        speech_rms = self._compute_rms(speech)
        noise_rms = self._compute_rms(noise)
        gain = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
        mixed = speech + gain * noise

        # Peak limiting
        peak = mixed.abs().max()
        if peak > self.target_peak_linear:
            mixed = mixed * (self.target_peak_linear / (peak + self.eps))
        return mixed

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p_apply:
            return waveform
        if not self.pools and not self.filelists:
            return waveform

        if waveform.dim() == 1:
            if self.pools:
                pool = random.choice(self.pools)
                noise = pool.get_random_segment(waveform.numel())
            else:
                files = random.choice(self.filelists)
                noise = self._get_segment_from_files(files, waveform.numel())
            snr_db = float(self.snr_db_choices[int(torch.randint(0, len(self.snr_db_choices), (1,)).item())])
            return self._mix_with_snr(waveform, noise, snr_db)

        if waveform.dim() == 2:
            batch_out = []
            for b in range(waveform.size(0)):
                if self.pools:
                    pool = random.choice(self.pools)
                    noise = pool.get_random_segment(waveform.size(1))
                else:
                    files = random.choice(self.filelists)
                    noise = self._get_segment_from_files(files, waveform.size(1))
                snr_db = float(self.snr_db_choices[int(torch.randint(0, len(self.snr_db_choices), (1,)).item())])
                batch_out.append(self._mix_with_snr(waveform[b], noise, snr_db))
            return torch.stack(batch_out, dim=0)

        return waveform

    # Helper to extract a segment from a file list in meta mode
    def _get_segment_from_files(self, files: List[Path], num_samples: int) -> torch.Tensor:
        if not files:
            return torch.zeros(num_samples)
        for _ in range(3):
            p = random.choice(files)
            try:
                wav, sr = torchaudio.load(str(p))
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != self.resample_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_sr)(wav)
                wav = wav.squeeze(0)
                if wav.numel() >= num_samples:
                    start = random.randint(0, wav.numel() - num_samples)
                    return wav[start:start + num_samples]
                reps = (num_samples + wav.numel() - 1) // wav.numel()
                return wav.repeat(reps)[:num_samples]
            except Exception:
                continue
        return torch.zeros(num_samples)


