import torch
from typing import Optional


class WaveformAugmentorWrapper:
    """
    Wrap a base dataset to apply waveform-level augmentation on-the-fly.
    Augmentation is applied only when mode == 'train'.
    """

    def __init__(self, base_dataset, augmentor, mode: str = 'train') -> None:
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        waveform, label = self.base_dataset[idx]
        if self.mode == 'train' and self.augmentor is not None:
            if isinstance(waveform, torch.Tensor):
                waveform = self.augmentor(waveform)
        return waveform, label


