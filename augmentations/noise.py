import torch
from typing import List, Optional


class WhiteNoiseAugmentor:
    """
    White noise augmentor with SNR control and peak limiting.

    Parameters are intentionally explicit to keep behavior clear.
    """

    def __init__(
        self,
        p_apply: float = 0.7,
        snr_db_choices: Optional[List[float]] = None,
        target_peak_dbfs: float = -1.0,
        eps: float = 1e-8,
    ) -> None:
        self.p_apply = float(p_apply)
        self.snr_db_choices = snr_db_choices or [20.0]
        self.target_peak_linear = 10.0 ** (target_peak_dbfs / 20.0)
        self.eps = float(eps)

    def _compute_rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x ** 2) + self.eps)

    def _mix_with_snr(self, speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        speech_rms = self._compute_rms(speech)
        noise_rms = self._compute_rms(noise)
        gain = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
        mixed = speech + gain * noise

        # Peak limiting to avoid clipping
        peak = mixed.abs().max()
        if peak > self.target_peak_linear:
            mixed = mixed * (self.target_peak_linear / (peak + self.eps))
        return mixed

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a single waveform (1D tensor) or batch (2D: batch, time).
        """
        if torch.rand(1).item() > self.p_apply:
            return waveform

        if waveform.dim() == 1:
            noise = torch.randn_like(waveform)
            snr_db = float(self.snr_db_choices[int(torch.randint(0, len(self.snr_db_choices), (1,)).item())])
            return self._mix_with_snr(waveform, noise, snr_db)

        if waveform.dim() == 2:
            batch_size = waveform.size(0)
            mixed_list = []
            for b in range(batch_size):
                w = waveform[b]
                noise = torch.randn_like(w)
                snr_db = float(self.snr_db_choices[int(torch.randint(0, len(self.snr_db_choices), (1,)).item())])
                mixed_list.append(self._mix_with_snr(w, noise, snr_db))
            return torch.stack(mixed_list, dim=0)

        # For higher dims, do nothing to avoid shape ambiguity
        return waveform


