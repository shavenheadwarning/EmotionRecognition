import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import torch
import torchaudio
import matplotlib.pyplot as plt
import sys

# Ensure project root is in sys.path so that sibling packages can be imported when
# running this file directly: `python visualization/vis_add_noise.py ...`.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local augmentors
from augmentations.noise import WhiteNoiseAugmentor
from augmentations.esc50_noise import ESC50NoiseAugmentor, NoiseClipPool


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_audio_mono(path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        sr = target_sr
    return wav, sr


def compute_rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2) + eps)


def mix_with_snr(
    speech: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    target_peak_dbfs: float = -1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    speech_rms = compute_rms(speech, eps=eps)
    noise_rms = compute_rms(noise, eps=eps)
    gain = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0) + eps))
    mixed = speech + gain * noise
    target_peak_linear = 10.0 ** (target_peak_dbfs / 20.0)
    peak = mixed.abs().max()
    if peak > target_peak_linear:
        mixed = mixed * (target_peak_linear / (peak + eps))
    return mixed


def sample_white_noise(length: int) -> torch.Tensor:
    return torch.randn(length)


def sample_esc50_noise(
    esc_aug: ESC50NoiseAugmentor,
    num_samples: int,
) -> Tuple[torch.Tensor, str]:
    # Prefer pools (explicit directories); otherwise sample from filelists (meta mode)
    if getattr(esc_aug, 'pools', None):
        pool_idx = int(torch.randint(0, len(esc_aug.pools), (1,)).item())
        pool: NoiseClipPool = esc_aug.pools[pool_idx]
        name = esc_aug.category_names[pool_idx] if esc_aug.category_names else 'unknown'
        noise = pool.get_random_segment(num_samples)
        return noise, name
    if getattr(esc_aug, 'filelists', None):
        files_idx = int(torch.randint(0, len(esc_aug.filelists), (1,)).item())
        files: List[Path] = esc_aug.filelists[files_idx]
        name = esc_aug.category_names[files_idx] if esc_aug.category_names else 'group'
        # Inline implementation similar to _get_segment_from_files
        for _ in range(3):
            p = files[int(torch.randint(0, len(files), (1,)).item())]
            try:
                wav, sr = torchaudio.load(str(p))
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != esc_aug.resample_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=esc_aug.resample_sr)(wav)
                wav = wav.squeeze(0)
                if wav.numel() >= num_samples:
                    start = int(torch.randint(0, wav.numel() - num_samples + 1, (1,)).item())
                    return wav[start:start + num_samples], name
                reps = (num_samples + wav.numel() - 1) // wav.numel()
                return wav.repeat(reps)[:num_samples], name
            except Exception:
                continue
        return torch.zeros(num_samples), name
    return torch.zeros(num_samples), 'none'


def make_mel_db(
    wav: torch.Tensor,
    sr: int,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
) -> torch.Tensor:
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
        normalized=False,
    )(wav.unsqueeze(0))  # (1, n_mels, time)
    mel_db = torchaudio.transforms.AmplitudeToDB(stype='power')(mel_spec).squeeze(0)
    return mel_db


def plot_wave_and_mel(
    speech: torch.Tensor,
    noise: torch.Tensor,
    mixed: torch.Tensor,
    sr: int,
    info_text: str,
    out_path: str,
) -> None:
    import numpy as np

    # Prepare data
    t = torch.arange(speech.numel(), dtype=torch.float32) / float(sr)
    mel_s = make_mel_db(speech, sr)
    mel_n = make_mel_db(noise, sr)
    mel_m = make_mel_db(mixed, sr)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), gridspec_kw={'hspace': 0.35, 'wspace': 0.15})

    # Waveforms
    axes[0, 0].plot(t.numpy(), speech.detach().cpu().numpy(), color='#1f77b4')
    axes[0, 0].set_title('原始语音 波形')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('幅度')

    axes[1, 0].plot(t.numpy(), noise.detach().cpu().numpy(), color='#ff7f0e')
    axes[1, 0].set_title('噪声 波形')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('幅度')

    axes[2, 0].plot(t.numpy(), mixed.detach().cpu().numpy(), color='#2ca02c')
    axes[2, 0].set_title('加噪后 波形')
    axes[2, 0].set_xlabel('时间 (s)')
    axes[2, 0].set_ylabel('幅度')

    # Mels (in dB)
    im0 = axes[0, 1].imshow(
        mel_s.detach().cpu().numpy(),
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='magma',
    )
    axes[0, 1].set_title('原始语音 Mel 频谱 (dB)')
    axes[0, 1].set_xlabel('帧')
    axes[0, 1].set_ylabel('Mel bin')

    im1 = axes[1, 1].imshow(
        mel_n.detach().cpu().numpy(),
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='magma',
    )
    axes[1, 1].set_title('噪声 Mel 频谱 (dB)')
    axes[1, 1].set_xlabel('帧')
    axes[1, 1].set_ylabel('Mel bin')

    im2 = axes[2, 1].imshow(
        mel_m.detach().cpu().numpy(),
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='magma',
    )
    axes[2, 1].set_title('加噪后 Mel 频谱 (dB)')
    axes[2, 1].set_xlabel('帧')
    axes[2, 1].set_ylabel('Mel bin')

    fig.suptitle(info_text)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='可视化语音加噪过程（白噪声 / ESC-50）')
    parser.add_argument('--audio', type=str, required=True, help='输入语音 wav 文件路径')
    parser.add_argument('--config', type=str, default=str(Path('config') / 'noise.yaml'), help='噪声配置 YAML')
    parser.add_argument('--type', type=str, choices=['white', 'esc50'], default=None, help='覆盖配置中的噪声类型')
    parser.add_argument('--out', type=str, default=None, help='输出可视化图片路径 (默认保存到 visualization/vis_results/)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--force-apply', action='store_true', help='无视 p，强制加噪')
    parser.add_argument('--snr', type=float, default=None, help='覆盖 SNR(dB)')
    args = parser.parse_args()

    set_seed(args.seed)

    cfg_all = load_yaml(args.config)
    na = cfg_all.get('noise_augmentation', {})
    enabled = bool(na.get('enabled', True))
    if not enabled:
        print('noise_augmentation.enabled = false，仍将进行可视化但不会应用噪声（除非 --force-apply）')

    aug_type = (args.type or na.get('type', 'white')).strip().lower()
    p_apply = float(na.get('p_apply', 0.7))
    snr_db_choices = na.get('snr_db_choices', [20.0 if aug_type == 'white' else 0.0])
    target_peak_dbfs = float(na.get('target_peak_dbfs', -1.0))

    esc_cfg = na.get('esc50', {}) if aug_type == 'esc50' else {}
    esc_sr = int(esc_cfg.get('resample_sr', 16000))

    # Load audio; for ESC-50 mixing, we resample speech to esc_sr
    target_sr = esc_sr if aug_type == 'esc50' else None
    speech, sr = load_audio_mono(args.audio, target_sr=target_sr)

    # Decide apply
    u = random.random()
    will_apply = args.force_apply or (u <= p_apply)

    # Prepare noise & snr
    picked_snr = float(args.snr if args.snr is not None else snr_db_choices[int(torch.randint(0, len(snr_db_choices), (1,)).item())])

    if aug_type == 'white':
        noise = sample_white_noise(speech.numel())
        noise_type_desc = 'white'
        noise_cat = 'white'
    else:
        esc_aug = ESC50NoiseAugmentor(
            categories=esc_cfg.get('categories', None),
            p_apply=p_apply,
            snr_db_choices=snr_db_choices,
            target_peak_dbfs=target_peak_dbfs,
            resample_sr=esc_sr,
            audio_root=esc_cfg.get('audio_root', None),
            meta_csv=esc_cfg.get('meta_csv', None),
            groups=esc_cfg.get('groups', None),
        )
        noise, noise_cat = sample_esc50_noise(esc_aug, speech.numel())
        noise_type_desc = f'esc50::{noise_cat}'

    if not will_apply:
        mixed = speech.clone()
        # Use zeros for visualization if not applied
        viz_noise = torch.zeros_like(speech)
    else:
        mixed = mix_with_snr(speech, noise, picked_snr, target_peak_dbfs=target_peak_dbfs)
        viz_noise = noise

    # Info text for title
    info_text = (
        f"type={aug_type}({noise_type_desc}) | p={p_apply:.2f}, u={u:.3f}, apply={'yes' if will_apply else 'no'} | "
        f"snr={picked_snr:.1f} dB | target_peak={target_peak_dbfs:.1f} dBFS | sr={sr} | seed={args.seed}"
    )

    # Output path
    if args.out is None:
        base = Path(args.audio).stem
        out_dir = Path('visualization') / 'vis_results'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{base}_{aug_type}_p{p_apply:.2f}_snr{picked_snr:.1f}_seed{args.seed}.png"
        out_path = str(out_dir / out_name)
    else:
        out_path = args.out

    plot_wave_and_mel(speech, viz_noise, mixed, sr, info_text, out_path)
    print(f"Saved visualization to: {out_path}")


if __name__ == '__main__':
    main()

