import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


def find_audio_files(root: Path, patterns: Tuple[str, ...] = (".wav",)) -> List[Path]:
    """递归检索音频文件。

    Args:
        root: 数据集根目录。
        patterns: 允许的文件扩展名集合。

    Returns:
        文件路径列表（按名称排序）。
    """
    if not root.exists():
        return []
    files: List[Path] = []
    for suffix in patterns:
        files.extend(root.rglob(f"*{suffix}"))
    files = [p for p in files if p.is_file()]
    files.sort()
    return files


def ensure_output_dir() -> Path:
    out_dir = Path("visualization") / "vis_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_audio(audio_path: Path, target_sr: Optional[int], duration: Optional[float]) -> Tuple[np.ndarray, int]:
    """加载音频为单声道浮点波形。

    Args:
        audio_path: 音频路径。
        target_sr: 目标采样率；None 表示保留原采样率。
        duration: 限制最大秒数；None 表示完整音频。

    Returns:
        (waveform, sr)
    """
    y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True, duration=duration)
    return y, sr


def plot_waveform_and_mfcc(
    y: np.ndarray,
    sr: int,
    title: str,
    output_path: Path,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    dpi: int = 600,
) -> None:
    """为给定音频绘制波形与 MFCC 并保存。

    Args:
        y: 波形。
        sr: 采样率。
        title: 图标题（包含数据集与文件名信息）。
        output_path: 输出 PNG 文件路径。
        n_mfcc: MFCC 维度。
        n_fft: STFT 窗长。
        hop_length: 帧移。
        dpi: 保存分辨率。
    """
    # 时间轴（秒）
    times = np.arange(len(y)) / float(sr)

    # 计算 MFCC（对数梅尔谱上做 DCT）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # 波形
    axes[0].plot(times, y, color="#1f77b4", linewidth=0.8)
    axes[0].set_title(f"{title} — Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", linewidth=0.3, alpha=0.6)

    # MFCC 热力图
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title(f"{title} — MFCC ({n_mfcc})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("MFCC Coefficients")
    cbar = fig.colorbar(img, ax=axes[1], format="%+2.0f")
    cbar.set_label("dB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_waveform_and_mel(
    y: np.ndarray,
    sr: int,
    title: str,
    output_path: Path,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_stft: bool = False,
    dpi: int = 600,
) -> None:
    """为给定音频绘制波形与梅尔谱图（可选加入线性 STFT 作为中间过程）。

    Args:
        y: 波形。
        sr: 采样率。
        title: 图标题。
        output_path: 输出 PNG 文件路径。
        n_mels: 梅尔滤波器组数量。
        n_fft: STFT 窗长。
        hop_length: 帧移。
        include_stft: 是否额外显示线性频率尺度的 STFT 频谱（对数频率坐标）。
        dpi: 保存分辨率。
    """
    # 计算中间与终端特征
    S_stft = None
    if include_stft:
        S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2  # 功率谱
        S_stft = librosa.power_to_db(S, ref=np.max)

    S_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    n_rows = 3 if include_stft else 2
    figsize = (10, 8) if include_stft else (10, 6)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, constrained_layout=True)
    if n_rows == 2:
        axes = np.array(axes)

    # 波形
    times = np.arange(len(y)) / float(sr)
    axes[0].plot(times, y, color="#1f77b4", linewidth=0.8)
    axes[0].set_title(f"{title} — Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", linewidth=0.3, alpha=0.6)

    row_idx = 1
    if include_stft and S_stft is not None:
        img_stft = librosa.display.specshow(
            S_stft,
            x_axis="time",
            y_axis="log",
            sr=sr,
            hop_length=hop_length,
            ax=axes[row_idx],
            cmap="viridis",
        )
        axes[row_idx].set_title("STFT Spectrogram (log-freq)")
        axes[row_idx].set_xlabel("Time (s)")
        axes[row_idx].set_ylabel("Frequency (Hz)")
        cbar = fig.colorbar(img_stft, ax=axes[row_idx])
        cbar.set_label("dB")
        row_idx += 1

    # 梅尔谱图
    img_mel = librosa.display.specshow(
        S_mel_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        ax=axes[row_idx],
        cmap="magma",
    )
    axes[row_idx].set_title(f"{title} — Mel-Spectrogram ({n_mels})")
    axes[row_idx].set_xlabel("Time (s)")
    axes[row_idx].set_ylabel("Mel bands")
    cbar = fig.colorbar(img_mel, ax=axes[row_idx])
    cbar.set_label("dB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def visualize_dataset(
    dataset_name: str,
    dataset_root: Path,
    out_dir: Path,
    max_files: int,
    target_sr: Optional[int],
    duration: Optional[float],
    n_mfcc: int,
    plot: str = "mel",
    include_stft: bool = False,
) -> int:
    """从指定数据集采样若干音频并保存可视化结果。

    Returns:
        实际处理的文件数。
    """
    audio_files = find_audio_files(dataset_root)
    if not audio_files:
        return 0

    count = 0
    for audio_path in audio_files[: max(0, max_files)]:
        try:
            y, sr = load_audio(audio_path, target_sr=target_sr, duration=duration)
            if y.size == 0:
                continue
            safe_name = audio_path.stem.replace(" ", "_")
            title = f"{dataset_name}: {audio_path.stem}"
            if plot in {"mel", "both"}:
                out_path_mel = out_dir / f"{dataset_name}_{safe_name}_mel.png"
                plot_waveform_and_mel(
                    y,
                    sr,
                    title,
                    out_path_mel,
                    n_mels=128,
                    include_stft=include_stft,
                )
            if plot in {"mfcc", "both"}:
                out_path_mfcc = out_dir / f"{dataset_name}_{safe_name}_mfcc.png"
                plot_waveform_and_mfcc(y, sr, title, out_path_mfcc, n_mfcc=n_mfcc)
            count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 跳过文件: {audio_path}，原因: {exc}")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 IEMOCAP 和 RAVDESS 中抽样音频，绘制波形与梅尔谱图/ MFCC 并以 600dpi 保存"
        )
    )
    parser.add_argument(
        "--iemocap",
        type=str,
        default=str(Path("data") / "IEMOCAP"),
        help="IEMOCAP 数据集根目录",
    )
    parser.add_argument(
        "--ravdess",
        type=str,
        default=str(Path("data") / "RAVDESS"),
        help="RAVDESS 数据集根目录",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=5,
        help="每个数据集最多可视化的样本数",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="重采样采样率（默认 None 表示保留原始采样率）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="每条音频的截断时长（秒），默认 None 表示不截断",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=40,
        help="MFCC 维度",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["mel", "mfcc", "both"],
        default="mel",
        help="选择可视化类型：mel（默认）、mfcc 或 both",
    )
    parser.add_argument(
        "--include-stft",
        action="store_true",
        help="绘制梅尔谱图时，额外加入线性 STFT 频谱作为中间过程",
    )

    args = parser.parse_args()

    out_dir = ensure_output_dir()

    datasets = [
        ("IEMOCAP", Path(args.iemocap)),
        ("RAVDESS", Path(args.ravdess)),
    ]

    total = 0
    for name, root in datasets:
        processed = visualize_dataset(
            dataset_name=name,
            dataset_root=root,
            out_dir=out_dir,
            max_files=args.max_per_dataset,
            target_sr=args.sr,
            duration=args.duration,
            n_mfcc=args.n_mfcc,
            plot=args.plot,
            include_stft=args.include_stft,
        )
        if processed == 0:
            print(f"[INFO] 未在 {root} 找到音频文件，跳过 {name}")
        else:
            print(f"[OK] {name} 处理完成：{processed} 个样本，结果已保存至 {out_dir}")
        total += processed

    if total == 0:
        print("[WARN] 未处理任何文件，请检查数据集路径是否正确以及是否包含 .wav 文件。")


if __name__ == "__main__":
    # 在 Windows PowerShell 中示例运行：
    # python visualization/vis_mfcc.py --max-per-dataset 6 --duration 5
    main()


