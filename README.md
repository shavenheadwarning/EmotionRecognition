# Speech Emotion Recognition (SER) Project

基于多特征和深度模型的语音情感识别系统研究项目。

## 项目概述

本项目实现了多种先进模型用于语音情感识别：

- **MLP (多层感知机)**: 处理MFCC特征
- **Shallow CNN (浅层卷积神经网络)**: 处理2D Mel频谱图特征
- **ResNet-18**: 处理2D Mel频谱图特征
- **AST (Audio Spectrogram Transformer)**: 基于Vision Transformer的音频分类模型 🆕

支持RAVDESS与IEMOCAP数据集：
- RAVDESS：8类（neutral, calm, happy, sad, angry, fearful, disgust, surprised）
- IEMOCAP：内置4类配置（angry/happy/sad/neutral，将excited合并到happy），可按需扩展

### 最近更新（数据增强与IEMOCAP集成）
- 新增 ESC-50 环境噪声增广（训练集生效，验证集保持干净），两种接入方式：
  - 目录模式：为两类环境噪声分别指定包含`.wav`的目录
  - meta模式（推荐）：使用官方`meta/esc50.csv`与`audio/`，按类别名分组抽取两类噪声
- 同时支持非 AST 管线与 AST 管线（波形级混合→特征提取）；SNR 支持`[0,5,10,20]`等集合
- IEMOCAP 数据集加载器完善：按会话划分、标签规范为0起连续索引，避免越界；支持4类映射
- 训练日志新增：在启用增广时打印所用增广器与`p_apply`


### 快速开始（训练命令）
```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) RAVDESS（示例：ResNet-18）
python main.py --config config/config.yaml --model resnet18

# 3) IEMOCAP（示例：AST Base）
python main.py --config config/iemocap.yaml --model ast_base

# 可选：指定设备/批大小/学习率
python main.py --config config/iemocap.yaml --model resnet18 --device cuda --batch-size 64 --lr 1e-4

# 可选：启用噪声增强（编辑 config/noise.yaml 将 enabled 设为 true）
#   - 白噪声：type: white
#   - ESC-50：type: esc50（见下方配置示例）
```



## 项目结构（已更新）

```
EmotionRecognition/

├── main.py                        # 主训练脚本（自动创建 experiments/<time>_<model>/）
├── models.py                      # 模型定义（MLP / ShallowCNN / ResNet18 / AST）
├── trainer.py                     # 训练器（训练曲线/混淆矩阵/epoch日志/最优指标）
├── data_processing/
│   ├── data_loader.py             # 数据分发（RAVDESS / IEMOCAP）
│   ├── iemocap_loader.py          # IEMOCAP 数据加载与标签解析（按会话划分）
│   ├── feature_extractor.py       # MFCC / Mel 特征
│   └── ast_feature_extractor.py   # AST专用fbank/SpecAugment/归一化
├── augmentations/
│   └── noise.py                   # 白噪声增强（SNR可控，波形级）
├── config/
│   ├── config.yaml                # RAVDESS 默认配置
│   ├── iemocap.yaml               # IEMOCAP 配置（4类，按会话划分）
│   └── noise.yaml                 # 噪声增强开关与参数
├── data/
│   ├── RAVDESS/                   # RAVDESS 数据
│   └── IEMOCAP/                   # IEMOCAP 数据（见 data/IEMOCAP/README.md）
├── experiments/                   # 每次运行自动归档（见下文）
├── utils/
│   └── experiment_manager.py      # 实验工具（预留）
└── requirements*.txt
```








### 数据准备

确保RAVDESS数据集已放置在 `data/RAVDESS/` 目录下，包含以下子目录：
- `Audio_Speech_Actors_01-24/`
- `Audio_Song_Actors_01-24/`

IEMOCAP：将官方解压后的`Session1..Session5`放置到 `data/IEMOCAP/`（支持`SessionX/SessionX/`嵌套结构）。
- 详细结构与标签说明：见 `data/IEMOCAP/README.md`
- 四类映射在 `config/iemocap.yaml: dataset.emotion_mapping` 中配置，未映射的标签会被跳过
- 默认按会话划分避免说话人泄漏：`train_sessions: [Session1..4]`，`val_sessions: [Session5]`





### 硬件建议



#### 原始AST
| GPU内存 | 推荐模型 | 批次大小 | 预期性能 |
|---------|----------|----------|----------|
| < 8GB | ast_tiny | 8 | ~75% |
| 8-12GB | ast_small | 16 | ~80% |
| > 12GB | ast_base | 8-16 | ~85% |

## 模型详情和性能对比（摘要）



### AST (原始实现)

AST是首个完全基于注意力机制的音频分类模型，改编自Vision Transformer。

#### 特点：
- **无卷积架构**: 纯Transformer结构
- **ImageNet预训练**: 利用视觉预训练权重
- **可变长度输入**: 支持不同长度的音频
- **SOTA性能**: 在多个音频分类基准上达到最先进结果

#### 模型变体：
- **ast_tiny**: ~5.7M参数，适合快速实验
- **ast_small**: ~22.1M参数，平衡性能和效率
- **ast_base**: ~86.6M参数，最佳性能

#### 输入特征：
- **FilterBank特征**: 128维Mel滤波器组
- **时间帧**: 300帧 (3秒音频，10ms帧移)
- **SpecAugment**: 训练时应用频率和时间掩码
- **归一化**: 使用数据集统计量归一化

### 其他模型

#### MLP模型
- **输入**: MFCC特征 (13维 × 2，包含均值和标准差)
- **架构**: 全连接层 [128, 64, 32] + Dropout
- **输出**: 8类情感分类


#### Shallow CNN模型
- **输入**: Mel频谱图 (1 × 128 × time_frames)
- **架构**: 3个卷积层 + 全局平均池化
- **特点**: 轻量级CNN架构，参数量适中

#### ResNet-18模型
- **输入**: Mel频谱图 (1 × 128 × time_frames)
- **架构**: 标准ResNet-18，适配单通道输入
- **特点**: 深度残差网络，强大的特征提取能力

## 完整性能对比

| 模型 | 参数量 | 特征类型 | 预期准确率 | 训练时间 | 内存需求 | 推荐度 |
|------|--------|----------|------------|----------|----------|--------|
| AST Base | 86.6M | FilterBank | ~85% | 慢 | 高 | ⭐⭐⭐ |
| AST Small | 22.1M | FilterBank | ~80% | 中等 | 中等 | ⭐⭐ |
| ResNet-18 | 11.2M | Mel谱图 | ~70% | 慢 | 高 | ⭐⭐ |
| Shallow CNN | 94K | Mel谱图 | ~65% | 中等 | 中等 | ⭐ |
| AST Tiny | 5.7M | FilterBank | ~75% | 中等 | 中等 | ⭐ |
| MLP | 14K | MFCC | ~30% | 快 | 低 | ⭐ |



## 实验与日志归档

每次运行自动创建：`experiments/<YYYYMMDD_HHMMSS>_<dataset>_<model>/`，包含：
- `config_resolved.yaml`: 本次运行的最终配置快照
- `main.log`: 训练日志
- `training_history.png`: 训练/验证曲线
- `confusion_matrix.png`: 验证集混淆矩阵
- `epoch_metrics.csv`: 每个epoch的关键指标
- `best_metrics.json`: 最优验证指标
- `<model>_training_summary.txt`: 摘要
- `checkpoints/`: `*_epoch_*.pth` 与 `*_best.pth`

## 评估指标

项目支持以下评估指标：
- **Accuracy**: 分类准确率
- **Precision**: 精确率 (宏平均)
- **Recall**: 召回率 (宏平均)
- **F1-Score**: F1分数 (宏平均)
- **Confusion Matrix**: 混淆矩阵

## 特征类型




### AST FilterBank Features
- 128维Mel滤波器组特征
- Kaldi兼容的fbank提取
- SpecAugment数据增强
- 标准化处理
- 适用于AST模型

### MFCC (Mel Frequency Cepstral Coefficients)
- 13维MFCC系数
- 计算均值和标准差作为统计特征
- 适用于MLP模型

### Mel Spectrogram
- 128个Mel滤波器
- 对数幅度谱
- 归一化到[0,1]范围
- 适用于CNN和ResNet模型

### 噪声增强（可选）
- 配置文件：`config/noise.yaml`
- 参数：`enabled`（开关）、`p_apply`、`snr_db_choices`（如 `[0,5,10,20]`）、`target_peak_dbfs`
- 生效范围：训练集的波形级增广（AST与非AST均已支持）

ESC-50 配置示例（二选一）：

1) 目录模式（若已手动整理两个噪声目录）
```yaml
noise_augmentation:
  enabled: true
  type: esc50
  p_apply: 0.7
  snr_db_choices: [0.0, 5.0, 10.0, 20.0]
  target_peak_dbfs: -1.0
  esc50:
    resample_sr: 16000
    categories:
      - name: natural_soundscapes
        path: d:/PTProjects/EmotionRecognition/data/ESC-50/audio/1-100/
      - name: human_non_speech
        path: d:/PTProjects/EmotionRecognition/data/ESC-50/audio/101-200/
```

2) meta模式（推荐，使用官方`meta/esc50.csv`）
```yaml
noise_augmentation:
  enabled: true
  type: esc50
  p_apply: 0.7
  snr_db_choices: [0.0, 5.0, 10.0, 20.0]
  target_peak_dbfs: -1.0
  esc50:
    resample_sr: 16000
    audio_root: d:/PTProjects/EmotionRecognition/data/ESC-50/audio
    meta_csv:   d:/PTProjects/EmotionRecognition/data/ESC-50/meta/esc50.csv
    groups:
      - name: natural_soundscapes
        categories: [rain, sea_waves, crickets, chirping_birds, wind, frog, thunderstorm, water_drops, crackling_fire, insects]
      - name: human_non_speech
        categories: [coughing, sneezing, breathing, laughing, footsteps, clapping]
```

提示：
- 若同时配置`categories`与`audio_root/meta_csv/groups`，将优先使用meta模式。
- ESC-50 目录与CSV路径请按你的本地路径修改；Windows路径区分大小写与盘符。





## 参考文献

- RAVDESS数据集: https://zenodo.org/record/1188976
- ResNet论文: "Deep Residual Learning for Image Recognition"
- **AST论文**: "AST: Audio Spectrogram Transformer" (Interspeech 2021)
- **AST代码库**: https://github.com/YuanGongND/ast
- **AST-Speech参考**: https://github.com/poojasethi/ast-speech



## 权威论文中的数据预处理流程（Mel特征）

以下流程与参数综合自音频分类/情感识别领域的权威/主流工作，涵盖AST、VGGish/AudioSet、SpecAugment等实践，给出可直接落地的推荐配置。

### 统一前处理
- 采样率（sr）: 建议统一到 16 kHz 或 22.05 kHz（AST与AudioSet主流为16 kHz）。
- 幅度范围: 浮点归一化到 [-1, 1]；可选做全局能量归一化以抑制录音强度差异。
- 预加重（可选）: 使用一阶高通滤波（如系数 0.97）增强高频，部分SER工作采用；AST/AudioSet常不显式使用预加重。

### 分帧与窗函数
- 窗长（window length）: 25 ms（AST/VGGish/AudioSet常用）
- 帧移（hop length）: 10 ms（AST/VGGish/AudioSet常用）
- 窗函数: Hamming 或 Hanning（AST实现常用 Hanning）

### STFT 与功率谱
- 计算短时傅里叶变换（STFT）并得到功率谱（magnitude²），n_fft 依据采样率设置：
  - sr=16kHz: n_fft≈400（25 ms）
  - sr=22.05kHz: n_fft≈512（≈23 ms，接近25 ms）

### Mel滤波器组与对数压缩（核心）
- Mel滤波器数量（n_mels）:
  - 64（VGGish/AudioSet传统配置）
  - 128（AST等Transformer/CNN常用配置，时频分辨率更高）
- 频率范围: f_min=0 或 30 Hz，f_max=sr/2
- 能量到对数域：对数功率或对数幅度（如 log(x+1e-6) 或分贝标度 AmplitudeToDB）
- 动态范围限制: 常见裁剪到 [−80 dB, 0 dB]

### 归一化
- 全局数据集级均值/标准差归一化（推荐）：
  - AST实践采用全局统计量（例如 AudioSet 统计：mean≈−4.2677, std≈4.5690）对 log-mel/fbank 做标准化
- 话语级 CMVN（备选）：每条样本做均值方差归一化，适合小数据集稳定训练

### 增强（可选但强烈推荐）
- SpecAugment（Park et al., Interspeech 2019）: 频率掩蔽与时间掩蔽（例如 freq_mask_param≈15, time_mask_param≈35）
- 噪声增广（本项目已集成）: 控制 SNR 的白噪声/自然噪声混合（如 0/5/10/20 dB），提升鲁棒性
- 轻微抖动（dither）: 部分Kaldi管线用于数值稳定（AST代码示例中常设 dither=0.0）

### 推荐参数组合（按模型）
- AST/Transformer：
  - sr=16 kHz, 窗长=25 ms, 帧移=10 ms, 窗函数=Hanning
  - 特征=128维 log-Mel/FBank；SpecAugment（train）+ 全局均值/方差归一化
  - 参考：AST (Interspeech 2021) 使用 Kaldi fbank（num_mel_bins=128, frame_shift=10ms, dither=0）并做全局标准化
- CNN/ResNet：
  - sr=16/22.05 kHz, 25/10 ms, Hamming；128维对数Mel谱，归一化到 [0,1] 或标准化
  - 频/时掩蔽增强；对齐项目中 `feature_extractor.py` 的实现
- VGGish风格：
  - sr=16 kHz, 64维 log-Mel，25/10 ms；可参考 AudioSet/VGGish 预处理

### 参考文献（建议检索原文以获取细节）
- Gong, Y., Chung, Y.-A., and Glass, J. “AST: Audio Spectrogram Transformer.” Interspeech, 2021.（AST使用128维fbank、10ms帧移、全局归一化）
- Park, D. S., et al. “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.” Interspeech, 2019.
- Hershey, S., et al. “CNN Architectures for Large-Scale Audio Classification.” ICASSP, 2017.（VGGish与AudioSet实践，64维log-Mel）
- Gemmeke, J. F., et al. “Audio Set: An ontology and human-labeled dataset for audio events.” ICASSP, 2017.