# -*- coding: utf-8 -*-
"""
AST (Audio Spectrogram Transformer) Model for RAVDESS Emotion Classification
Adapted from the original AST implementation: https://github.com/YuanGongND/ast
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import logging
from typing import Dict
import numpy as np

# Override the timm package to relax the input shape constraint
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    Audio Spectrogram Transformer Model adapted for RAVDESS emotion classification
    
    Args:
        label_dim: Number of emotion classes (8 for RAVDESS)
        fstride: Frequency stride for patch splitting (default: 10)
        tstride: Time stride for patch splitting (default: 10) 
        input_fdim: Number of frequency bins (default: 128)
        input_tdim: Number of time frames (depends on audio length)
        imagenet_pretrain: Use ImageNet pretrained weights
        audioset_pretrain: Use AudioSet pretrained weights
        model_size: Model size [tiny224, small224, base224, base384]
    """
    
    def __init__(self, label_dim=8, fstride=10, tstride=10, input_fdim=128, 
                 input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, 
                 model_size='base384', verbose=True):
        
        super(ASTModel, self).__init__()
        
        if verbose:
            logging.info('---------------AST Model Summary---------------')
            logging.info(f'ImageNet pretraining: {imagenet_pretrain}, AudioSet pretraining: {audioset_pretrain}')
            logging.info(f'Model size: {model_size}, Label dim: {label_dim}')
        
        # Check timm version (relax this constraint for newer versions)
        try:
            assert timm.__version__ == '0.4.5'
        except AssertionError:
            logging.warning(f'Using timm version {timm.__version__}, original AST uses 0.4.5. Some features may not work.')
        
        # Override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # Load base model without AudioSet pretraining first
        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        
        self.original_num_patches = self.v.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        
        # Create MLP head for emotion classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim), 
            nn.Linear(self.original_embedding_dim, label_dim)
        )
        
        # Get intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        if verbose:
            logging.info(f'Frequency stride: {fstride}, Time stride: {tstride}')
            logging.info(f'Number of patches: {num_patches}')
            logging.info(f'Input shape: (batch_size, {input_tdim}, {input_fdim})')
        
        # Adapt the linear projection layer for single-channel input
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, 
                                 kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain:
            # Sum RGB channels for single-channel adaptation
            new_proj.weight = torch.nn.Parameter(
                torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
            )
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        
        # Adapt positional embeddings
        self._adapt_positional_embeddings(imagenet_pretrain, f_dim, t_dim, num_patches)
        
        # Try to load AudioSet pretrained weights if requested
        if audioset_pretrain and not self._load_audioset_weights():
            logging.warning("AudioSet pretrained weights not found, using ImageNet only")
    
    def _adapt_positional_embeddings(self, imagenet_pretrain, f_dim, t_dim, num_patches):
        """Adapt positional embeddings for the new input shape"""
        if imagenet_pretrain:
            # Get positional embedding from DEIT model, skip cls and dist tokens
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                1, self.original_num_patches, self.original_embedding_dim
            ).transpose(1, 2).reshape(
                1, self.original_embedding_dim, self.original_hw, self.original_hw
            )
            
            # Adjust for time dimension
            if t_dim <= self.original_hw:
                start_idx = int(self.original_hw / 2) - int(t_dim / 2)
                new_pos_embed = new_pos_embed[:, :, :, start_idx:start_idx + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(self.original_hw, t_dim), mode='bilinear'
                )
            
            # Adjust for frequency dimension
            if f_dim <= self.original_hw:
                start_idx = int(self.original_hw / 2) - int(f_dim / 2)
                new_pos_embed = new_pos_embed[:, :, start_idx:start_idx + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode='bilinear'
                )
            
            # Flatten and concatenate with cls and dist tokens
            new_pos_embed = new_pos_embed.reshape(
                1, self.original_embedding_dim, num_patches
            ).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([
                self.v.pos_embed[:, :2, :].detach(), new_pos_embed
            ], dim=1))
        else:
            # Random initialization for new positional embeddings
            new_pos_embed = nn.Parameter(torch.zeros(
                1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim
            ))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
    
    def _load_audioset_weights(self):
        """Try to load AudioSet pretrained weights"""
        try:
            # Look for AudioSet weights in common locations
            possible_paths = [
                'ref_repos/ast/pretrained_models/audioset_10_10_0.4593.pth',
                'pretrained_models/audioset_10_10_0.4593.pth',
                'audioset_10_10_0.4593.pth'
            ]
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for path in possible_paths:
                if os.path.exists(path):
                    logging.info(f"Loading AudioSet pretrained weights from {path}")
                    sd = torch.load(path, map_location=device)
                    
                    # Load compatible weights
                    model_dict = self.state_dict()
                    pretrained_dict = {k: v for k, v in sd.items() 
                                     if k in model_dict and v.size() == model_dict[k].size()}
                    model_dict.update(pretrained_dict)
                    self.load_state_dict(model_dict)
                    
                    logging.info(f"Loaded {len(pretrained_dict)} pretrained parameters")
                    return True
                    
            return False
        except Exception as e:
            logging.error(f"Failed to load AudioSet weights: {e}")
            return False
    
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        """Calculate output shape after patch embedding"""
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, 
                            kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    @autocast()
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input spectrogram of shape (batch_size, time_frames, frequency_bins)
               e.g., (32, 1024, 128)
        
        Returns:
            Emotion predictions of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, time_frames, frequency_bins)
        # Convert to: (batch_size, 1, frequency_bins, time_frames)
        x = x.unsqueeze(1)  # Add channel dimension
        x = x.transpose(2, 3)  # Swap time and frequency for conv2d
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # Add class and distillation tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        # Add positional embeddings
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # Transformer blocks
        for blk in self.v.blocks:
            x = blk(x)
        
        x = self.v.norm(x)
        
        # Average cls and dist tokens
        x = (x[:, 0] + x[:, 1]) / 2
        
        # Classification head
        x = self.mlp_head(x)
        return x


class RAVDESSASTModel(nn.Module):
    """
    Wrapper for AST model specifically configured for RAVDESS dataset
    """
    
    def __init__(self, config: Dict):
        """
        Initialize RAVDESS AST model
        
        Args:
            config: Model configuration dictionary
        """
        super(RAVDESSASTModel, self).__init__()
        
        self.num_classes = config['num_classes']
        model_size = config.get('model_size', 'base384')
        imagenet_pretrain = config.get('imagenet_pretrain', True)
        audioset_pretrain = config.get('audioset_pretrain', False)
        
        # Calculate input time dimension based on audio length
        sample_rate = 22050  # Standard sample rate
        audio_length = 3.0   # 3 seconds
        frame_shift = 10     # 10ms frame shift
        input_tdim = int(audio_length * 1000 / frame_shift)  # ~300 frames for 3s
        
        self.ast_model = ASTModel(
            label_dim=self.num_classes,
            fstride=10,
            tstride=10,
            input_fdim=128,  # 128 mel bins
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
            verbose=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"AST Model initialized with {total_params:,} total parameters")
        logging.info(f"Trainable parameters: {trainable_params:,}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input mel spectrogram of shape (batch_size, time_frames, mel_bins)
        
        Returns:
            Emotion predictions
        """
        return self.ast_model(x)


if __name__ == "__main__":
    # Test AST model
    config = {
        'num_classes': 8,
        'model_size': 'small224',  # Use smaller model for testing
        'imagenet_pretrain': True,
        'audioset_pretrain': False
    }
    
    model = RAVDESSASTModel(config)
    
    # Test with dummy input
    batch_size = 4
    time_frames = 300  # ~3 seconds of audio
    mel_bins = 128
    
    dummy_input = torch.randn(batch_size, time_frames, mel_bins)
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {config['num_classes']})")
    
    print("AST model test completed successfully!") 