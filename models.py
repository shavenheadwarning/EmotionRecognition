import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import logging


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MFCC features
    """

    def __init__(self, config: Dict):
        """
        Initialize MLP model
        
        Args:
            config: Model configuration dictionary
        """
        super(MLP, self).__init__()

        self.input_size = config['input_size'] * 2  # MFCC mean + std
        self.hidden_sizes = config['hidden_sizes']
        self.num_classes = config['num_classes']
        self.dropout = config['dropout']

        # Build layers
        layers = []
        prev_size = self.input_size

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, self.num_classes))

        self.network = nn.Sequential(*layers)

        logging.info(f"MLP initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input MFCC features of shape (batch_size, feature_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.network(x)





class ShallowCNN(nn.Module):
    """
    Shallow CNN for Mel Spectrogram features
    """

    def __init__(self, config: Dict):
        """
        Initialize Shallow CNN model
        
        Args:
            config: Model configuration dictionary
        """
        super(ShallowCNN, self).__init__()

        self.num_classes = config['num_classes']
        self.dropout = config['dropout']

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.dropout_layer = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(128, self.num_classes)

        logging.info(f"ShallowCNN initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input mel spectrogram of shape (batch_size, 1, n_mels, time_frames)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Classifier
        x = self.dropout_layer(x)
        x = self.classifier(x)

        return x


class BasicBlock(nn.Module):
    """
    Basic block for ResNet
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 for Mel Spectrogram features
    """

    def __init__(self, config: Dict):
        """
        Initialize ResNet-18 model
        
        Args:
            config: Model configuration dictionary
        """
        super(ResNet18, self).__init__()

        self.num_classes = config['num_classes']
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, self.num_classes)

        # Defensive check to avoid device-side asserts when labels are out of range
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        # Initialize weights
        self._initialize_weights()

        logging.info(f"ResNet18 initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def _make_layer(self, block, planes, num_blocks, stride):
        """Make ResNet layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input mel spectrogram of shape (batch_size, 1, n_mels, time_frames)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)

        return x


def create_model(config: Dict) -> nn.Module:
    """
    Create model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch model
    """
    model_name = config['training']['model_name']
    model_config = config['models'][model_name]
    model_type = model_config['type']

    if model_type == 'MLP':
        model = MLP(model_config)

    elif model_type == 'ShallowCNN':
        model = ShallowCNN(model_config)
    elif model_type == 'ResNet18':
        model = ResNet18(model_config)
    elif model_type == 'AST':
        # Import AST model
        from ast_model import RAVDESSASTModel
        model = RAVDESSASTModel(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logging.info(f"Created {model_type} model")
    return model


if __name__ == "__main__":
    # Test models
    from data_processing.data_loader import load_config

    config = load_config()

    # Test MLP
    config['training']['model_name'] = 'mlp'
    mlp_model = create_model(config)
    test_mfcc = torch.randn(4, 26)  # batch_size=4, features=13*2
    output = mlp_model(test_mfcc)
    print(f"MLP output shape: {output.shape}")

    # Test ShallowCNN
    config['training']['model_name'] = 'shallow_cnn'
    cnn_model = create_model(config)
    test_mel = torch.randn(4, 1, 128, 130)  # batch_size=4, channels=1, n_mels=128, time_frames=130
    output = cnn_model(test_mel)
    print(f"ShallowCNN output shape: {output.shape}")

    # Test ResNet18
    config['training']['model_name'] = 'resnet18'
    resnet_model = create_model(config)
    output = resnet_model(test_mel)
    print(f"ResNet18 output shape: {output.shape}")