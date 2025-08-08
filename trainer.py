import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import time
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model to save best weights
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
                logging.info("Restored best weights")
            return True
        return False

    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()


class MetricsCalculator:
    """Calculate and track training metrics"""

    def __init__(self, num_classes: int, class_names: List[str]):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        """Reset metrics"""
        self.all_predictions = []
        self.all_labels = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor, loss: float):
        """
        Update metrics with batch results
        
        Args:
            predictions: Model predictions (logits)
            labels: True labels
            loss: Batch loss
        """
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()

        self.all_predictions.extend(pred_classes)
        self.all_labels.extend(true_labels)
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics
        
        Returns:
            Dictionary of metrics
        """
        avg_loss = self.total_loss / self.num_batches
        accuracy = accuracy_score(self.all_labels, self.all_predictions)

        # Compute precision, recall, F1 (macro average)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, average='macro', zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.all_labels, self.all_predictions)

    def plot_confusion_matrix(self, save_path: str = None, title: str = "Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot
            title: Plot title
        """
        cm = self.get_confusion_matrix()

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        # plt.show()


class Trainer:
    """
    Main trainer class for emotion recognition models
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = self._get_device()
        self.emotion_labels = list(config['emotions'].values())

        # Create output directories
        self._create_directories()

        # Setup logging
        self._setup_logging()

        # Initialize tracking
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}

        logging.info(f"Trainer initialized with device: {self.device}")

    def _get_device(self) -> torch.device:
        """Get training device"""
        device_config = self.config['device']
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device_config)

    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration"""
        # Logging is already configured in main.py
        # Just ensure log directory exists
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        logging.info("Trainer logging initialized")

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])

        logging.info(f"Creating optimizer: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")

        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer: optim.Optimizer):
        """Create learning rate scheduler"""
        scheduler_name = self.config['training']['scheduler'].lower()

        if scheduler_name == 'step':
            return StepLR(optimizer,
                         step_size=self.config['training']['step_size'],
                         gamma=self.config['training']['gamma'])
        else:
            return None

    def train_epoch(self, model: nn.Module, train_loader, criterion, optimizer) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Training metrics
        """
        model.train()
        metrics_calc = MetricsCalculator(len(self.emotion_labels), self.emotion_labels)

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device, dtype=torch.long)



            optimizer.zero_grad()
            output = model(data)



            # Defensive label range check to surface issues early
            try:
                expected_classes = self.config['models'][self.config['training']['model_name']]['num_classes']
                if target.numel() > 0 and (target.min().item() < 0 or target.max().item() >= expected_classes):
                    raise ValueError(f"Label index out of range: min={int(target.min().item())}, max={int(target.max().item())}, expected [0,{expected_classes-1}]")
            except Exception as e:
                logging.error(str(e))
                raise

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Update metrics
            metrics_calc.update(output, target, loss.item())

            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return metrics_calc.compute()

    def validate_epoch(self, model: nn.Module, val_loader, criterion) -> Tuple[Dict[str, float], MetricsCalculator]:
        """
        Validate for one epoch
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Validation metrics and metrics calculator
        """
        model.eval()
        metrics_calc = MetricsCalculator(len(self.emotion_labels), self.emotion_labels)

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device, dtype=torch.long)
                output = model(data)
                loss = criterion(output, target)

                # Update metrics
                metrics_calc.update(output, target, loss.item())

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        metrics = metrics_calc.compute()
        return metrics, metrics_calc

    def save_model(self, model: nn.Module, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save regular checkpoint
        model_name = self.config['training']['model_name']
        checkpoint_path = os.path.join(self.config['training']['checkpoint_dir'], f'{model_name}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.config['training']['checkpoint_dir'], f'{model_name}_best.pth')
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model with validation accuracy: {metrics['accuracy']:.4f}")

    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.train_history['loss'], label='Train Loss')
        ax1.plot(self.val_history['loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.train_history['accuracy'], label='Train Accuracy')
        ax2.plot(self.val_history['accuracy'], label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        # plt.show()

    def train(self, model: nn.Module, train_loader, val_loader) -> Dict[str, float]:
        """
        Full training loop
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Best validation metrics
        """
        # Move model to device
        model = model.to(self.device)

        # Setup training components
        criterion = nn.CrossEntropyLoss()
        # Sanity log: ensure labels range fits num_classes
        try:
            model_num_classes = self.config['models'][self.config['training']['model_name']]['num_classes']
            logging.info(f"Trainer: expecting label indices in [0, {model_num_classes-1}]")
        except Exception:
            pass
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta']
        )

        best_val_metrics = None
        start_time = time.time()

        logging.info("Starting training...")

        for epoch in range(self.config['training']['num_epochs']):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer)

            # Validate
            val_metrics, val_metrics_calc = self.validate_epoch(model, val_loader, criterion)

            # Update learning rate
            if scheduler:
                scheduler.step()

            # Track history
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['accuracy'].append(val_metrics['accuracy'])

            # Check if best model
            is_best = best_val_metrics is None or val_metrics['accuracy'] > best_val_metrics['accuracy']
            if is_best:
                best_val_metrics = val_metrics.copy()

            # Save model
            if self.config['training']['save_model']:
                if self.config['training']['save_best_only']:
                    if is_best:
                        self.save_model(model, epoch, val_metrics, is_best=True)
                else:
                    self.save_model(model, epoch, val_metrics, is_best)

            # Logging
            epoch_time = time.time() - epoch_start
            logging.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
                        f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f} - "
                        f"Time: {epoch_time:.2f}s")

            # Persist per-epoch metrics to experiment folder
            try:
                exp_dir = self.config['logging']['log_dir']
                per_epoch_log = os.path.join(exp_dir, 'epoch_metrics.csv')
                header_needed = not os.path.exists(per_epoch_log)
                with open(per_epoch_log, 'a', encoding='utf-8') as f:
                    if header_needed:
                        f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')
                    f.write(f"{epoch+1},{train_metrics['loss']:.6f},{train_metrics['accuracy']:.6f},{val_metrics['loss']:.6f},{val_metrics['accuracy']:.6f}\n")
            except Exception:
                pass

            # Early stopping
            if early_stopping(val_metrics['loss'], model):
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Training completed
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f}s")
        logging.info(f"Best validation metrics: {best_val_metrics}")

        # Plot training history
        plot_path = os.path.join(self.config['logging']['log_dir'], 'training_history.png')
        self.plot_training_history(plot_path)

        # Plot final confusion matrix
        if self.config['evaluation']['confusion_matrix']:
            cm_path = os.path.join(self.config['logging']['log_dir'], 'confusion_matrix.png')
            val_metrics_calc.plot_confusion_matrix(cm_path, "Validation Confusion Matrix")

        # Save best metrics as JSON for downstream tools
        try:
            import json
            best_path = os.path.join(self.config['logging']['log_dir'], 'best_metrics.json')
            with open(best_path, 'w', encoding='utf-8') as f:
                json.dump(best_val_metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return best_val_metrics


if __name__ == "__main__":
    # Test trainer
    from data_processing.data_loader import load_config
    from models import create_model
    from data_processing.feature_extractor import create_feature_data_loaders

    config = load_config()

    # Create model and data loaders
    model = create_model(config)
    train_loader, val_loader = create_feature_data_loaders(config)

    # Create trainer and train
    trainer = Trainer(config)
    best_metrics = trainer.train(model, train_loader, val_loader)

    print("Training completed!")
    print(f"Best validation metrics: {best_metrics}")