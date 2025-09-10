"""
Model trainer for trajectory generation
轨迹生成模型训练器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from ..utils.logger import TrainingLogger


class ModelTrainer:
    def __init__(self, 
                 model: nn.Module,
                 dataset: Any,
                 config: Dict[str, Any],
                 logger: logging.Logger,
                 model_name: str = "model",
                 seed: int = 42,
                 output_dir: str = "experiments"):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger
        self.model_name = model_name
        self.seed = seed
        self.output_dir = output_dir
        self.training_logger = TrainingLogger(logger)
        
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        self.num_epochs = config.get('num_epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        self.early_stopping = config.get('early_stopping', {})
        self.patience = self.early_stopping.get('patience', 20)
        self.min_delta = self.early_stopping.get('min_delta', 1e-4)
        self.patience_counter = 0
        
        self.use_amp = config.get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        
        self.save_every = config.get('save_every', 10)
        self.eval_every = config.get('eval_every', 5)
        
        self.use_wandb = config.get('tracking', {}).get('use_wandb', False) and WANDB_AVAILABLE
        
        self.logger.info(f"训练器初始化完成，设备: {self.device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        optimizer_type = self.config.get('optimizer', 'adam')
        
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
        else:
            return None
    
    def train(self):
        """开始训练"""
        self.logger.info("开始训练...")
        
        # 记录模型信息
        self.training_logger.log_model_info(self.model)
        self.training_logger.log_data_info(self.dataset)
        
        # 创建数据加载器
        from ..data.dataset import collate_trajectory_batch
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_trajectory_batch
        )
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = None
            if epoch % self.eval_every == 0:
                val_loss = self._validate()
                if val_loss is not None:
                    self.val_losses.append(val_loss)
            
            # 记录日志
            self.training_logger.log_epoch(epoch, train_loss, val_loss)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 保存检查点
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, train_loss, val_loss)
            
            # 早停检查
            if val_loss is not None and self._should_early_stop(val_loss):
                self.logger.info(f"早停触发，在第{epoch}轮停止训练")
                break
            
            # Wandb记录
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            batch = self._move_batch_to_device(batch)
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model.compute_loss(batch)
                    loss = loss_dict['loss']
            else:
                loss_dict = self.model.compute_loss(batch)
                loss = loss_dict['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # 梯度裁剪
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # 记录步骤日志
            if batch_idx % 100 == 0:
                metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items() if k != 'loss'}
                self.training_logger.log_step(loss.item(), metrics)
        
        return total_loss / num_batches
    
    def _validate(self) -> Optional[float]:
        """验证模型"""
        if not hasattr(self.dataset, 'val_loader') or self.dataset.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.dataset.val_loader:
                batch = self._move_batch_to_device(batch)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.model.compute_loss(batch)
                        loss = loss_dict['loss']
                else:
                    loss_dict = self.model.compute_loss(batch)
                    loss = loss_dict['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else None
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将批次数据移动到设备"""
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """检查是否应该早停"""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 创建checkpoint目录
        checkpoint_dir = Path(self.output_dir) / "checkpoints" / f"{self.model_name}_{self.seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最新检查点
        checkpoint_path = checkpoint_dir / f"{self.model_name}_ckpt_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存
        if val_loss is not None and val_loss <= self.best_val_loss:
            best_path = checkpoint_dir / f"{self.model_name}_ckpt_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型，验证损失: {val_loss:.6f}")
    
    def save_checkpoint(self, path: Path):
        """保存检查点到指定路径"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"检查点已保存到: {path}")
    
    def load_checkpoint(self, path: Path):
        """从路径加载检查点"""
        if not path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"检查点已从{path}加载，epoch: {self.current_epoch}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }