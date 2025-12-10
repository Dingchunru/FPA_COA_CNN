# main.py - GPU强制加速版
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                           recall_score, confusion_matrix, classification_report,
                           roc_auc_score)
import json

from config import Config
from data.preprocess import load_cicids2017, reshape_for_cnn
from algorithms.hybrid_optimizer import HybridFPA_COA_Optimizer
from models.cnn_model import IDSCNN
from utils.metrics import calculate_detailed_metrics as calculate_metrics
from utils.metrics import plot_confusion_matrix, plot_roc_curve

# GPU加速检查
print("="*60)
print("GPU加速检查")
print("="*60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 设置GPU优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # 启用TF32（如果支持）
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("启用TF32精度（Ampere架构以上）")
else:
    print("警告: CUDA不可用，将使用CPU训练")

class GPUTrainer:
    """GPU优化的训练辅助类"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.use_gpu = self.device.type == 'cuda'
        
        # 混合精度训练
        self.scaler = None
        self.autocast = None
        if self.use_gpu and config.TRAIN_CONFIG.get('mixed_precision', False):
            try:
                from torch.cuda.amp import GradScaler, autocast
                self.scaler = GradScaler()
                self.autocast = autocast
                print("✅ 启用混合精度训练")
            except ImportError:
                print("⚠ 混合精度训练不可用，使用普通训练")
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """GPU优化的训练epoch"""
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        accumulation_steps = self.config.TRAIN_CONFIG.get('gradient_accumulation_steps', 1)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 数据移动到GPU
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 混合精度训练
            if self.scaler is not None and self.autocast is not None:
                with self.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # 普通训练
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # 统计
            train_loss += loss.item() * inputs.size(0) * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 定期清理GPU缓存
            if self.use_gpu and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        return train_loss / len(train_loader.dataset), 100 * train_correct / train_total
    
    def evaluate(self, model, data_loader, criterion):
        """GPU优化的评估"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 混合精度推理
                if self.scaler is not None and self.autocast is not None:
                    with self.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        return val_loss / len(data_loader.dataset), 100 * val_correct / val_total

class FPACOACNNTrainer:
    """FPA-COA-CNN训练器（GPU强制加速版）"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.use_gpu = self.device.type == 'cuda'
        self.model = None
        self.selected_features = None
        
        # 根据设备设置模型保存路径
        if self.use_gpu:
            self.best_model_path = "model/best_fpa_coa_cnn_gpu.pth"
        else:
            self.best_model_path = "model/best_fpa_coa_cnn_cpu.pth"
        
        # GPU优化设置
        if self.use_gpu:
            self.gpu_trainer = GPUTrainer(config)
            print(f"\n✅ GPU加速已启用")
            print(f"   设备: {torch.cuda.get_device_name(self.device.index)}")
            print(f"   内存: {torch.cuda.get_device_properties(self.device.index).total_memory / 1024**3:.2f} GB")
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
        else:
            self.gpu_trainer = None
            print("\n⚠ 使用CPU训练，速度可能较慢")
        
        # 创建目录
        os.makedirs("model", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    def create_feature_selection_objective(self, X_train, y_train, X_val, y_val):
        """创建特征选择的目标函数（GPU优化版）"""
        
        def objective(feature_subset):
            # 选择特征
            selected_indices = np.where(feature_subset > 0.5)[0]
            
            if len(selected_indices) == 0:
                return 2.0
            
            # 确保选择足够多的特征
            min_features = self.config.FEATURE_SELECTION['min_features']
            max_features_ratio = self.config.FEATURE_SELECTION['max_features_ratio']
            
            if len(selected_indices) < min_features:
                return 1.5
            if len(selected_indices) > X_train.shape[1] * max_features_ratio:
                return 1.3
            
            # 使用选定特征
            X_train_selected = X_train[:, selected_indices]
            X_val_selected = X_val[:, selected_indices]
            
            # 重塑为CNN格式
            sequence_length = 100
            X_train_cnn = reshape_for_cnn(X_train_selected, sequence_length)
            X_val_cnn = reshape_for_cnn(X_val_selected, sequence_length)
            
            # 转换为Tensor - 保持在CPU上，让DataLoader处理GPU传输
            X_train_t = torch.FloatTensor(X_train_cnn)  # 保持在CPU
            y_train_t = torch.LongTensor(y_train)       # 保持在CPU
            X_val_t = torch.FloatTensor(X_val_cnn)      # 保持在CPU
            
            # 创建数据集
            train_dataset = TensorDataset(X_train_t, y_train_t)
            
            # 使用快速评估配置
            eval_batch_size = self.config.FEATURE_SELECTION.get('eval_batch_size', 256 if self.use_gpu else 64)
            
            # 修复pin_memory问题
            # 数据在CPU上时，可以启用pin_memory加速GPU传输
            pin_memory = self.use_gpu  # 如果使用GPU，启用pin_memory
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=eval_batch_size, 
                shuffle=True,
                pin_memory=pin_memory,
                num_workers=0  # 对于快速评估，不需要多进程
            )
            
            # 创建CNN模型
            num_classes = len(np.unique(y_train))
            
            # 获取CNN配置并移除冲突参数
            cnn_config = self.config.CNN_CONFIG.copy()
            conflict_keys = ['input_channels', 'sequence_length', 'num_classes']
            for key in conflict_keys:
                cnn_config.pop(key, None)
            
            model = IDSCNN(
                input_channels=1,
                sequence_length=sequence_length,
                num_classes=num_classes,
                **cnn_config
            ).to(self.device)
            
            # 快速训练评估
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 快速训练（减少epoch以加速）
            eval_epochs = self.config.FEATURE_SELECTION.get('eval_epochs', 2)
            model.train()
            
            for epoch in range(eval_epochs):
                for inputs, labels in train_loader:
                    # 在训练循环中移动数据到GPU
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    
                    if self.gpu_trainer and self.gpu_trainer.scaler:
                        with self.gpu_trainer.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        self.gpu_trainer.scaler.scale(loss).backward()
                        self.gpu_trainer.scaler.step(optimizer)
                        self.gpu_trainer.scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
            
            # 快速评估
            model.eval()
            # 将验证数据移动到GPU
            X_val_t_gpu = X_val_t.to(self.device)
            
            with torch.no_grad():
                if self.gpu_trainer and self.gpu_trainer.scaler:
                    with self.gpu_trainer.autocast():
                        outputs = model(X_val_t_gpu)
                else:
                    outputs = model(X_val_t_gpu)
                
                _, predictions = torch.max(outputs, 1)
                accuracy = (predictions.cpu().numpy() == y_val).mean()
            
            # 复合适应度函数
            feature_ratio = len(selected_indices) / X_train.shape[1]
            
            if feature_ratio < 0.2:
                feature_penalty = np.exp(0.2 - feature_ratio) - 1
            elif feature_ratio > 0.6:
                feature_penalty = np.exp(feature_ratio - 0.6) - 1
            else:
                feature_penalty = 0
            
            fitness = (1 - accuracy) * 0.6 + feature_penalty * 0.4
            
            # 清理GPU内存
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            return fitness
        
        return objective
    
    def prepare_cnn_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                            sequence_length=100, batch_size=512):
        """准备CNN数据集（GPU优化）"""
        
        # 重塑为CNN格式
        X_train_cnn = reshape_for_cnn(X_train, sequence_length)
        X_val_cnn = reshape_for_cnn(X_val, sequence_length)
        X_test_cnn = reshape_for_cnn(X_test, sequence_length)
        
        # 转换为Tensor（保持在CPU上，让DataLoader处理GPU传输）
        X_train_t = torch.FloatTensor(X_train_cnn)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val_cnn)
        y_val_t = torch.LongTensor(y_val)
        X_test_t = torch.FloatTensor(X_test_cnn)
        y_test_t = torch.LongTensor(y_test)
        
        # 创建TensorDataset
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # GPU优化的DataLoader
        if self.use_gpu:
            # GPU训练：启用pin_memory和num_workers加速
            num_workers = min(4, os.cpu_count() // 2)
            pin_memory = True
        else:
            # CPU训练：禁用优化
            num_workers = 0
            pin_memory = False
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        return train_loader, val_loader, test_loader
    
    def train_final_model(self, X_train, y_train, X_val, y_val, selected_features):
        """训练最终的CNN模型（GPU强制加速）"""
        print("\n" + "="*60)
        if self.use_gpu:
            print("训练最终CNN模型（GPU强制加速）")
        else:
            print("训练最终CNN模型（CPU模式）")
        print("="*60)
        
        # 使用选定特征
        X_train_selected = X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]
        
        sequence_length = 100
        num_classes = len(np.unique(y_train))
        
        # 准备数据集
        train_loader, val_loader, _ = self.prepare_cnn_datasets(
            X_train_selected, X_val_selected, X_val_selected,
            y_train, y_val, y_val,
            sequence_length=sequence_length,
            batch_size=self.config.TRAIN_CONFIG['batch_size']
        )
        
        # 创建CNN模型
        cnn_config = self.config.CNN_CONFIG.copy()
        conflict_keys = ['input_channels', 'sequence_length', 'num_classes']
        for key in conflict_keys:
            cnn_config.pop(key, None)
        
        self.model = IDSCNN(
            input_channels=1,
            sequence_length=sequence_length,
            num_classes=num_classes,
            **cnn_config
        ).to(self.device)
        
        # 如果有多个GPU，使用数据并行
        if self.use_gpu and torch.cuda.device_count() > 1:
            print(f"✅ 使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
            self.model = nn.DataParallel(self.model)
        
        # 设置优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.TRAIN_CONFIG['learning_rate'],
            weight_decay=self.config.TRAIN_CONFIG['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=self.config.TRAIN_CONFIG['patience'] // 2,
            verbose=True
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        print(f"\n开始训练，共{self.config.TRAIN_CONFIG['epochs']}个epoch...")
        print(f"训练样本: {len(train_loader.dataset):,}")
        print(f"验证样本: {len(val_loader.dataset):,}")
        print(f"批量大小: {self.config.TRAIN_CONFIG['batch_size']}")
        if self.use_gpu:
            print(f"数据加载优化: pin_memory=True, num_workers={min(4, os.cpu_count()//2)}")
        print("-"*80)
        
        for epoch in range(self.config.TRAIN_CONFIG['epochs']):
            epoch_start = time.time()
            
            # GPU优化的训练
            if self.gpu_trainer:
                train_loss, train_accuracy = self.gpu_trainer.train_epoch(
                    self.model, train_loader, criterion, optimizer, epoch
                )
                val_loss, val_accuracy = self.gpu_trainer.evaluate(
                    self.model, val_loader, criterion
                )
            else:
                # CPU训练
                train_loss, train_accuracy = self._train_epoch_cpu(
                    self.model, train_loader, criterion, optimizer
                )
                val_loss, val_accuracy = self._evaluate_cpu(
                    self.model, val_loader, criterion
                )
            
            # 学习率调度
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            epoch_time = time.time() - epoch_start
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'lr': current_lr,
                'time': epoch_time
            })
            
            # 打印进度（带GPU内存信息）
            gpu_memory = ""
            if self.use_gpu:
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                gpu_memory = f" | GPU内存: {allocated:.2f}/{cached:.2f} GB"
            
            print(f"Epoch {epoch+1:3d}/{self.config.TRAIN_CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:6.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:6.2f}% | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s{gpu_memory}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存模型（如果是DataParallel，保存module）
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'selected_features': selected_features,
                }, self.best_model_path)
                print(f"  ✅ 保存最佳模型 (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.TRAIN_CONFIG['patience']:
                    print(f"\n早停于第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 计算验证集详细指标
        all_val_preds = []
        all_val_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro')
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
        
        print("\n验证集性能:")
        print(f"  准确率: {val_accuracy:.2f}%")
        print(f"  F1分数: {val_f1:.4f}")
        print(f"  精确率: {val_precision:.4f}")
        print(f"  召回率: {val_recall:.4f}")
        
        # 清理GPU内存
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        return training_history
    
    def _train_epoch_cpu(self, model, train_loader, criterion, optimizer):
        """CPU训练epoch"""
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        return train_loss / len(train_loader.dataset), 100 * train_correct / train_total
    
    def _evaluate_cpu(self, model, data_loader, criterion):
        """CPU评估"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        return val_loss / len(data_loader.dataset), 100 * val_correct / val_total
    
    def evaluate_model(self, X_test, y_test, label_names=None):
        """评估模型在测试集上的性能"""
        print("\n" + "="*60)
        print("测试集评估")
        print("="*60)
        
        # 使用选定特征
        X_test_selected = X_test[:, self.selected_features]
        
        # 重塑为CNN格式
        sequence_length = 100
        
        # 准备测试集
        _, _, test_loader = self.prepare_cnn_datasets(
            X_test_selected, X_test_selected, X_test_selected,
            y_test, y_test, y_test,
            sequence_length=sequence_length,
            batch_size=512 if self.use_gpu else 128
        )
        
        # 预测
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                outputs = self.model(inputs)
                end_time = time.time()
                
                # 记录时间（毫秒/样本）
                batch_time = (end_time - start_time) * 1000 / len(inputs)
                inference_times.extend([batch_time] * len(inputs))
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        # 计算AUC
        try:
            if len(np.unique(all_labels)) > 2:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            else:
                auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        except:
            auc = 0.0
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # 计算推理统计
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        throughput = 1000 / avg_inference_time  # 样本/秒
        
        # 打印结果
        print(f"\n测试集性能指标:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1分数 (宏平均): {f1_macro:.4f}")
        print(f"  F1分数 (微平均): {f1_micro:.4f}")
        print(f"  精确率 (宏平均): {precision:.4f}")
        print(f"  召回率 (宏平均): {recall:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"\n推理性能:")
        print(f"  平均推理时间: {avg_inference_time:.2f} ± {std_inference_time:.2f} 毫秒/样本")
        print(f"  吞吐量: {throughput:.1f} 样本/秒")
        
        if self.use_gpu:
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  峰值GPU内存使用: {peak_memory:.2f} GB")
        
        if label_names is not None:
            print(f"\n分类报告:")
            print(classification_report(all_labels, all_preds, 
                                      target_names=label_names, digits=4))
        
        # 保存结果
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'confusion_matrix': cm,
            'inference_time_mean': avg_inference_time,
            'inference_time_std': std_inference_time,
            'throughput': throughput,
            'selected_features': self.selected_features.tolist(),
            'num_selected_features': len(self.selected_features),
            'device': str(self.device)
        }
        
        np.savez("results/test_results.npz", **results)
        
        return results
    
    def run(self):
        """运行完整的FPA-COA-CNN流程"""
        print("\n" + "="*60)
        if self.use_gpu:
            print("FPA-COA-CNN入侵检测系统（GPU强制加速版）")
        else:
            print("FPA-COA-CNN入侵检测系统（CPU模式）")
        print("="*60)
        
        total_start_time = time.time()
        
        # 1. 加载数据
        print("\n[1/5] 加载CICIDS2017数据...")
        try:
            features, labels, label_encoder = load_cicids2017(
                self.config.DATA_PATH,
                sample_frac=self.config.SAMPLE_FRACTION
            )
        except Exception as e:
            print(f"加载数据失败: {e}")
            # 生成测试数据
            print("生成测试数据...")
            n_samples = 10000
            n_features = 78
            n_classes = 2
            features = np.random.randn(n_samples, n_features)
            labels = np.random.randint(0, n_classes, n_samples)
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)
            print(f"生成测试数据: {features.shape[0]} 样本, {features.shape[1]} 特征")
        
        num_classes = len(np.unique(labels))
        print(f"数据统计: {features.shape[0]:,} 样本, {features.shape[1]} 特征, {num_classes} 类别")
        
        # 2. 数据分割
        print("\n[2/5] 分割数据集...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=self.config.TEST_SIZE,
            stratify=labels,
            random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.VAL_SIZE,
            stratify=y_train,
            random_state=42
        )
        
        print(f"训练集: {X_train.shape[0]:,} 样本")
        print(f"验证集: {X_val.shape[0]:,} 样本")
        print(f"测试集: {X_test.shape[0]:,} 样本")
        
        # 3. 特征选择
        print("\n[3/5] 使用FPA-COA进行特征选择...")
        objective_func = self.create_feature_selection_objective(
            X_train, y_train, X_val, y_val
        )
        
        optimizer = HybridFPA_COA_Optimizer(
            objective_func=objective_func,
            dim=X_train.shape[1],
            pop_size=self.config.OPTIMIZER_CONFIG['pop_size'],
            iter_max=self.config.OPTIMIZER_CONFIG['iter_max'],
            fpa_params=self.config.OPTIMIZER_CONFIG['fpa_params'],
            coa_params=self.config.OPTIMIZER_CONFIG['coa_params'],
            hybrid_params=self.config.OPTIMIZER_CONFIG['hybrid_params']
        )
        
        print("开始特征选择优化...")
        best_solution, best_fitness = optimizer.run(bounds=(0, 1))
        
        # 选择特征
        threshold = self.config.FEATURE_SELECTION['threshold']
        self.selected_features = np.where(best_solution > threshold)[0]
        
        print(f"\n特征选择结果:")
        print(f"  原始特征数: {X_train.shape[1]}")
        print(f"  选择特征数: {len(self.selected_features)}")
        print(f"  选择比例: {len(self.selected_features)/X_train.shape[1]:.1%}")
        print(f"  最佳适应度: {best_fitness:.6f}")
        
        # 如果选择的特征太少，使用前30%的特征
        if len(self.selected_features) < self.config.FEATURE_SELECTION['min_features']:
            print(f"选择特征过少，使用前30%的特征...")
            n_selected = max(self.config.FEATURE_SELECTION['min_features'], 
                           int(X_train.shape[1] * 0.3))
            self.selected_features = np.arange(n_selected)
        
        # 保存特征选择结果
        np.savez("results/feature_selection.npz",
                best_solution=best_solution,
                selected_features=self.selected_features,
                best_fitness=best_fitness)
        
        # 4. 训练最终模型
        print("\n[4/5] 训练最终CNN模型...")
        training_history = self.train_final_model(
            X_train, y_train, X_val, y_val, self.selected_features
        )
        
        # 5. 评估模型
        print("\n[5/5] 在测试集上评估模型...")
        test_results = self.evaluate_model(
            X_test, y_test, 
            label_encoder.classes_ if hasattr(label_encoder, 'classes_') else None
        )
        
        total_time = time.time() - total_start_time
        
        # 最终报告
        print("\n" + "="*60)
        print("FPA-COA-CNN训练完成!")
        print("="*60)
        print(f"总运行时间: {total_time/60:.2f} 分钟")
        print(f"最终测试准确率: {test_results['accuracy']:.4f}")
        print(f"F1分数 (宏平均): {test_results['f1_macro']:.4f}")
        print(f"选择的特征数: {len(self.selected_features)}")
        print(f"模型已保存到: {self.best_model_path}")
        
        if self.use_gpu:
            # GPU性能统计
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"峰值GPU内存使用: {peak_memory:.2f} GB")
        
        print("="*60)
        
        return test_results

def main():
    """主函数"""
    print("FPA-COA-CNN入侵检测系统")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 创建配置实例
    config = Config()
    
    # 创建训练器
    trainer = FPACOACNNTrainer(config)
    
    try:
        # 运行训练
        results = trainer.run()
        
        # 保存最终配置
        with open("results/config_summary.json", "w") as f:
            json.dump({
                'accuracy': float(results['accuracy']),
                'f1_macro': float(results['f1_macro']),
                'num_selected_features': int(results['num_selected_features']),
                'inference_time': float(results['inference_time_mean']),
                'throughput': float(results['throughput']),
                'device': str(config.DEVICE),
                'batch_size': config.TRAIN_CONFIG['batch_size'],
                'mixed_precision': config.TRAIN_CONFIG.get('mixed_precision', False),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        print("\n✅ 结果已保存到 results/ 目录")
        
    except KeyboardInterrupt:
        print("\n⚠ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()