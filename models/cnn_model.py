# models/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IDSCNN(nn.Module):
    """
    入侵检测专用的一维CNN模型
    针对网络流量特征进行优化
    修复：添加**kwargs接收额外参数，避免重复赋值错误
    """
    
    def __init__(self, input_channels=1, sequence_length=100, num_classes=2,
                 hidden_channels=[64, 128, 256], kernel_sizes=[3, 3, 3],
                 pool_sizes=[2, 2, 2], fc_sizes=[128, 64], dropout_rate=0.3,
                 batch_norm=True, **kwargs):
        """
        参数说明：
            input_channels: 输入通道数，固定为1（一维CNN）
            sequence_length: 序列长度
            num_classes: 分类类别数
            hidden_channels: 隐藏层通道数列表
            kernel_sizes: 卷积核大小列表
            pool_sizes: 池化大小列表
            fc_sizes: 全连接层大小列表
            dropout_rate: dropout比例
            batch_norm: 是否使用批归一化
            **kwargs: 接收额外参数，防止重复赋值错误
        """
        super(IDSCNN, self).__init__()
        
        # 如果有额外的参数，忽略它们（防止重复赋值）
        if kwargs:
            # 可选：打印警告信息
            # print(f"IDSCNN忽略额外参数: {list(kwargs.keys())}")
            pass
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # 验证参数长度一致
        assert len(hidden_channels) == len(kernel_sizes) == len(pool_sizes), \
            "hidden_channels, kernel_sizes, pool_sizes长度必须一致"
        
        # 卷积层模块
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        current_length = sequence_length  # 使用新变量避免修改原参数
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(hidden_channels, kernel_sizes, pool_sizes)):
            
            # 卷积层
            conv_layer = nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size//2
            )
            self.conv_layers.append(conv_layer)
            
            # 批归一化
            if batch_norm:
                self.conv_layers.append(nn.BatchNorm1d(out_channels))
            
            # 激活函数
            self.conv_layers.append(nn.ReLU())
            
            # 池化层
            self.conv_layers.append(nn.MaxPool1d(kernel_size=pool_size))
            
            # Dropout
            self.conv_layers.append(nn.Dropout(dropout_rate))
            
            # 更新通道数和序列长度
            in_channels = out_channels
            current_length = current_length // pool_size
        
        # 计算全连接层输入大小
        self.conv_output_size = in_channels * current_length
        
        # 全连接层模块
        self.fc_layers = nn.ModuleList()
        in_features = self.conv_output_size
        
        for out_features in fc_sizes:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        
        # 输出层
        self.output_layer = nn.Linear(in_features, num_classes)
        
        # 初始化权重
        self._initialize_weights()
        
        # 打印模型信息（可选）
        # print(f"IDSCNN初始化完成: input_channels={input_channels}, "
        #       f"sequence_length={sequence_length}, num_classes={num_classes}")
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 确保输入形状正确 [batch, channels, sequence]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 卷积层
        for layer in self.conv_layers:
            x = layer(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        for layer in self.fc_layers:
            x = layer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x
    
    def get_feature_importance(self, x):
        """获取特征重要性（用于可解释性）"""
        features = []
        
        # 保存中间特征
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                # 计算卷积层的特征重要性（平均激活）
                importance = torch.mean(torch.abs(x), dim=(0, 2))
                features.append(importance.cpu().detach().numpy())
        
        return np.concatenate(features) if features else np.array([])
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_channels': self.input_channels,
            'sequence_length': self.sequence_length,
            'num_classes': self.num_classes,
            'conv_output_size': self.conv_output_size,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


class MultiScaleCNN(IDSCNN):
    """多尺度CNN，捕捉不同时间尺度的特征"""
    
    def __init__(self, input_channels=1, sequence_length=100, num_classes=2,
                 scales=[3, 5, 7], base_channels=32, dropout_rate=0.3, **kwargs):
        
        # 调用父类初始化（但我们会覆盖大部分功能）
        super().__init__(input_channels, sequence_length, num_classes)
        
        # 忽略从父类继承的不需要的参数
        self.scales = scales
        self.num_scales = len(scales)
        self.dropout_rate = dropout_rate
        
        # 多尺度卷积分支
        self.scale_branches = nn.ModuleList()
        for kernel_size in scales:
            branch = nn.Sequential(
                nn.Conv1d(input_channels, base_channels, 
                         kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate),
                
                nn.Conv1d(base_channels, base_channels*2, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels*2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            )
            self.scale_branches.append(branch)
        
        # 计算合并后的特征大小
        branch_output_length = sequence_length // 4  # 两次池化
        self.combined_features = base_channels * 2 * branch_output_length * self.num_scales
        
        # 全连接层（覆盖父类的fc_layers）
        self.fc_layers = nn.Sequential(
            nn.Linear(self.combined_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # 移除父类中不需要的层
        self.conv_layers = nn.ModuleList()  # 清空父类的卷积层
        self.output_layer = None  # 父类的输出层不再使用
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 多尺度特征提取
        scale_features = []
        for branch in self.scale_branches:
            feature = branch(x)
            feature = feature.view(feature.size(0), -1)
            scale_features.append(feature)
        
        # 特征融合
        combined = torch.cat(scale_features, dim=1)
        
        # 分类
        output = self.fc_layers(combined)
        
        return output