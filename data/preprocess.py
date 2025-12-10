# data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import warnings
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

def load_cicids2017(data_path, sample_frac=0.3):
    """加载CICIDS2017数据"""
    print("\n=== 开始加载CICIDS2017数据 ===")
    
    # 文件列表（根据您的实际文件调整）
    files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv", 
        "Wednesday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    ]
    
    # 检查存在的文件
    existing_files = []
    for f in files:
        file_path = os.path.join(data_path, f)
        if os.path.exists(file_path):
            existing_files.append(f)
    
    print(f"找到 {len(existing_files)}/{len(files)} 个数据文件")
    
    if len(existing_files) == 0:
        raise ValueError(f"在 {data_path} 中没有找到CICIDS2017数据文件")
    
    # 读取所有文件
    dfs = []
    for file in tqdm(existing_files, desc="加载数据文件"):
        file_path = os.path.join(data_path, file)
        
        try:
            # 分块读取大文件
            chunk_list = []
            for chunk in pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=100000):
                chunk.columns = chunk.columns.str.strip()
                if 'Label' not in chunk.columns:
                    continue
                chunk['Label'] = chunk['Label'].astype(str).str.upper().str.strip()
                chunk_list.append(chunk)
            
            if chunk_list:
                df = pd.concat(chunk_list, ignore_index=True)
                if sample_frac < 1.0:
                    df = df.sample(frac=sample_frac, random_state=42)
                dfs.append(df)
                print(f"  已加载 {file}: {len(df)} 行")
                
        except Exception as e:
            print(f"  加载文件 {file} 时出错: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("没有成功加载任何数据文件")
    
    # 合并所有数据
    data = pd.concat(dfs, ignore_index=True)
    print(f"合并后总数据量: {len(data)} 行")
    
    # 数据清理
    print("清理数据...")
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0)
    data = data.drop_duplicates()
    
    # 转换数据类型节省内存
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = data[col].astype('float32')
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = data[col].astype('int32')
    
    # 处理标签
    if 'Label' not in data.columns:
        raise ValueError("数据中未找到'Label'列")
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    
    # 分离特征和标签
    features = data.drop('Label', axis=1)
    labels = data['Label'].values
    
    # 处理无限值和NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    
    # 标准化特征
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"数据处理完成: {features_scaled.shape[0]} 样本, {features_scaled.shape[1]} 特征")
    print(f"类别分布: {np.bincount(labels)}")
    
    return features_scaled, labels, label_encoder

def reshape_for_cnn(features, sequence_length=100):
    """
    将特征向量重塑为CNN输入格式
    """
    n_samples, n_features = features.shape
    
    if n_features >= sequence_length:
        # 截断
        reshaped = features[:, :sequence_length]
    else:
        # 零填充
        reshaped = np.zeros((n_samples, sequence_length))
        reshaped[:, :n_features] = features
    
    # 添加通道维度
    reshaped = reshaped.reshape(n_samples, 1, sequence_length)
    
    return reshaped

def prepare_cnn_datasets(X_train, X_val, X_test, y_train, y_val, y_test, 
                         sequence_length=100, batch_size=64):
    """准备CNN数据集"""
    
    # 重塑数据
    X_train_cnn = reshape_for_cnn(X_train, sequence_length)
    X_val_cnn = reshape_for_cnn(X_val, sequence_length)
    X_test_cnn = reshape_for_cnn(X_test, sequence_length)
    
    # 转换为Tensor
    X_train_t = torch.FloatTensor(X_train_cnn)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_cnn)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test_cnn)
    y_test_t = torch.LongTensor(y_test)
    
    # 创建TensorDataset
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

class NetworkTrafficDataset(Dataset):
    """网络流量数据集"""
    def __init__(self, features, labels, sequence_length=100):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length
        
        # 确保正确的形状
        if self.features.dim() == 2:
            self.features = self.features.unsqueeze(1)  # 添加通道维度
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]