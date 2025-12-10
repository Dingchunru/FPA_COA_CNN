# 🚀 FPA-COA-CNN 入侵检测算法

一个基于混合优化算法（花授粉算法 FPA + 杜鹃优化算法 COA）和卷积神经网络（CNN）的智能入侵检测系统，专门用于恶意网络流量数据分析。

## 📊 项目简介

本项目实现了一个完整的入侵检测管道，包含特征选择、模型训练和性能评估三个阶段。通过FPA和COA混合优化算法自动选择最优特征子集，然后使用一维CNN对网络流量进行分类，实现了高效准确的入侵检测。

### ✨ 主要特性

- **混合优化特征选择**：FPA和COA协同工作，自动选择最优特征子集
- **GPU加速训练**：支持CUDA加速，大幅提升训练速度
- **自适应学习**：动态调整算法权重，平衡探索与开发
- **完整评估体系**：多种评估指标，包括准确率、F1分数、AUC-ROC等
- **可视化分析**：提供混淆矩阵、ROC曲线等可视化工具

## 🏗️ 系统架构

```
FPA-COA-CNN 入侵检测系统
├── 数据预处理层
│   └── CICIDS2017数据集加载与清洗
├── 特征选择层
│   ├── 花授粉算法 (FPA) - 全局探索
│   ├── 杜鹃优化算法 (COA) - 局部优化
│   └── 混合优化策略 - 协同工作
├── 模型训练层
│   └── 一维CNN分类器
├── 评估层
│   ├── 性能指标计算
│   └── 可视化分析
└── 结果输出层
    └── 模型保存与结果导出
```

## 📁 项目结构

```
FPA-COA-CNN-IDS/
├── algorithms/           # 优化算法实现
│   ├── fpa.py           # 花授粉算法
│   ├── coa.py           # 杜鹃优化算法
│   └── hybrid_optimizer.py  # 混合优化器
├── models/              # 模型定义
│   ├── cnn_model.py     # CNN模型
│   └── model_utils.py   # 模型工具
├── data/                # 数据处理
│   ├── preprocess.py    # 数据预处理
│   └── CICIDS2017/      # 数据集目录
├── utils/               # 工具函数
│   ├── metrics.py       # 评估指标
│   └── visualization.py # 可视化工具
├── config.py            # 配置文件
├── main.py              # 主程序
├── requirements.txt     # 依赖包
└── README.md           # 项目说明
```

## ⚙️ 环境要求

### 基础环境
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (推荐，用于GPU加速)

### 依赖安装

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 2. 安装依赖
pip install -r requirements.txt
```

### requirements.txt 内容

```txt
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0

# 数据平衡
imbalanced-learn>=0.10.0

# 进度显示
tqdm>=4.65.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 其他工具
pyyaml>=6.0
```

## 📥 数据集准备

### 使用CICIDS2017数据集

1. **下载数据集**：
   - 访问 [CICIDS2017官方网站](https://www.unb.ca/cic/datasets/ids-2017.html)
   - 填写申请表单获取下载链接
   - 或从Kaggle下载：https://www.kaggle.com/datasets/cicdataset/cicids2017

2. **数据集结构**：
   ```
   data/CICIDS2017/
   ├── Monday-WorkingHours.pcap_ISCX.csv
   ├── Tuesday-WorkingHours.pcap_ISCX.csv
   ├── Wednesday-WorkingHours.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   └── Friday-WorkingHours-Morning.pcap_ISCX.csv
   ```

3. **自动下载脚本**：
   ```bash
   python download_cicids2017.py
   ```

### 使用合成数据（测试用）

如果无法获取真实数据集，可以使用合成数据测试：

```python
python generate_test_data.py
```

## 🚀 快速开始

### 1. 基础运行

```bash
# 使用默认配置运行完整流程
python main.py
```

### 2. 使用GPU加速

```bash
# 检查GPU状态
python check_gpu.py

# 启用GPU运行
python main.py --use-gpu
```

### 3. 参数配置

通过修改 `config.py` 调整参数：

```python
# 主要配置项
config = Config(
    DATA_PATH = "data/CICIDS2017",      # 数据路径
    SAMPLE_FRACTION = 0.3,              # 数据采样比例
    USE_GPU = True,                     # 启用GPU
    BATCH_SIZE = 512,                   # 批大小
    POP_SIZE = 30,                      # 优化算法种群大小
    ITER_MAX = 20                       # 最大迭代次数
)
```

### 4. 分步执行

```python
# 1. 仅加载和预处理数据
from data.preprocess import load_cicids2017
features, labels, label_encoder = load_cicids2017("data/CICIDS2017")

# 2. 仅运行特征选择
from algorithms.hybrid_optimizer import HybridFPA_COA_Optimizer
optimizer = HybridFPA_COA_Optimizer(objective_func, dim=78)
best_features, fitness = optimizer.run()

# 3. 仅训练CNN模型
from models.cnn_model import IDSCNN
model = IDSCNN(input_channels=1, sequence_length=100, num_classes=2)
# ... 训练代码
```

## 📈 算法详解

### 花授粉算法 (FPA)

FPA模拟开花植物的授粉过程，包含两种授粉方式：

1. **全局授粉（异花授粉）**：
   - 使用Lévy飞行进行长距离搜索
   - 公式：`x_i^{t+1} = x_i^t + L(λ) * (g_best - x_i^t)`
   
2. **局部授粉（自花授粉）**：
   - 在局部范围内进行精细搜索
   - 公式：`x_i^{t+1} = x_i^t + ε * (x_j^t - x_k^t)`

### 杜鹃优化算法 (COA)

COA模拟杜鹃鸟的巢寄生行为：

1. **产卵行为**：在宿主巢中产卵（生成新解）
2. **宿主发现**：有一定概率发现并抛弃外来蛋
3. **Lévy飞行**：进行长距离搜索

### 混合优化策略

FPA和COA通过以下方式协同工作：

1. **动态权重调整**：根据算法性能自动调整权重
2. **种群迁移**：定期交换两个算法的种群个体
3. **精英选择**：保留最优个体，加速收敛

### CNN架构

```python
IDSCNN(
    input_channels=1,           # 输入通道
    sequence_length=100,        # 序列长度
    num_classes=2,              # 输出类别
    hidden_channels=[64,128,256], # 卷积层通道
    kernel_sizes=[3,3,3],       # 卷积核大小
    fc_sizes=[256,128]          # 全连接层
)
```

## 🔧 配置参数

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 512 | 批处理大小 |
| learning_rate | 0.001 | 学习率 |
| epochs | 100 | 训练轮数 |
| patience | 10 | 早停耐心值 |
| mixed_precision | True | 混合精度训练 |

### 优化器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| pop_size | 30 | 种群大小 |
| iter_max | 20 | 最大迭代次数 |
| elite_rate | 0.1 | 精英保留比例 |
| migration_rate | 0.2 | 种群迁移比例 |

### 特征选择配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| threshold | 0.5 | 特征选择阈值 |
| min_features | 10 | 最小特征数 |
| max_features_ratio | 0.8 | 最大特征比例 |

## 📊 性能评估

### 评估指标

1. **准确率 (Accuracy)**: 整体分类准确率
2. **F1分数**: 宏平均和微平均F1分数
3. **精确率/召回率**: 各类别的精确率和召回率
4. **AUC-ROC**: ROC曲线下面积
5. **推理速度**: 毫秒/样本，样本/秒

### 结果输出

运行完成后，结果将保存在 `results/` 目录：

```
results/
├── test_results.npz          # 测试结果数据
├── feature_selection.npz     # 特征选择结果
├── config_summary.json       # 配置摘要
└── best_fpa_coa_cnn.pth     # 最佳模型权重
```

### 可视化结果

```python
# 生成混淆矩阵
python plot_results.py --type confusion_matrix

# 生成ROC曲线
python plot_results.py --type roc_curve

# 生成训练历史图
python plot_results.py --type training_history
```

## 🎯 使用示例

### 示例1：完整流程

```python
from main import FPACOACNNTrainer
from config import Config

# 加载配置
config = Config(USE_GPU=True, SAMPLE_FRACTION=0.5)

# 创建训练器
trainer = FPACOACNNTrainer(config)

# 运行完整流程
results = trainer.run()

print(f"准确率: {results['accuracy']:.4f}")
print(f"F1分数: {results['f1_macro']:.4f}")
```

### 示例2：自定义特征选择

```python
from algorithms.hybrid_optimizer import HybridFPA_COA_Optimizer

# 自定义目标函数
def custom_objective(feature_subset):
    selected = np.where(feature_subset > 0.5)[0]
    # 计算适应度...
    return fitness

# 创建优化器
optimizer = HybridFPA_COA_Optimizer(
    objective_func=custom_objective,
    dim=78,
    pop_size=40,
    iter_max=30
)

# 运行优化
best_solution, best_fitness = optimizer.run(bounds=(0, 1))
```

### 示例3：模型推理

```python
import torch
from models.cnn_model import IDSCNN

# 加载训练好的模型
model = IDSCNN(input_channels=1, sequence_length=100, num_classes=2)
model.load_state_dict(torch.load("model/best_fpa_coa_cnn.pth"))
model.eval()

# 推理
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小batch_size
   python main.py --batch-size 256
   
   # 启用梯度累积
   python main.py --gradient-accumulation 2
   ```

2. **数据集加载失败**
   ```bash
   # 使用合成数据测试
   python main.py --use-synthetic-data
   
   # 检查数据路径
   python check_dataset.py --path data/CICIDS2017
   ```

3. **训练速度慢**
   ```bash
   # 启用混合精度训练
   python main.py --mixed-precision
   
   # 增加num_workers
   python main.py --num-workers 8
   ```

4. **特征选择时间过长**
   ```bash
   # 减少种群大小和迭代次数
   python main.py --pop-size 20 --iter-max 10
   
   # 启用快速评估模式
   python main.py --fast-eval
   ```

### 调试模式

```bash
# 启用详细日志
python main.py --verbose

# 仅运行数据预处理
python main.py --step preprocess

# 仅运行特征选择
python main.py --step feature_selection

# 仅运行模型训练
python main.py --step training
```

## 📚 参考文献

1. **CICIDS2017数据集**:
   - Sharafaldin, I., et al. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"

2. **花授粉算法**:
   - Yang, X. S. (2012). "Flower pollination algorithm for global optimization"

3. **杜鹃优化算法**:
   - Rajabioun, R. (2011). "Cuckoo optimization algorithm"

4. **CNN用于入侵检测**:
   - Kim, J., et al. (2016). "Long short-term memory recurrent neural network classifier for intrusion detection"

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 代码规范

- 遵循 PEP 8 编码规范
- 添加适当的注释和文档
- 编写单元测试
- 更新 README.md 文档

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目地址：https://github.com/Dingchunru/FPA_COA_CNN
- 问题反馈：https://github.com/Dingchunru/FPA_COA_CNN/issues
- 邮箱：2022211636@bupt.cn
## 🙏 致谢

- 感谢 Canadian Institute for Cybersecurity 提供的 CICIDS2017 数据集
- 感谢所有为本项目做出贡献的开发者
- 特别感谢 PyTorch 和 scikit-learn 社区的优秀工具

---

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！ ⭐**

**📈 祝您使用愉快，检测准确率高达99%！**
