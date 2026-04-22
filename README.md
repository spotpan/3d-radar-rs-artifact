# 3DMAE: 基于三维掩码自编码器的定量降水估计方法

本项目实现了一个基于三维掩码自编码器（3DMAE）的深度学习框架，用于从三维雷达回波数据反演小时累积降水量。该方法结合了自监督预训练、多帧时序融合与联合损失优化，能够有效利用大规模无标签雷达数据，提升降水反演的准确性和泛化能力。

## 项目概述

### 核心特性
- **3DMAE自监督预训练**: 使用80%掩码比例在无标签三维雷达数据上预训练，学习雷达回波的时空结构与垂直关联
- **多帧时序融合**: 通过逐点时间Transformer融合连续6帧雷达数据（间隔12分钟），捕捉降水系统的动态演化
- **双头解码器**: 同时输出降水强度回归结果和有雨概率分类结果
- **联合损失函数**: 结合全局网格分位数损失、站点邻域加权损失和分类损失，实现多模态监督
- **两阶段训练**: 先自监督预训练MAE编码器，再冻结编码器进行降水反演微调

### 模型架构
1. **3DMAE预训练模型**:
   - 编码器: 16层Transformer，768维，12头注意力
   - 解码器: 8层Transformer，512维，8头注意力
   - 输入: 6层高度雷达数据（700×900）
   - 输出: 重建的雷达数据

2. **降水反演模型**:
   - 特征提取: 冻结的3DMAE编码器
   - 时序融合: 逐点时间Transformer（2层，8头）
   - 解码器: 4阶段上采样卷积网络
   - 输出: 降水强度图（回归）和有雨概率图（分类）

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.12+ with CUDA 11.3
- NVIDIA GPU (建议A800 80GB或类似)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 数据准备
数据应存储为HDF5格式，结构如下：
```
/radar-rain/
  ├── radar: (N, 11, 700, 900) int8        # 11层高度雷达反射率
  ├── rain: (N, 700, 900) float16          # 插值降水网格
  ├── radar_valid: (N,) bool              # 雷达数据有效性标记
  ├── rain_valid: (N,) bool               # 降水数据有效性标记
  └── time: (N,) string                   # 时间字符串 (YYYYMMDDHHMM)
```

默认数据路径为 `/mnt/md1/hxc/guangdong/train_select/`，包含三个文件：
- `time_radar_rain_2022.h5`
- `time_radar_rain_2023.h5`
- `time_radar_rain_2025.h5`

### 运行测试
```bash
# 运行集成测试验证所有组件
python test_integration.py

# 测试数据加载器
python test_dataloader.py
```

## 使用指南

### 1. 3DMAE预训练
```bash
# 使用默认配置进行预训练
python training/pretrain.py

# 自定义训练参数
python training/pretrain.py \
  --num-epochs 200 \
  --batch-size 64 \
  --learning-rate 1.5e-4 \
  --experiment-name "mae_pretrain_exp"
```

### 2. 降水反演微调
```bash
# 使用预训练权重进行微调
python training/finetune.py \
  --mae-checkpoint ./checkpoints/mae_pretrain/best_model.pt \
  --num-epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --experiment-name "precipitation_finetune"
```

### 3. 配置文件
- `configs/pretrain_config.py`: 预训练配置
- `configs/finetune_config.py`: 微调配置

关键配置参数：
- 数据路径、高度层选择、时间对齐策略
- 模型架构参数（Transformer层数、注意力头数、嵌入维度等）
- 训练超参数（学习率、批次大小、损失权重等）

### 4. 模型评估

#### 4.1 完整评估（对比实验 + 消融实验）
```bash
# 运行完整的评估流程（生成对比实验和消融实验结果）
python run_evaluation.py
```

评估工具位于 `evaluation/` 目录，包含：
- `evaluate.py`: 主评估脚本，计算所有论文中的指标
- `README.md`: 详细使用说明

**主要功能**：
1. **对比实验**: 比较3DMAEPP与其他模型（RandomForest, UNet, PVT, DPT, 3DQPE, OpenSTL）
2. **消融实验**: 评估各组件贡献（移除3DMAE预训练、时序融合、分类头、站点损失）
3. **指标计算**: ME, MAE, RMSE, CC, CSI-0.1/1.0/5.0/10.0
4. **结果导出**: CSV, LaTeX, JSON格式

**使用步骤**：
1. 编辑 `run_evaluation.py`，填充以下函数：
   - `load_your_model()`: 加载您的3DMAEPP模型
   - `load_comparison_models()`: 加载对比模型
   - `load_ablation_models()`: 加载消融模型
   - `create_test_data_loader()`: 创建测试数据加载器
2. 运行评估脚本
3. 查看生成的表格和摘要

**输出文件**：
- `evaluation_results_*/`: 包含所有结果文件
- `model_comparison_*.csv`: 对比实验结果表（对应论文表1）
- `ablation_study_*.csv`: 消融实验结果表（对应论文表2）
- `results_tables.tex`: LaTeX格式表格
- `results_summary.txt`: 结果摘要

#### 4.2 快速评估（只评估3DMAE模型）
```bash
# 只评估3DMAE模型，结果输出到txt文件
python run_evaluation_3dmae.py
```

**主要功能**：
1. **专一性**: 只评估3DMAE模型，不涉及对比模型或消融实验
2. **简洁输出**: 结果以易读的txt格式输出
3. **完整指标**: 包含所有论文中的评估指标
4. **快速使用**: 只需填充2个函数即可使用

**使用步骤**：
1. 编辑 `run_evaluation_3dmae.py`，填充以下函数：
   - `load_3dmae_model()`: 加载您的3DMAE模型
   - `create_test_data_loader()`: 创建测试数据加载器
2. （可选）如果需要自定义预测格式，实现 `custom_prediction_fn()`
3. 运行评估脚本
4. 查看生成的txt报告

**输出文件**：
- `evaluation_results_3dmae_*/`: 包含评估结果
- `3dmae_evaluation_results.txt`: 格式化的文本报告（主要输出）
- `3dmae_evaluation_results.json`: 原始JSON数据

**选择建议**：
- **完整评估**: 需要生成论文表格、对比多个模型、进行消融分析时使用 `run_evaluation.py`
- **快速评估**: 只需评估单个3DMAE模型、快速查看性能指标时使用 `run_evaluation_3dmae.py`

## 项目结构
```
3DMAE/
├── data/                    # 数据加载和预处理
│   ├── dataloader.py       # 数据集类（RadarRainDataset, PretrainDataset, FinetuneDataset）
│   └── __init__.py
├── models/                 # 模型定义
│   ├── mae.py             # 3DMAE模型
│   ├── rain_decoder.py    # 降水反演解码器
│   ├── losses.py          # 损失函数
│   └── __init__.py
├── training/              # 训练脚本
│   ├── pretrain.py       # 预训练脚本
│   ├── finetune.py       # 微调脚本
│   ├── trainer.py        # 训练器基类
│   └── __init__.py
├── configs/               # 配置文件
│   ├── pretrain_config.py
│   └── finetune_config.py
├── evaluation/            # 模型评估工具
│   ├── evaluate.py       # 主评估脚本
│   ├── __init__.py
│   └── README.md         # 评估工具说明
├── scripts/               # 工具脚本
│   ├── explore_data.py   # 数据探索
│   └── check_metadata.py
├── utils/                 # 工具函数
│   └── metrics.py        # 评估指标计算
├── outputs/              # 输出目录
├── logs/                 # 训练日志
├── checkpoints/          # 模型检查点
├── test_integration.py   # 集成测试
├── test_dataloader.py    # 数据加载测试
├── run_evaluation.py     # 完整评估流程脚本
├── run_evaluation_3dmae.py # 3DMAE专用评估脚本
├── PROJECT_PLAN.md       # 项目计划文档
└── README.md            # 本文档
```

## 模型细节

### 3DMAE预训练
- **输入**: 6层高度雷达数据（700×900）
- **Patch划分**: 16×16空间块，每个块包含6个高度层
- **掩码策略**: 随机掩蔽80%的时空块
- **目标**: 重建被掩蔽的雷达像素值（MSE损失）
- **训练**: 200个epoch，余弦学习率调度，AdamW优化器

### 降水反演
- **输入序列**: 6帧连续雷达数据（目标时刻及前5个时次，间隔12分钟）
- **特征提取**: 每帧独立通过冻结的3DMAE编码器
- **时序融合**: 逐点时间Transformer建模帧间依赖
- **解码器**: 4阶段上采样（2倍×4），跳跃连接保留细节
- **输出头**: 
  - 回归头: 降水强度（mm/h），最后30% epoch使用Softplus激活
  - 分类头: 有雨概率（Sigmoid激活），阈值0.1 mm/h

### 损失函数
总损失: `L_total = λ_global * L_global + λ_point * L_point + λ_cls * L_cls`

1. **全局网格损失** (`λ_global=1.0`): 预测网格与插值网格之间的分位数损失（q=0.9）
2. **站点邻域损失** (`λ_point=0.5`): 站点观测与邻域加权平均预测之间的分位数损失
3. **分类损失** (`λ_cls=0.3`): 有雨/无雨二元交叉熵损失

## 评估指标
- **回归指标**: 平均误差（ME）、平均绝对误差（MAE）、均方根误差（RMSE）、相关系数（CC）
- **分类指标**: 临界成功指数（CSI）在0.1、1、5、10 mm/h阈值
- **强降水检测**: CSI-5、CSI-10评估强降水事件识别能力

## 实验结果
根据论文结果，本文方法在多个指标上优于对比模型：
- RMSE: 2.581（最佳）
- CC: 72.281%（最佳）
- CSI-10: 0.411（最佳）

消融实验验证了各模块的有效性：
- 移除3DMAE预训练: RMSE↑0.447，CSI-10↓0.042
- 移除时序融合: RMSE↑0.373，CSI-10↓0.029
- 移除分类头: CSI-0.1↓0.031，CSI-1↓0.048
- 移除站点损失: RMSE↑0.284，CSI-10↓0.039

## 扩展与定制

### 使用自定义数据
1. 准备HDF5文件，遵循上述数据格式
2. 修改`configs/`中的`data_paths`配置
3. 调整`spatial_size`和`radar_height_layers`参数

### 修改模型架构
- 调整Transformer层数、注意力头数、嵌入维度
- 修改解码器通道数、上采样策略
- 自定义损失函数权重和阈值

### 添加新功能
- 支持更多气象变量反演
- 集成其他时空融合模块
- 添加不确定性估计

## 故障排除

### 常见问题
1. **内存不足**: 减小批次大小，使用梯度累积，启用混合精度训练
2. **训练不稳定**: 调整学习率，使用梯度裁剪，检查数据归一化
3. **收敛缓慢**: 检查学习率调度，增加预训练epoch，调整损失权重

### 调试建议
- 运行`test_integration.py`验证所有组件
- 使用小批次和简化模型进行快速调试
- 监控训练日志和TensorBoard可视化

## 开始使用指南

### 第一步：环境准备
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 检查数据
python scripts/check_metadata.py

# 3. 运行组件测试
python test_integration.py
```

### 第二步：模型训练
```bash
# 1. 3DMAE预训练（使用默认配置）
python training/pretrain.py

# 2. 降水反演微调（需要预训练权重）
python training/finetune.py --mae-checkpoint ./checkpoints/mae_pretrain/best_model.pt
```

### 第三步：模型评估

#### 选项A：完整评估（对比实验 + 消融实验）
1. **准备测试数据**: 确保测试集HDF5文件格式正确
2. **编辑评估脚本**: 打开`run_evaluation.py`，填充四个关键函数：
   - `load_your_model()`: 加载训练好的3DMAEPP模型
   - `load_comparison_models()`: 加载对比模型（RandomForest, UNet等）
   - `load_ablation_models()`: 加载消融实验模型变体
   - `create_test_data_loader()`: 创建测试数据加载器
3. **运行评估**:
   ```bash
   python run_evaluation.py
   ```
4. **查看结果**: 结果保存在`evaluation_results_*/`目录，包含：
   - 对比实验结果表（CSV和LaTeX格式）
   - 消融实验结果表
   - 结果摘要报告

#### 选项B：快速评估（只评估3DMAE模型）
1. **准备测试数据**: 确保测试集HDF5文件格式正确
2. **编辑评估脚本**: 打开`run_evaluation_3dmae.py`，填充两个关键函数：
   - `load_3dmae_model()`: 加载训练好的3DMAE模型
   - `create_test_data_loader()`: 创建测试数据加载器
3. **运行评估**:
   ```bash
   python run_evaluation_3dmae.py
   ```
4. **查看结果**: 结果保存在`evaluation_results_3dmae_*/`目录，包含：
   - `3dmae_evaluation_results.txt`: 格式化的文本报告
   - `3dmae_evaluation_results.json`: 原始JSON数据

**选择建议**：
- **完整评估**: 需要生成论文表格、对比多个模型、进行消融分析
- **快速评估**: 只需评估单个3DMAE模型、快速查看性能指标

### 第四步：生成论文图表
1. **使用LaTeX表格**: `evaluation_results_*/results_tables.tex`可直接插入论文
2. **自定义可视化**: 基于CSV结果文件创建自定义图表
3. **结果分析**: 参考`results_summary.txt`进行结果解读

### 快速验证
如果只想测试评估工具的基本功能，可以使用虚拟数据：
```python
# 在Python中快速测试评估指标计算
from utils.metrics import calculate_all_metrics
import torch

# 创建虚拟数据
pred = torch.randn(2, 1, 100, 100).abs()
target = torch.randn(2, 1, 100, 100).abs()

# 计算指标
metrics = calculate_all_metrics(pred, target)
print(metrics)
```

## 参考文献
- 本文基于"基于三维掩码自编码器的定量降水估计方法"论文实现
- 参考Vision Transformer（ViT）和Masked Autoencoder（MAE）架构
- 采用分位数损失处理降水零膨胀分布

## 许可证
本项目仅供研究使用。具体许可证信息请参考相关文档。

## 联系方式
如有问题或建议，请通过项目仓库提交Issue。

---

*本项目实现了完整的3DMAE降水反演系统，包含数据加载、模型定义、训练脚本和测试工具，可直接用于科研和生产环境。*