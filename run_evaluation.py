#!/usr/bin/env python3
"""
运行模型评估的完整示例。

这个脚本演示如何使用评估工具来：
1. 加载训练好的模型
2. 在测试集上评估模型性能
3. 生成对比实验结果表（Table 1）
4. 生成消融实验结果表（Table 2）

用户需要填充以下部分：
1. 加载自己的模型
2. 准备测试数据加载器
3. 定义对比模型和消融模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime

from evaluation.evaluate import ModelEvaluator

def load_your_model(model_path: str, device: torch.device) -> nn.Module:
    """
    加载您的3DMAEPP模型。

    用户需要实现这个函数来加载自己的训练好的模型。

    Args:
        model_path: 模型检查点路径
        device: torch设备

    Returns:
        加载的PyTorch模型
    """
    # ============================================================
    # 用户需要填充这部分代码
    # ============================================================

    # 示例代码（需要替换为您的实际模型加载代码）
    print(f"加载模型从: {model_path}")

    # 1. 创建模型架构
    # 假设您有一个 PrecipitationDecoder 类
    # from models.rain_decoder import PrecipitationDecoder

    # 2. 加载权重
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # 3. 设置模型为评估模式
    # model.eval()
    # model.to(device)

    # return model

    # 暂时返回一个占位符
    raise NotImplementedError("用户需要实现 load_your_model 函数")


def load_comparison_models(device: torch.device) -> dict:
    """
    加载对比模型（RandomForest, UNet, PVT, DPT, 3DQPE, OpenSTL）。

    用户需要实现这个函数来加载所有对比模型。

    Args:
        device: torch设备

    Returns:
        字典: {模型名称: 模型实例}
    """
    # ============================================================
    # 用户需要填充这部分代码
    # ============================================================

    models = {}

    # 示例：加载各个对比模型
    # models['RandomForest'] = load_random_forest_model()
    # models['UNet'] = load_unet_model()
    # models['PVT'] = load_pvt_model()
    # models['DPT'] = load_dpt_model()
    # models['3DQPE'] = load_3dqpe_model()
    # models['OpenSTL'] = load_openstl_model()

    # 将模型移动到设备并设置为评估模式
    # for name, model in models.items():
    #     if hasattr(model, 'to'):
    #         model.to(device)
    #     if hasattr(model, 'eval'):
    #         model.eval()

    # return models

    # 暂时返回一个占位符
    raise NotImplementedError("用户需要实现 load_comparison_models 函数")


def load_ablation_models(base_model: nn.Module, device: torch.device) -> dict:
    """
    加载消融实验模型。

    用户需要实现这个函数来加载消融实验的模型变体。

    Args:
        base_model: 基础模型（3DMAEPP）
        device: torch设备

    Returns:
        字典: {消融模型名称: 模型实例}
    """
    # ============================================================
    # 用户需要填充这部分代码
    # ============================================================

    ablation_models = {}

    # 示例：创建不同的消融模型变体
    # 1. 没有3DMAE预训练的模型
    # ablation_models['w/o 3DMAE'] = create_model_without_pretraining()

    # 2. 没有时序融合的模型（单帧输入）
    # ablation_models['w/o Temporal'] = create_model_without_temporal_fusion()

    # 3. 没有分类头的模型
    # ablation_models['w/o Classification'] = create_model_without_classification_head()

    # 4. 没有站点损失的模型
    # ablation_models['w/o StationLoss'] = create_model_without_station_loss()

    # 将模型移动到设备并设置为评估模式
    # for name, model in ablation_models.items():
    #     model.to(device)
    #     model.eval()

    # return ablation_models

    # 暂时返回一个占位符
    raise NotImplementedError("用户需要实现 load_ablation_models 函数")


def create_test_data_loader(batch_size: int = 4) -> DataLoader:
    """
    创建测试数据加载器。

    用户需要实现这个函数来准备测试数据。

    Args:
        batch_size: 批次大小

    Returns:
        PyTorch DataLoader
    """
    # ============================================================
    # 用户需要填充这部分代码
    # ============================================================

    # 示例代码（需要替换为您的实际数据加载代码）
    # from data.dataloader import FinetuneDataset

    # 1. 创建测试数据集
    # test_dataset = FinetuneDataset(
    #     data_paths=['/path/to/your/test_data.h5'],
    #     radar_height_layers=[0, 1, 2, 3, 4, 5],
    #     spatial_size=(700, 900),
    #     target_minutes=[0, 30],
    #     history_frames=6,
    #     frame_interval=12,
    # )

    # 2. 创建数据加载器
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    # return test_loader

    # 暂时返回一个占位符
    raise NotImplementedError("用户需要实现 create_test_data_loader 函数")


def custom_prediction_fn(model, inputs):
    """
    自定义预测函数（如果模型输出不是标准格式）。

    如果您的模型返回的预测格式不是标准的 (B, 1, H, W) 或字典格式，
    需要实现这个函数来提取预测结果。

    Args:
        model: PyTorch模型
        inputs: 模型输入

    Returns:
        预测结果张量 (B, H, W) 或 (B, 1, H, W)
    """
    # ============================================================
    # 用户需要填充这部分代码（如果需要）
    # ============================================================

    # 默认行为：直接调用模型
    outputs = model(inputs)

    # 如果模型返回字典，提取回归输出
    if isinstance(outputs, dict):
        if 'reg_output' in outputs:
            return outputs['reg_output']
        elif 'prediction' in outputs:
            return outputs['prediction']
        else:
            # 尝试找到第一个张量输出
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    return value

    # 如果模型返回元组，假设第一个元素是预测
    elif isinstance(outputs, tuple):
        return outputs[0]

    # 否则直接返回
    return outputs


def main():
    """主函数：运行完整的模型评估流程。"""
    print("=" * 70)
    print("3DMAE模型评估工具")
    print("=" * 70)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = f"./evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")

    # 初始化评估器
    evaluator = ModelEvaluator(output_dir=output_dir)

    try:
        # ============================================================
        # 步骤1: 准备测试数据
        # ============================================================
        print("\n" + "="*70)
        print("步骤1: 准备测试数据")
        print("="*70)

        test_loader = create_test_data_loader(batch_size=4)
        print(f"测试数据加载器创建完成")
        print(f"测试批次数量: {len(test_loader)}")

        # ============================================================
        # 步骤2: 加载3DMAEPP模型（您的主要模型）
        # ============================================================
        print("\n" + "="*70)
        print("步骤2: 加载3DMAEPP模型")
        print("="*70)

        # 您的3DMAEPP模型路径
        your_model_path = "./checkpoints/precipitation/best_model.pt"  # 修改为您的模型路径

        model_3dmaepp = load_your_model(your_model_path, device)

        # 评估3DMAEPP模型
        metrics_3dmaepp = evaluator.evaluate(
            model_name='3DMAEPP',
            model=model_3dmaepp,
            data_loader=test_loader,
            device=device,
            prediction_fn=custom_prediction_fn,
            verbose=True
        )

        # ============================================================
        # 步骤3: 加载并评估对比模型
        # ============================================================
        print("\n" + "="*70)
        print("步骤3: 评估对比模型")
        print("="*70)

        comparison_models = load_comparison_models(device)

        for name, model in comparison_models.items():
            evaluator.evaluate(
                model_name=name,
                model=model,
                data_loader=test_loader,
                device=device,
                prediction_fn=custom_prediction_fn,
                verbose=True
            )

        # ============================================================
        # 步骤4: 生成对比实验结果表（对应论文表1）
        # ============================================================
        print("\n" + "="*70)
        print("步骤4: 生成对比实验结果表")
        print("="*70)

        comparison_df = evaluator.compare_models(save_to_file=True)

        print("\n对比实验结果表:")
        print("-" * 70)
        print(comparison_df.to_string())

        # ============================================================
        # 步骤5: 进行消融实验（对应论文表2）
        # ============================================================
        print("\n" + "="*70)
        print("步骤5: 进行消融实验")
        print("="*70)

        ablation_models = load_ablation_models(model_3dmaepp, device)

        ablation_df = evaluator.ablation_study(
            base_model_name='3DMAEPP',
            ablation_models=ablation_models,
            data_loader=test_loader,
            device=device,
            save_to_file=True
        )

        print("\n消融实验结果表:")
        print("-" * 70)
        print(ablation_df.to_string())

        # ============================================================
        # 步骤6: 生成LaTeX表格用于论文
        # ============================================================
        print("\n" + "="*70)
        print("步骤6: 生成LaTeX表格")
        print("="*70)

        # 对比实验结果表（表1）
        comparison_latex = comparison_df.to_latex(
            float_format="%.3f",
            caption="对比实验结果（ME、MAE、RMSE单位为mm/h，CC为相关系数，CSI为临界成功指数）",
            label="tab:comparison_results"
        )

        # 消融实验结果表（表2）
        ablation_latex = ablation_df.to_latex(
            float_format="%.3f",
            caption="消融实验结果（括号内为相对基线的变化百分比）",
            label="tab:ablation_results"
        )

        # 保存LaTeX表格
        latex_file = os.path.join(output_dir, "results_tables.tex")
        with open(latex_file, 'w') as f:
            f.write("\\subsection{对比实验结果}\n")
            f.write(comparison_latex)
            f.write("\n\\subsection{消融实验结果}\n")
            f.write(ablation_latex)

        print(f"LaTeX表格已保存到: {latex_file}")

        # ============================================================
        # 步骤7: 生成结果摘要
        # ============================================================
        print("\n" + "="*70)
        print("步骤7: 结果摘要")
        print("="*70)

        # 提取关键指标
        key_metrics = ['ME', 'MAE', 'RMSE', 'CC', 'CSI_0.1', 'CSI_1.0', 'CSI_5.0', 'CSI_10.0']

        summary_file = os.path.join(output_dir, "results_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("3DMAE模型评估结果摘要\n")
            f.write("=" * 50 + "\n\n")

            f.write("一、对比实验结果（3DMAEPP vs 其他模型）\n")
            f.write("-" * 50 + "\n")
            for metric in key_metrics:
                if metric in comparison_df.columns:
                    best_value = comparison_df[metric].max() if metric == 'CC' else comparison_df[metric].min()
                    best_model = comparison_df[comparison_df[metric] == best_value].index[0]
                    f.write(f"{metric}: 最佳模型 = {best_model} ({best_value:.3f})\n")

            f.write("\n二、消融实验结果\n")
            f.write("-" * 50 + "\n")
            for model_name in ablation_df.index:
                if model_name != '3DMAEPP':
                    f.write(f"\n{model_name}:\n")
                    for metric in key_metrics:
                        if metric in ablation_df.columns:
                            value = ablation_df.loc[model_name, metric]
                            rel_col = f"{metric}_rel"
                            if rel_col in ablation_df.columns:
                                rel_change = ablation_df.loc[model_name, rel_col]
                                if rel_change > 0:
                                    change_str = f"(↑{rel_change*100:.1f}%)"
                                else:
                                    change_str = f"(↓{abs(rel_change)*100:.1f}%)"
                                f.write(f"  {metric}: {value:.3f} {change_str}\n")
                            else:
                                f.write(f"  {metric}: {value:.3f}\n")

        print(f"结果摘要已保存到: {summary_file}")

        print("\n" + "="*70)
        print("评估完成！")
        print("="*70)
        print(f"所有结果已保存到: {output_dir}")
        print(f"1. 对比实验结果表: {output_dir}/model_comparison_*.csv")
        print(f"2. 消融实验结果表: {output_dir}/ablation_study_*.csv")
        print(f"3. LaTeX表格: {output_dir}/results_tables.tex")
        print(f"4. 结果摘要: {output_dir}/results_summary.txt")

    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请按照脚本中的注释提示，填充相应的函数实现。")
        print("主要需要实现的函数:")
        print("  1. load_your_model() - 加载您的3DMAEPP模型")
        print("  2. load_comparison_models() - 加载对比模型")
        print("  3. load_ablation_models() - 加载消融模型")
        print("  4. create_test_data_loader() - 创建测试数据加载器")
    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()