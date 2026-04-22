#!/usr/bin/env python3
"""
专门评估3DMAE模型的脚本。

这个脚本只评估3DMAE模型，并将结果输出到txt文件中。
适用于快速评估单个模型的性能。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from evaluation.evaluate import evaluate_model


def load_3dmae_model(model_path: str, device: torch.device) -> nn.Module:
    """
    加载3DMAE降水反演模型。

    用户需要实现这个函数来加载训练好的3DMAE模型。

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
    print(f"加载3DMAE模型从: {model_path}")

    # 1. 创建模型架构
    # 假设您有一个 PrecipitationDecoder 类
    # from models.rain_decoder import create_precipitation_decoder
    # from models.mae import MAE3D, MAE3DConfig

    # 2. 加载预训练的MAE编码器
    # mae_checkpoint = torch.load(mae_checkpoint_path, map_location=device)
    # mae = MAE3D(mae_config)
    # mae.load_state_dict(mae_checkpoint['model_state_dict'])

    # 3. 创建降水解码器
    # model = create_precipitation_decoder(mae_encoder=mae, num_frames=6)

    # 4. 加载微调后的权重
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # 5. 设置模型为评估模式
    # model.eval()
    # model.to(device)

    # return model

    # 暂时返回一个占位符
    raise NotImplementedError("用户需要实现 load_3dmae_model 函数")


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
        inputs: 模型输入（雷达序列）

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


def save_results_to_txt(results: dict, output_path: str):
    """
    将评估结果保存到txt文件。

    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("3DMAE模型评估结果\n")
        f.write("=" * 70 + "\n\n")

        f.write("评估时间: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("\n")

        # 分类显示指标
        f.write("一、回归指标\n")
        f.write("-" * 50 + "\n")
        reg_metrics = ['ME', 'MAE', 'RMSE', 'CC', 'bias', 'relative_bias']
        for metric in reg_metrics:
            if metric in results:
                f.write(f"{metric:20s}: {results[metric]:.4f}\n")

        f.write("\n二、分类指标（临界成功指数CSI）\n")
        f.write("-" * 50 + "\n")
        csi_metrics = [m for m in results.keys() if m.startswith('CSI_')]
        for metric in sorted(csi_metrics):
            threshold = metric.split('_')[1]
            f.write(f"CSI_{threshold:4s} mm/h: {results[metric]:.4f}\n")

        f.write("\n三、不同降水强度的偏差\n")
        f.write("-" * 50 + "\n")
        bias_metrics = [m for m in results.keys() if m.startswith('bias_') and not m.endswith('_rel')]
        for metric in sorted(bias_metrics):
            intensity = metric.split('_')[1]
            f.write(f"{intensity:10s}降水偏差: {results[metric]:.4f}\n")

        f.write("\n四、所有指标\n")
        f.write("-" * 50 + "\n")
        for metric, value in sorted(results.items()):
            if isinstance(value, (int, float)):
                f.write(f"{metric:25s}: {value:.6f}\n")
            else:
                f.write(f"{metric:25s}: {value}\n")

    print(f"评估结果已保存到: {output_path}")


def print_results_summary(results: dict):
    """
    打印评估结果摘要。

    Args:
        results: 评估结果字典
    """
    print("\n" + "=" * 70)
    print("3DMAE模型评估结果摘要")
    print("=" * 70)

    print("\n主要指标:")
    print("-" * 50)
    print(f"平均误差 (ME)      : {results.get('ME', 'N/A'):.4f} mm/h")
    print(f"平均绝对误差 (MAE) : {results.get('MAE', 'N/A'):.4f} mm/h")
    print(f"均方根误差 (RMSE)  : {results.get('RMSE', 'N/A'):.4f} mm/h")
    print(f"相关系数 (CC)      : {results.get('CC', 'N/A'):.4f}")

    print("\n临界成功指数 (CSI):")
    print("-" * 50)
    csi_metrics = [m for m in results.keys() if m.startswith('CSI_')]
    for metric in sorted(csi_metrics):
        threshold = metric.split('_')[1]
        print(f"  CSI_{threshold} mm/h: {results[metric]:.4f}")


def main():
    """主函数：运行3DMAE模型评估。"""
    print("=" * 70)
    print("3DMAE模型评估工具")
    print("=" * 70)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = f"./evaluation_results_3dmae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")

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
        # 步骤2: 加载3DMAE模型
        # ============================================================
        print("\n" + "="*70)
        print("步骤2: 加载3DMAE模型")
        print("="*70)

        # 您的3DMAE模型路径
        model_path = "./checkpoints/precipitation/best_model.pt"  # 修改为您的模型路径

        model = load_3dmae_model(model_path, device)
        print(f"3DMAE模型加载完成")

        # ============================================================
        # 步骤3: 评估模型
        # ============================================================
        print("\n" + "="*70)
        print("步骤3: 评估模型")
        print("="*70)

        metrics = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            prediction_fn=custom_prediction_fn,
            verbose=True
        )

        # ============================================================
        # 步骤4: 显示和保存结果
        # ============================================================
        print("\n" + "="*70)
        print("步骤4: 保存评估结果")
        print("="*70)

        # 打印结果摘要
        print_results_summary(metrics)

        # 保存详细结果到txt文件
        txt_file = os.path.join(output_dir, "3dmae_evaluation_results.txt")
        save_results_to_txt(metrics, txt_file)

        # 保存原始数据到JSON文件（可选）
        import json
        json_file = os.path.join(output_dir, "3dmae_evaluation_results.json")
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"原始数据已保存到: {json_file}")

        print("\n" + "="*70)
        print("评估完成！")
        print("="*70)
        print(f"评估结果已保存到: {output_dir}")
        print(f"1. 文本格式结果: {txt_file}")
        print(f"2. JSON格式数据: {json_file}")

    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请按照脚本中的注释提示，填充相应的函数实现。")
        print("主要需要实现的函数:")
        print("  1. load_3dmae_model() - 加载您的3DMAE模型")
        print("  2. create_test_data_loader() - 创建测试数据加载器")
        print("\n如果您需要自定义预测函数，请实现 custom_prediction_fn()")
    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()