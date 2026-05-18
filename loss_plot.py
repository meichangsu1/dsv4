import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import datetime

def extract_loss_from_log(file_path):
    """
    从日志文件中提取 loss 值列表
    """
    losses = []
    pattern = r"'loss': '(\d+\.\d+)'"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                loss_value = float(match.group(1))
                losses.append(loss_value)
    
    return losses

def save_loss_data_to_csv(npu_losses, gpu_losses, output_dir="output"):
    """
    将 loss 数据保存到 CSV 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    min_len = min(len(npu_losses), len(gpu_losses))
    iterations = list(range(1, min_len + 1))
    npu_losses = npu_losses[:min_len]
    gpu_losses = gpu_losses[:min_len]
    loss_diff = [npu - gpu for npu, gpu in zip(npu_losses, gpu_losses)]
    
    csv_path = os.path.join(output_dir, "loss_data.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'NPU_Loss', 'GPU_Loss', 'Loss_Difference_NPU_GPU'])
        
        for i, (npu, gpu, diff) in enumerate(zip(npu_losses, gpu_losses, loss_diff), start=1):
            writer.writerow([i, npu, gpu, diff])
    
    print(f"✅ Loss 数据已保存到: {csv_path}")
    return csv_path

def save_loss_data_to_txt(npu_losses, gpu_losses, output_dir="output"):
    """
    将 loss 数据保存到文本文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    min_len = min(len(npu_losses), len(gpu_losses))
    npu_losses = npu_losses[:min_len]
    gpu_losses = gpu_losses[:min_len]
    loss_diff = [npu - gpu for npu, gpu in zip(npu_losses, gpu_losses)]
    loss_diff = np.abs(loss_diff)  # 取绝对值，关注差值大小而非方向
    
    txt_path = os.path.join(output_dir, "loss_data.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Loss Comparison Data (NPU vs GPU)\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Iteration':<10} {'NPU Loss':<12} {'GPU Loss':<12} {'Difference':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, (npu, gpu, diff) in enumerate(zip(npu_losses, gpu_losses, loss_diff), start=1):
            f.write(f"{i:<10} {npu:<12.6f} {gpu:<12.6f} {diff:<12.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Statistics Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total iterations: {min_len}\n")
        f.write(f"NPU mean loss: {np.mean(npu_losses):.6f}\n")
        f.write(f"GPU mean loss: {np.mean(gpu_losses):.6f}\n")
        f.write(f"Mean difference: {np.mean(loss_diff):.6f}\n")
        f.write(f"Std difference: {np.std(loss_diff):.6f}\n")
        f.write(f"Min difference: {np.min(loss_diff):.6f}\n")
        f.write(f"Max difference: {np.max(loss_diff):.6f}\n")
    
    print(f"✅ Loss 数据已保存到: {txt_path}")
    return txt_path

def save_plot_to_file(npu_losses, gpu_losses, output_dir="output", filename="loss_comparison.png"):
    """
    绘制并保存 loss 对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    min_len = min(len(npu_losses), len(gpu_losses))
    iterations = np.arange(1, min_len + 1)
    
    npu_losses = npu_losses[:min_len]
    gpu_losses = gpu_losses[:min_len]
    loss_diff = np.array(npu_losses) - np.array(gpu_losses)
    #loss_diff = np.abs(loss_diff)  # 取绝对值，关注差值大小而非方向
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 第一张图：Loss 曲线对比
    ax1.plot(iterations, npu_losses, label='NPU Loss', color='red', linewidth=1.5)
    ax1.plot(iterations, gpu_losses, label='GPU Loss', color='blue', linewidth=1.5)
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Comparison: NPU vs GPU')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 第二张图：Loss 差值
    ax2.plot(iterations, loss_diff, label='NPU Loss - GPU Loss', color='green', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Difference')
    ax2.set_title('Loss Difference Curve')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {plot_path}")
    
    # 显示图片（可选）
    plt.show()
    
    return plot_path

def plot_loss_comparison(npu_losses, gpu_losses, output_dir="output"):
    """
    绘制并保存 loss 曲线和差值曲线
    """
    # 保存 CSV 文件
    save_loss_data_to_csv(npu_losses, gpu_losses, output_dir)
    
    # 保存 TXT 文件
    save_loss_data_to_txt(npu_losses, gpu_losses, output_dir)
    
    # 保存图片
    save_plot_to_file(npu_losses, gpu_losses, output_dir)
    
    # 打印统计信息
    min_len = min(len(npu_losses), len(gpu_losses))
    npu_losses = npu_losses[:min_len]
    gpu_losses = gpu_losses[:min_len]
    loss_diff = np.array(npu_losses) - np.array(gpu_losses)
    #loss_diff = np.abs(loss_diff)  # 取绝对值，关注差值大小而非方向
    
    print("\n" + "=" * 60)
    print("📊 统计信息汇总")
    print("=" * 60)
    print(f"✅ 有效迭代数: {min_len}")
    print(f"📈 NPU 平均 Loss: {np.mean(npu_losses):.6f}")
    print(f"📉 GPU 平均 Loss: {np.mean(gpu_losses):.6f}")
    print(f"🔍 平均 Loss 差值: {np.mean(loss_diff):.6f}")
    print(f"📊 Loss 差值标准差: {np.std(loss_diff):.6f}")
    print(f"📈 Loss 差值最小值: {np.min(loss_diff):.6f}")
    print(f"📉 Loss 差值最大值: {np.max(loss_diff):.6f}")
    print("=" * 60)

def main():
    # 请修改为你的实际文件路径
    npu_log_file = "/Users/linjiajia/Desktop/dsv4/dsv4-gpu-tran-loss-layers-2-10000.log"
    gpu_log_file = "/Users/linjiajia/Desktop/dsv4/dsv4-npu-tran-loss-layers-2-10000.log"
    # npu_log_file = sys.argv[1]
    # gpu_log_file = sys.argv[2]
    # 设置输出目录（可以修改）
    output_directory = "loss_analysis_results"
    
    try:
        print("🔍 正在提取 NPU loss...")
        npu_losses = extract_loss_from_log(npu_log_file)
        print(f"   提取到 {len(npu_losses)} 个 loss 值")
        
        print("🔍 正在提取 GPU loss...")
        gpu_losses = extract_loss_from_log(gpu_log_file)
        print(f"   提取到 {len(gpu_losses)} 个 loss 值")
        
        if len(npu_losses) == 0 or len(gpu_losses) == 0:
            print("❌ 错误：未从日志文件中提取到 loss 数据")
            return
        
        loss_len = len(npu_losses)

        # 绘图并保存到文件
        plot_loss_comparison(npu_losses[:loss_len], gpu_losses[:loss_len], output_directory)
        
        print(f"\n✨ 所有结果已保存到 '{output_directory}' 目录中")
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()