import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator

# ==========================================
# 【配置区域】
# ==========================================
X_LIMIT = (0,700) 
FILE_BF16 = 'bf16_deepseek_1118.log'      # 基准日志 (绿色背景)
FILE_CURRENT = 'mxfp8_deepseek.log'  # 当前日志 (红色主角)
OUTPUT_IMG = f'draw/compare_loss_{X_LIMIT}.png' # 输出图片名

# 【关键】只看前 2000 步 (锁定范围)
# ==========================================

def read_log_data(file_path):
    """
    读取日志函数
    正则逻辑：匹配 "iteration 数字 / ... lm loss: 数值"
    """
    iters = []
    losses = []
    
    # 这个正则强制要求 iteration 后面必须跟一个 "/"，防止匹配到 nan iterations
    pattern = re.compile(r"iteration\s*(\d+)\s*/\s*\d+.*?lm loss:\s*([\d\.E\+\-]+)", re.IGNORECASE)

    print(f"正在读取: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    try:
                        iter_val = int(match.group(1))
                        loss_val = float(match.group(2))
                        iters.append(iter_val)
                        losses.append(loss_val)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"【错误】找不到文件: {file_path}")
        return [], []

    return iters, losses

def main():
    # 1. 读取数据
    iters_bf16, losses_bf16 = read_log_data(FILE_BF16)
    iters_curr, losses_curr = read_log_data(FILE_CURRENT)

    if not iters_bf16 or not iters_curr:
        print("停止：数据读取不完整，请检查日志。")
        return

    # =======================================================
    # 2. 【计算功能】计算 0 ~ X_LIMIT 范围内的 Loss 比值 (Current / BF16)
    # =======================================================
    # 将 BF16 转为字典 {step: loss}，方便快速查找
    dict_bf16 = dict(zip(iters_bf16, losses_bf16))
    
    ratios = []
    print("-" * 40)
    print(f"正在计算前 {X_LIMIT} 步的 Loss 平均倍率...")

    for i, step in enumerate(iters_curr):
        # 筛选条件：在 2000 步以内，且 BF16 日志里也有这一步的数据
        if step >= X_LIMIT[0] and step <= X_LIMIT[1] and step in dict_bf16:
            loss_now = losses_curr[i]
            loss_base = dict_bf16[step]
            
            if loss_base != 0:
                # 计算比值
                ratios.append(loss_now / loss_base)
    
    # 生成标题用的统计文字
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        ratio_str = f"Avg Ratio (0-{X_LIMIT}): {avg_ratio:.4f}x"
        print(f"【结果】平均比值: {avg_ratio:.4f}")
        print("       (>1.0 表示当前Loss偏高，<1.0 表示当前Loss偏低)")
    else:
        avg_ratio = 0
        ratio_str = "No Overlap Data"
        print("【警告】没有找到重叠步数，无法计算比值。")
    print("-" * 40)

    # 3. 开始绘图
    plt.figure(figsize=(12, 8), dpi=300) # 高清画布

    # --- 画基准线 (Green) ---
    # 绿色，半透明，放在底层作为参考
    plt.plot(iters_bf16, losses_bf16, 
             label='Baseline (BF16)', 
             color='green',      # 【颜色】绿色
             linestyle='-',      # 实线
             linewidth=1.0,      # 【线宽】细线条 (显示细节)
             alpha=0.5,          # 【透明度】半透明 (0.5)
             zorder=1)           # 图层层级 1 (底层)

    # --- 画当前线 (Red) ---
    # 红色，不透明，放在顶层
    plt.plot(iters_curr, losses_curr, 
             label='Current Run', 
             color='red',        # 【颜色】红色
             linestyle='-',      # 实线
             linewidth=1.0,      # 【线宽】细线条
             alpha=1.0,          # 【透明度】不透明 (1.0)
             zorder=10)          # 图层层级 10 (顶层)

    # 4. 设置坐标轴范围与刻度
    ax = plt.gca()
    
    # 【关键】强制锁定横坐标显示范围：0 到 2000
    plt.xlim(0, X_LIMIT[1])
    
    # 智能刻度：让横坐标只显示 8 个左右的数字，清爽不拥挤
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # 5. 装饰图片
    # 把计算出来的倍率直接写在标题里
    plt.title(f'Training Loss Comparison\n{ratio_str}', fontsize=18, fontweight='bold')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('LM Loss', fontsize=14)
    
    # 图例放在右上角
    plt.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
    
    # 网格线
    plt.grid(True, which='major', linestyle='-', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # 6. 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"图片已生成: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()