import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from datetime import datetime
import warnings
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore')

# 设置更美观的样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 10
})

# 使用美观的调色板
colors = plt.cm.viridis(np.linspace(0, 1, 6))
accent_color = plt.cm.Set2(np.linspace(0, 1, 8))


def catenary_model_advanced(x, a, x0, c, b1, b2, b3, L):
    """
    修正悬链线模型 + 修正项
    a: 悬链线参数
    x0: 水平偏移
    c: 垂直偏移
    b1, b2, b3: 修正项系数
    L: 总跨度
    """
    # 防止数值溢出
    arg = np.clip((x - x0) / a, -50, 50)

    # 主要悬链线项
    y_catenary = a * np.cosh(arg) + c

    # 修正项
    delta = b1 * x + b2 * x ** 2 + b3 * np.sin(np.pi * x / L)

    return y_catenary + delta


def fit_advanced_catenary(x_data, y_data, left_point, right_point, exclude_margin=10):
    """
    使用修正悬链线模型拟合，排除挂点附近的样本

    exclude_margin: 挂点附近要排除的边缘宽度（像素）
    """
    L = right_point[0] - left_point[0]  # 总跨度

    # 筛选出不在挂点附近的样本点
    valid_indices = (x_data > left_point[0] + exclude_margin) & (x_data < right_point[0] - exclude_margin)
    x_filtered = x_data[valid_indices]
    y_filtered = y_data[valid_indices]

    if len(x_filtered) < 10:
        print(f"警告: 过滤后的样本点太少 ({len(x_filtered)})，可能影响拟合质量")
        # 如果过滤后样本太少，减少排除边缘
        if exclude_margin > 2:
            return fit_advanced_catenary(x_data, y_data, left_point, right_point, exclude_margin=2)

    print(f"使用 {len(x_filtered)} 个样本点进行拟合 (排除挂点附近 {exclude_margin} 像素的点)")

    # 初始参数估计
    x0_init = L / 2  # 对称中心位置
    a_init = L / 4  # 经验估计值
    c_init = min(left_point[1], right_point[1]) - a_init  # 垂直偏移
    b1_init = (right_point[1] - left_point[1]) / L  # 线性趋势系数

    initial_params = [a_init, x0_init, c_init, b1_init, 0, 0]

    # 包装函数用于curve_fit
    def fitting_func(x, a, x0, c, b1, b2, b3):
        return catenary_model_advanced(x, a, x0, c, b1, b2, b3, L)

    try:
        # 使用curve_fit进行拟合
        popt, _ = curve_fit(fitting_func, x_filtered, y_filtered,
                            p0=initial_params, maxfev=10000)

        # 拟合结果
        a_fit, x0_fit, c_fit, b1_fit, b2_fit, b3_fit = popt

        # 定义预测函数
        def final_prediction_func(x):
            # 确保输入是标量或一维数组
            if isinstance(x, (int, float, np.int64, np.float64)):
                return fitting_func(np.array([x]), *popt)[0]
            else:
                x_arr = np.asarray(x)
                return fitting_func(x_arr, *popt)

        # 计算拟合优度（使用所有原始数据点计算，包括挂点）
        y_pred = final_prediction_func(x_data)
        r_squared = 1 - np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

        # 使用过滤后的点计算RMSE（更准确反映拟合质量）
        y_pred_filtered = final_prediction_func(x_filtered)
        rmse = np.sqrt(np.mean((y_filtered - y_pred_filtered) ** 2))

        print(f"\n修正悬链线拟合参数：")
        print(f"a = {a_fit:.4f}")
        print(f"x0 = {x0_fit:.4f}")
        print(f"c = {c_fit:.4f}")
        print(f"b1 = {b1_fit:.6f}")
        print(f"b2 = {b2_fit:.8f}")
        print(f"b3 = {b3_fit:.4f}")
        print(f"拟合优度 R² = {r_squared:.4f}")
        print(f"RMSE = {rmse:.4f}")

        return final_prediction_func, popt, r_squared, rmse, valid_indices

    except Exception as e:
        print(f"拟合失败: {e}")
        return None, None, 0, 0, None


def calculate_sag_distance(x, y, x1, y1, x2, y2):
    """计算点(x,y)到两端连线的垂直距离"""
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    distance = np.abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
    return distance


def find_max_sag(fitted_function, left_point, right_point, num_points=10000):
    """从拟合曲线中找到最大弧垂"""
    x_dense = np.linspace(left_point[0], right_point[0], num_points)
    y_dense = fitted_function(x_dense)

    # 计算两端连线方程: y = mx + b
    m = (right_point[1] - left_point[1]) / (right_point[0] - left_point[0])
    b = left_point[1] - m * left_point[0]

    # 计算连线上对应的y值
    y_line = m * x_dense + b

    # 计算每个点的弧垂（曲线点到连线的垂直距离）
    sag_distances = np.abs(y_dense - y_line)

    # 找到最大弧垂
    max_sag_idx = np.argmax(sag_distances)
    max_sag = sag_distances[max_sag_idx]
    max_sag_x = x_dense[max_sag_idx]
    max_sag_y = y_dense[max_sag_idx]

    return max_sag, max_sag_x, max_sag_y, x_dense, y_dense, y_line, sag_distances


def extract_wire_centerline(binary_img, left_point, right_point):
    """提取电线中心线"""
    y_coords, x_coords = np.where(binary_img == 255)

    if len(x_coords) == 0:
        return []

    # 按x坐标分组
    x_to_y_dict = {}
    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        if x not in x_to_y_dict:
            x_to_y_dict[x] = []
        x_to_y_dict[x].append(y)

    # 提取每个x坐标的中心点
    wire_center_points = []
    for x in sorted(x_to_y_dict.keys()):
        if x < left_point[0] or x > right_point[0]:
            continue

        y_list = np.array(x_to_y_dict[x])

        # 过滤离群点
        if len(y_list) > 3:
            q1, q3 = np.percentile(y_list, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            y_filtered = y_list[(y_list >= lower_bound) & (y_list <= upper_bound)]

            if len(y_filtered) > 0:
                center_y = np.median(y_filtered)
                wire_center_points.append((x, center_y))
        else:
            center_y = np.median(y_list)
            wire_center_points.append((x, center_y))

    return wire_center_points


def create_main_visualization(img, left_point, right_point, wire_points, fitted_function,
                              max_sag_data, r_squared, valid_indices, output_dir):
    """创建主要拟合结果可视化"""
    # 解包最大弧垂数据
    max_sag, max_sag_x, max_sag_y, x_dense, y_dense, y_line, _ = max_sag_data

    # 创建图表
    plt.figure(figsize=(14, 10))

    # 绘制原始图像
    plt.imshow(img, cmap='gray', alpha=0.6)

    # 绘制原始数据点
    if wire_points:
        x_data = np.array([p[0] for p in wire_points])
        y_data = np.array([p[1] for p in wire_points])

        # 绘制未用于拟合的点（挂点附近的点）
        if valid_indices is not None:
            excluded_x = x_data[~valid_indices]
            excluded_y = y_data[~valid_indices]


        # 绘制用于拟合的点
        if valid_indices is not None:
            included_x = x_data[valid_indices]
            included_y = y_data[valid_indices]
            if len(included_x) > 0:
                plt.scatter(included_x, included_y, color=accent_color[0], s=40, alpha=0.8,
                            label='用于拟合的点')

    # 绘制挂点
    plt.plot(left_point[0], left_point[1], 'o', color='red', markersize=12,
             label='挂点', markeredgecolor='white', markeredgewidth=2)
    plt.plot(right_point[0], right_point[1], 'o', color='red', markersize=12,
             markeredgecolor='white', markeredgewidth=2)

    # 绘制两端连线
    plt.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]],
             color=colors[2], linestyle='--', linewidth=2, label='两端连线')

    # 绘制拟合曲线
    plt.plot(x_dense, y_dense, color=colors[0], linewidth=3,
             label=f'修正悬链线拟合')

    # 绘制最大弧垂点和弧垂线
    plt.scatter(max_sag_x, max_sag_y, color='orange', s=150, marker='*',
                label=f'最大弧垂点')

    # 绘制弧垂线
    line_y_at_max = np.interp(max_sag_x, x_dense, y_line)
    plt.plot([max_sag_x, max_sag_x], [line_y_at_max, max_sag_y],
             color='red', linewidth=2.5, linestyle='-')

    # 添加标注
    plt.annotate(f'最大弧垂: {max_sag:.2f}m',
                 xy=(max_sag_x, (max_sag_y + line_y_at_max) / 2),
                 xytext=(max_sag_x + 50, (max_sag_y + line_y_at_max) / 2 + 5),
                 arrowprops=dict(facecolor='black', arrowstyle='fancy', connectionstyle="arc3,rad=.3"),
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    # 填充弧垂区域
    plt.fill_between(x_dense, y_dense, y_line, where=(y_dense > y_line),
                     color='lightblue', alpha=0.3)

    plt.title('输电线路修正悬链线拟合分析 ', fontsize=18, fontweight='bold', pad=15)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    # 保存图像
    output_path = os.path.join(output_dir, "01_main_fitting_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"主拟合可视化已保存为: {output_path}")
    plt.close()


def create_sag_distribution_visualization(left_point, right_point, max_sag_data, output_dir):
    """创建弧垂分布可视化"""
    # 解包最大弧垂数据
    max_sag, max_sag_x, max_sag_y, x_dense, _, _, sag_distances = max_sag_data

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制弧垂分布曲线
    plt.plot(x_dense, sag_distances, color='blue', linewidth=3)
    plt.fill_between(x_dense, sag_distances, color='blue', alpha=0.3)

    # 标注最大弧垂位置
    plt.axvline(x=max_sag_x, color='red', linestyle='--', alpha=0.7)
    plt.scatter(max_sag_x, max_sag, color='red', s=120, marker='*')

    # 添加标注
    plt.annotate(f'最大弧垂: {max_sag:.2f}m',
                 xy=(max_sag_x, max_sag),
                 xytext=(max_sag_x - 150, max_sag + 1) if max_sag_x > left_point[0] + 200 else (
                 max_sag_x + 50, max_sag + 1),
                 arrowprops=dict(facecolor='black', arrowstyle='fancy', connectionstyle="arc3,rad=.3"),
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    # 设置图表属性
    plt.title('输电线路弧垂分布', fontsize=16, fontweight='bold')
    plt.xlabel('水平距离 (m)')
    plt.ylabel('弧垂 (m)')
    plt.xlim(left_point[0], right_point[0])
    plt.ylim(0, max_sag * 1.1)
    plt.grid(True, alpha=0.3)

    # 保存图像
    output_path = os.path.join(output_dir, "02_sag_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"弧垂分布可视化已保存为: {output_path}")
    plt.close()


def create_sag_diagram_visualization(left_point, right_point, fitted_function, max_sag_data, output_dir):
    """创建弧垂沿线分布示意图"""
    # 解包最大弧垂数据
    max_sag, max_sag_x, max_sag_y, x_dense, y_dense, y_line, _ = max_sag_data

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制悬链线曲线
    ax.plot(x_dense, y_dense, color=colors[0], linewidth=3, label='悬链线')

    # 绘制端点连线
    ax.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]],
            color=colors[2], linestyle='--', linewidth=2, label='端点连线')

    # 创建弧垂示意图
    step = max((right_point[0] - left_point[0]) // 10, 1)  # 自适应步长
    cmap = plt.cm.plasma

    # 选择显示弧垂线的位置
    sag_positions = np.arange(left_point[0] + step, right_point[0], step)

    for x_pos in sag_positions:
        if x_pos == left_point[0] or x_pos == right_point[0]:  # 跳过端点
            continue

        # 确保输入是标量
        y_curve = float(fitted_function(float(x_pos)))
        y_line_pos = float(np.interp(x_pos, x_dense, y_line))
        sag = abs(y_curve - y_line_pos)

        # 使用渐变色表示弧垂大小
        color = cmap(sag / max_sag)

        # 绘制弧垂线
        ax.plot([x_pos, x_pos], [y_line_pos, y_curve], color=color, linewidth=2)

        # 为较大的弧垂添加标签
        if sag > max_sag * 0.7:  # 只为较大的弧垂添加标签
            ax.text(x_pos + 5, (y_curve + y_line_pos) / 2, f"{sag:.2f}m", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

    # 突出显示最大弧垂
    line_y_at_max = np.interp(max_sag_x, x_dense, y_line)
    ax.plot([max_sag_x, max_sag_x], [line_y_at_max, max_sag_y],
            color='red', linewidth=3, label=f'最大弧垂: {max_sag:.2f}m')
    ax.scatter(max_sag_x, max_sag_y, color='red', s=120, marker='*')

    # 添加彩色标签，表示弧垂的大小 - 修复colorbar错误
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_sag))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)  # 显式传递ax参数
    cbar.set_label('弧垂大小 (m)')

    # 设置标题和标签
    ax.set_title('弧垂沿线分布示意图', fontsize=16, fontweight='bold')
    ax.set_xlabel('水平距离 (m)')
    ax.set_ylabel('高度 (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # 保存图像
    output_path = os.path.join(output_dir, "03_sag_distribution_diagram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"弧垂沿线分布示意图已保存为: {output_path}")
    plt.close()


def create_error_analysis_visualization(wire_points, fitted_function, valid_indices, rmse, output_dir):
    """创建拟合误差分析可视化"""
    if not wire_points or valid_indices is None:
        return

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 获取用于拟合的数据点
    x_data = np.array([p[0] for p in wire_points])
    y_data = np.array([p[1] for p in wire_points])

    # 只使用有效点计算误差
    included_x = x_data[valid_indices]
    included_y = y_data[valid_indices]

    if len(included_x) > 0:
        # 计算拟合点的误差
        y_pred = fitted_function(included_x)
        residuals = included_y - y_pred

        # 绘制误差柱状图
        bars = plt.bar(included_x, residuals, width=max((included_x.max() - included_x.min()) / 100, 1), alpha=0.8)

        # 为柱状图设置不同的颜色
        for i, bar in enumerate(bars):
            if residuals[i] >= 0:
                bar.set_color(accent_color[0])
            else:
                bar.set_color(accent_color[1])

        # 添加误差均值和标准差线
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        plt.axhline(y=mean_error, color='red', linestyle='-', linewidth=1.5,
                    label=f'均值: {mean_error:.2f}')
        plt.axhline(y=mean_error + std_error, color='orange', linestyle='--', linewidth=1,
                    label=f'标准差: {std_error:.2f}')
        plt.axhline(y=mean_error - std_error, color='orange', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # 添加RMSE指标
        plt.text(0.02, 0.95, f"均方根误差(RMSE): {rmse:.4f}",
                 transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

        # 设置标题和标签
        plt.title('拟合误差分析 ', fontsize=16, fontweight='bold')
        plt.xlabel('水平距离 (m)')
        plt.ylabel('误差 (m)')

        # 调整y轴范围以便更好地显示误差
        max_abs_error = max(abs(max(residuals)), abs(min(residuals)))
        plt.ylim(-max_abs_error * 1.2, max_abs_error * 1.2)
        plt.legend()

        # 保存图像
        output_path = os.path.join(output_dir, "04_error_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"拟合误差分析已保存为: {output_path}")
        plt.close()


def create_model_info_visualization(fitting_params, r_squared, rmse, max_sag, max_sag_x, left_point, right_point,
                                    output_dir):
    """创建模型信息和参数可视化"""
    # 创建图表
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.axis('off')  # 不显示坐标轴

    a, x0, c, b1, b2, b3 = fitting_params
    L = right_point[0] - left_point[0]

    # 添加模型信息文本
    model_text = (
        "修正悬链线模型 (排除挂点附近样本)\n"
        "————————————————————————————————\n"
        "模型方程: y(x) = a·cosh((x-x₀)/a) + c + δ(x)\n"
        "修正项: δ(x) = b₁·x + b₂·x² + b₃·sin(πx/L)\n"
        "————————————————————————————————\n"
        f"参数值:\n"
        f"  a = {a:.4f}\n"
        f"  x₀ = {x0:.4f}\n"
        f"  c = {c:.4f}\n"
        f"  b₁ = {b1:.6f}\n"
        f"  b₂ = {b2:.8f}\n"
        f"  b₃ = {b3:.4f}\n"
        f"  L = {L:.2f}\n"
        "————————————————————————————————\n"
        f"拟合统计:\n"
        f"  拟合优度 R² = {r_squared:.4f}\n"
        f"  均方根误差 RMSE = {rmse:.4f}\n"
        f"  最大弧垂 = {max_sag:.2f} m\n"
        f"  最大弧垂位置 = {max_sag_x:.2f} m\n"
    )

    plt.text(0.5, 0.5, model_text, fontsize=14, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=1", facecolor='#f0f0f0',
                       edgecolor='gray', alpha=0.9))

    # 保存图像
    output_path = os.path.join(output_dir, "05_model_parameters.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"模型参数信息已保存为: {output_path}")
    plt.close()


def save_results(max_sag, max_sag_pos, fitting_params, r_squared, rmse, image_path,
                 exclude_margin, num_filtered, output_dir):
    """保存分析结果"""
    filename = os.path.join(output_dir, "弧垂分析结果.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("输电线路弧垂分析结果报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"分析图像: {image_path}\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 60 + "\n")

            f.write("主要结果:\n")
            f.write(f"  最大弧垂: {max_sag:.2f} 米\n")
            f.write(f"  最大弧垂位置: ({max_sag_pos[0]:.2f}, {max_sag_pos[1]:.2f}) 米\n")
            f.write("-" * 60 + "\n")

            f.write("拟合方法: 修正悬链线 (排除挂点附近样本)\n")
            f.write(f"排除挂点附近 {exclude_margin} 像素的样本\n")
            f.write(f"使用了 {num_filtered} 个样本点进行拟合\n")
            f.write(f"拟合优度 (R²): {r_squared:.6f}\n")
            f.write(f"均方根误差 (RMSE): {rmse:.6f}\n")
            f.write("-" * 60 + "\n")

            a, x0, c, b1, b2, b3 = fitting_params
            f.write("模型方程: y(x) = a·cosh((x-x₀)/a) + c + δ(x)\n")
            f.write("修正项: δ(x) = b₁·x + b₂·x² + b₃·sin(πx/L)\n\n")
            f.write("模型参数:\n")
            f.write(f"  a = {a:.6f}\n")
            f.write(f"  x₀ = {x0:.6f}\n")
            f.write(f"  c = {c:.6f}\n")
            f.write(f"  b₁ = {b1:.6f}\n")
            f.write(f"  b₂ = {b2:.8f}\n")
            f.write(f"  b₃ = {b3:.6f}\n")
            f.write("-" * 60 + "\n")

            f.write("说明: 根据题目条件，每像素对应1米\n")
            f.write("=" * 60 + "\n")

        print(f"分析结果已保存到: {filename}")
    except Exception as e:
        print(f"保存结果文件时出错: {e}")


def find_max_sag_with_advanced_catenary(image_path, exclude_margin=15, output_dir="output"):
    """
    使用修正悬链线拟合的弧垂检测主函数

    exclude_margin: 挂点附近要排除的边缘宽度（像素）
    output_dir: 输出文件夹
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 '{image_path}'")
        return None

    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法读取图像文件 '{image_path}'")
        return None

    original_img = img.copy()
    print(f"成功读取图像: {image_path}")
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]} 像素")

    # 图像预处理
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)

    # 找到电线像素点
    y_coords, x_coords = np.where(binary_clean == 255)

    if len(x_coords) == 0:
        print("错误: 未在图像中找到电线")
        return None

    # 寻找挂点
    left_idx = np.argmin(x_coords)
    right_idx = np.argmax(x_coords)
    left_point = (x_coords[left_idx], y_coords[left_idx])
    right_point = (x_coords[right_idx], y_coords[right_idx])

    print(f"找到挂点: 左挂点{left_point}, 右挂点{right_point}")

    # 提取电线中心线
    print("提取电线中心线...")
    wire_center_points = extract_wire_centerline(binary_clean, left_point, right_point)

    if len(wire_center_points) < 4:
        print("错误: 电线中心点数量不足")
        return None

    print(f"提取到 {len(wire_center_points)} 个中心点")

    # 准备拟合数据
    x_data = np.array([point[0] for point in wire_center_points])
    y_data = np.array([point[1] for point in wire_center_points])

    # 进行修正悬链线拟合，排除挂点附近的样本
    print(f"开始修正悬链线拟合 (排除挂点附近 {exclude_margin} 像素的样本)...")
    fitted_function, fitting_params, r_squared, rmse, valid_indices = fit_advanced_catenary(
        x_data, y_data, left_point, right_point, exclude_margin=exclude_margin)

    if fitted_function is None:
        print("错误: 拟合失败")
        return None

    # 统计用于拟合的样本数
    num_filtered = np.sum(valid_indices) if valid_indices is not None else 0

    # 计算最大弧垂
    print("计算最大弧垂...")
    max_sag_data = find_max_sag(fitted_function, left_point, right_point)
    max_sag, max_sag_x, max_sag_y = max_sag_data[:3]

    print(f"最大弧垂: {max_sag:.2f} 米，位置: ({max_sag_x:.2f}, {max_sag_y:.2f})")

    # 创建各种可视化（分别保存）
    create_main_visualization(original_img, left_point, right_point, wire_center_points,
                              fitted_function, max_sag_data, r_squared, valid_indices, output_dir)

    create_sag_distribution_visualization(left_point, right_point, max_sag_data, output_dir)

    create_sag_diagram_visualization(left_point, right_point, fitted_function, max_sag_data, output_dir)

    create_error_analysis_visualization(wire_center_points, fitted_function, valid_indices, rmse, output_dir)

    create_model_info_visualization(fitting_params, r_squared, rmse, max_sag, max_sag_x, left_point, right_point,
                                    output_dir)

    # 保存结果
    save_results(max_sag, (max_sag_x, max_sag_y), fitting_params,
                 r_squared, rmse, image_path, exclude_margin, num_filtered, output_dir)

    # 计算统计信息
    span_distance = np.sqrt((right_point[0] - left_point[0]) ** 2 + (right_point[1] - left_point[1]) ** 2)
    sag_ratio = max_sag / span_distance if span_distance > 0 else 0

    print("\n" + "=" * 60)
    print("分析结果统计:")
    print("=" * 60)
    print(f"拟合方法: 修正悬链线 (排除挂点附近 {exclude_margin} 像素的样本)")
    print(f"使用了 {num_filtered} 个样本点进行拟合")
    print(f"拟合优度 (R²): {r_squared:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"档距长度: {span_distance:.2f} 米")
    print(f"最大弧垂: {max_sag:.2f} 米")
    print(f"弧垂比: {sag_ratio:.4f}")
    print("=" * 60)

    return max_sag


# 使用函数
if __name__ == "__main__":
    image_path = "img_2.png"  # 替换为您的图像路径
    exclude_margin = 15  # 挂点附近要排除的边缘宽度（像素）
    output_dir = "output_visualizations"  # 输出文件夹

    print("开始基于修正悬链线的输电线路弧垂分析...")
    print("-" * 50)

    max_sag = find_max_sag_with_advanced_catenary(image_path, exclude_margin, output_dir)

    if max_sag is not None:
        print(f"\n最终结果: 根据题目条件，每像素对应1米")
        print(f"基于修正悬链线拟合的最大弧垂为 {max_sag:.2f} 米")
        print(f"\n分析完成！请查看'{output_dir}'文件夹中的可视化图像和结果文件。")
    else:
        print("\n分析失败，请检查图像文件和程序设置。")
