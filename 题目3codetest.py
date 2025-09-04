import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

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

# 创建输出目录
output_dir = "camera_inverse_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")


def simulate_camera_perspective(img, theta_deg=30, phi_deg=20, scale=0.8):
    """
    模拟相机透视变换

    参数:
        img: 原始图像
        theta_deg: 绕x轴旋转角度(度)
        phi_deg: 绕y轴旋转角度(度)
        scale: 投影缩放因子

    返回:
        transformed_img: 模拟相机透视变换后的图像
        M: 变换矩阵
    """
    h, w = img.shape[:2]

    # 将角度转换为弧度
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    # 创建旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    # 组合旋转
    R = np.dot(Ry, Rx)

    # 创建简单的投影矩阵
    f = w  # 近似焦距
    K = np.array([
        [f * scale, 0, w / 2],
        [0, f * scale, h / 2],
        [0, 0, 1]
    ])

    # 定义源点（原始图像的角点）
    src_pts = np.array([
        [0, 0],  # 左上角
        [w - 1, 0],  # 右上角
        [0, h - 1],  # 左下角
        [w - 1, h - 1]  # 右下角
    ], dtype=np.float32)

    # 将点投影到3D并应用旋转
    src_3d = np.zeros((4, 3))
    src_3d[:, :2] = src_pts
    src_3d[:, 2] = 0  # z = 0（平面）

    # 旋转前将点居中到原点
    src_3d[:, 0] -= w / 2
    src_3d[:, 1] -= h / 2

    # 应用旋转
    dst_3d = np.zeros_like(src_3d)
    for i in range(4):
        dst_3d[i] = np.dot(R, src_3d[i])

    # 将点移回
    dst_3d[:, 0] += w / 2
    dst_3d[:, 1] += h / 2

    # 投影回2D
    dst_pts = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        x, y, z = dst_3d[i]
        dst_pts[i, 0] = x * f / (z + f) + w / 2 * (1 - f / (z + f))
        dst_pts[i, 1] = y * f / (z + f) + h / 2 * (1 - f / (z + f))

    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 应用透视变换
    transformed_img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return transformed_img, M


def inverse_perspective_transform(img, M):
    """
    应用逆透视变换恢复原始视角

    参数:
        img: 变换后的图像
        M: 原始变换矩阵

    返回:
        recovered_img: 应用逆变换后的图像
    """
    h, w = img.shape[:2]

    # 计算逆变换矩阵
    M_inv = cv2.invert(M)[1]

    # 应用逆变换
    recovered_img = cv2.warpPerspective(img, M_inv, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return recovered_img, M_inv


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

    plt.title('输电线路修正悬链线拟合分析', fontsize=18, fontweight='bold', pad=15)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    # 保存图像
    output_path = os.path.join(output_dir, "main_fitting_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"主拟合可视化已保存为: {output_path}")
    plt.close()


def analyze_image(img_path, exclude_margin=15, output_dir="output"):
    """分析图像计算最大弧垂"""
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法读取图像文件 '{img_path}'")
        return None

    original_img = img.copy()
    print(f"成功读取图像: {img_path}")
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

    # 计算最大弧垂
    print("计算最大弧垂...")
    max_sag_data = find_max_sag(fitted_function, left_point, right_point)
    max_sag, max_sag_x, max_sag_y = max_sag_data[:3]

    print(f"最大弧垂: {max_sag:.2f} 米，位置: ({max_sag_x:.2f}, {max_sag_y:.2f})")

    # 创建主要可视化
    create_main_visualization(original_img, left_point, right_point, wire_center_points,
                              fitted_function, max_sag_data, r_squared, valid_indices, output_dir)

    # 计算统计信息
    span_distance = np.sqrt((right_point[0] - left_point[0]) ** 2 + (right_point[1] - left_point[1]) ** 2)
    sag_ratio = max_sag / span_distance if span_distance > 0 else 0

    print("\n" + "=" * 60)
    print("分析结果统计:")
    print("=" * 60)
    print(f"拟合方法: 修正悬链线 (排除挂点附近 {exclude_margin} 像素的样本)")
    print(f"档距长度: {span_distance:.2f} 米")
    print(f"最大弧垂: {max_sag:.2f} 米")
    print(f"弧垂比: {sag_ratio:.4f}")
    print("=" * 60)

    return max_sag


def create_visualization_comparison(original_img, transformed_img, recovered_img, output_dir,
                                    title1="原始图像", title2="相机透视图", title3="逆变换恢复图像"):
    """创建三个图像的比较可视化"""
    plt.figure(figsize=(16, 6))

    # 绘制原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(title1, fontsize=14)
    plt.axis('off')

    # 绘制透视变换图像
    plt.subplot(1, 3, 2)
    plt.imshow(transformed_img, cmap='gray')
    plt.title(title2, fontsize=14)
    plt.axis('off')

    # 绘制恢复图像
    plt.subplot(1, 3, 3)
    plt.imshow(recovered_img, cmap='gray')
    plt.title(title3, fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "image_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图像对比可视化已保存为: {output_path}")
    plt.close()


def run_inverse_workflow(camera_image_path, output_dir, camera_params=None):
    """
    运行从相机图像到最大弧垂计算的完整逆向工作流程

    参数:
        camera_image_path: 相机图像路径
        output_dir: 输出目录
        camera_params: 相机参数(theta, phi, scale)，如果为None则尝试估计
    """
    print("\n" + "=" * 60)
    print("开始从相机图像恢复原始视角并计算最大弧垂")
    print("=" * 60)

    # 读取相机图像
    camera_img = cv2.imread(camera_image_path, cv2.IMREAD_GRAYSCALE)
    if camera_img is None:
        print(f"错误: 无法读取相机图像 '{camera_image_path}'")
        return None

    print(f"成功读取相机图像: {camera_image_path}")
    print(f"图像尺寸: {camera_img.shape[1]} x {camera_img.shape[0]} 像素")

    # 步骤1: 应用逆透视变换
    print("\n步骤1: 应用逆透视变换恢复原始视角...")

    # 对于真实场景，我们可能需要从EXIF或用户输入获取相机参数
    # 在这个示例中，我们使用给定的相机参数来模拟相机透视
    # 这些参数应该与生成相机视角图像时使用的参数相同

    # 如果提供了相机参数，使用它们；否则使用默认值
    if camera_params:
        theta, phi, scale = camera_params
    else:
        # 从之前代码中的模拟参数
        theta = 20  # 绕x轴旋转角度(度)
        phi = 15  # 绕y轴旋转角度(度)
        scale = 0.9  # 缩放因子

    # 使用相同的相机参数，但是构建逆向变换矩阵
    h, w = camera_img.shape[:2]

    # 将角度转换为弧度
    theta = np.radians(theta)
    phi = np.radians(phi)

    # 创建旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    # 组合旋转
    R = np.dot(Ry, Rx)

    # 创建简单的投影矩阵
    f = w  # 近似焦距
    K = np.array([
        [f * scale, 0, w / 2],
        [0, f * scale, h / 2],
        [0, 0, 1]
    ])

    # 定义源点（原始图像的角点）
    src_pts = np.array([
        [0, 0],  # 左上角
        [w - 1, 0],  # 右上角
        [0, h - 1],  # 左下角
        [w - 1, h - 1]  # 右下角
    ], dtype=np.float32)

    # 将点投影到3D并应用旋转
    src_3d = np.zeros((4, 3))
    src_3d[:, :2] = src_pts
    src_3d[:, 2] = 0  # z = 0（平面）

    # 旋转前将点居中到原点
    src_3d[:, 0] -= w / 2
    src_3d[:, 1] -= h / 2

    # 应用旋转
    dst_3d = np.zeros_like(src_3d)
    for i in range(4):
        dst_3d[i] = np.dot(R, src_3d[i])

    # 将点移回
    dst_3d[:, 0] += w / 2
    dst_3d[:, 1] += h / 2

    # 投影回2D
    dst_pts = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        x, y, z = dst_3d[i]
        dst_pts[i, 0] = x * f / (z + f) + w / 2 * (1 - f / (z + f))
        dst_pts[i, 1] = y * f / (z + f) + h / 2 * (1 - f / (z + f))

    # 获取透视变换矩阵 - 这将从正视图转换到相机视角
    M_forward = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 计算逆变换矩阵 - 这将从相机视角转换回正视图
    M_inverse = cv2.invert(M_forward)[1]

    # 应用逆变换恢复原始视角
    recovered_img = cv2.warpPerspective(camera_img, M_inverse, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=255)

    # 保存恢复后的图像
    recovered_img_path = os.path.join(output_dir, "recovered_image.png")
    cv2.imwrite(recovered_img_path, recovered_img)
    print(f"恢复的正视图图像已保存为: {recovered_img_path}")

    # 步骤2: 验证恢复效果（通过图像比较）
    # 读取原始图像进行比较（如果有）
    original_img_path = "img_2.png"  # 原始图像路径
    original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)

    if original_img is not None:
        # 创建比较可视化
        create_visualization_comparison(
            original_img, camera_img, recovered_img,
            output_dir,
            "原始正视图",
            f"相机视角 (θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°)",
            "逆变换恢复图像"
        )

    # 步骤3: 分析恢复后的图像
    print("\n步骤3: 分析恢复后的图像计算最大弧垂...")
    max_sag = analyze_image(recovered_img_path, exclude_margin=15, output_dir=output_dir)

    if max_sag is not None:
        print("\n" + "=" * 60)
        print(f"最终结果: 从相机图像恢复的最大弧垂为 {max_sag:.2f} 米")
        print("=" * 60)

        # 创建结果记录文件
        result_file = os.path.join(output_dir, "分析结果.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("从相机图像恢复的输电线路弧垂分析结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"原始相机图像: {camera_image_path}\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"相机参数: θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°, scale={scale}\n")
            f.write(f"透视校正后的最大弧垂: {max_sag:.2f} 米\n")
            f.write("=" * 60 + "\n")

        print(f"分析结果已保存到: {result_file}")

        return max_sag
    else:
        print("\n分析失败，无法计算最大弧垂")
        return None


if __name__ == "__main__":
    # 相机拍摄的图像路径
    camera_image_path = "img_2.png"  # 请替换为您的相机图像路径

    # 已知的相机参数（应与生成相机视图时使用的参数相同）
    camera_params = (20, 15, 0.9)  # theta, phi, scale

    # 运行完整的逆向工作流程
    max_sag = run_inverse_workflow(camera_image_path, output_dir, camera_params)

    if max_sag is not None:
        print(f"\n工作流程完成！从相机图像恢复的最大弧垂为 {max_sag:.2f} 米")
        print(f"所有结果已保存到 '{output_dir}' 文件夹")
    else:
        print("\n工作流程未成功完成，请检查图像和程序设置")
