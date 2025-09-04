import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

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
    'axes.grid': False,  # 关闭网格，图像处理图不需要网格
    'lines.linewidth': 2,
    'lines.markersize': 10
})

# 创建自定义配色方案
colors = plt.cm.viridis(np.linspace(0, 1, 6))
accent_color = plt.cm.Set2(np.linspace(0, 1, 8))

# 创建输出目录
output_dir = "图像处理过程"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def visualize_image_processing(image_path):
    """可视化图像处理过程"""

    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 '{image_path}'")
        return

    # 读取图像
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print(f"错误: 无法读取图像文件 '{image_path}'")
        return

    print(f"成功读取图像: {image_path}")
    print(f"图像尺寸: {original_img.shape[1]} x {original_img.shape[0]} 像素")

    # 保存原始图像
    save_image_comparison(
        [original_img],
        ["原始图像"],
        "1_原始图像.png",
        "输电线路原始图像",
        show_colorbar=False
    )

    # 1. 高斯模糊 - 减少噪声
    img_blur = cv2.GaussianBlur(original_img, (5, 5), 0)

    # 可视化高斯模糊结果
    save_image_comparison(
        [original_img, img_blur],
        ["原始图像", "高斯模糊"],
        "2_高斯模糊.png",
        "高斯模糊降噪处理",
        cmaps=['gray', 'gray']
    )

    # 2. 二值化处理
    _, binary = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

    # 可视化二值化结果
    save_image_comparison(
        [img_blur, binary],
        ["高斯模糊", "二值化处理 (反转)"],
        "3_二值化处理.png",
        "二值化处理 - 将电线提取为前景",
        cmaps=['gray', 'binary']
    )

    # 3. 形态学处理 - 开运算和闭运算
    kernel = np.ones((3, 3), np.uint8)

    # 开运算 - 先腐蚀后膨胀，去除小噪点
    morph_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 闭运算 - 先膨胀后腐蚀，填充小孔洞
    binary_clean = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

    # 可视化形态学处理结果
    save_image_comparison(
        [binary, morph_open, binary_clean],
        ["二值化图像", "开运算 (去噪点)", "闭运算 (填充孔洞)"],
        "4_形态学处理.png",
        "形态学处理 - 净化二值图像",
        cmaps=['binary', 'binary', 'binary']
    )

    # 4. 提取电线中心线
    # 首先找到电线的左右端点
    y_coords, x_coords = np.where(binary_clean == 255)

    if len(x_coords) == 0:
        print("错误: 未在图像中找到电线")
        return

    # 寻找挂点
    left_idx = np.argmin(x_coords)
    right_idx = np.argmax(x_coords)
    left_point = (x_coords[left_idx], y_coords[left_idx])
    right_point = (x_coords[right_idx], y_coords[right_idx])

    print(f"找到挂点: 左挂点{left_point}, 右挂点{right_point}")

    # 提取中心线
    wire_center_points = extract_wire_centerline(binary_clean, left_point, right_point)

    # 创建一个新的图像来显示中心线
    center_line_img = np.zeros_like(binary_clean)
    for point in wire_center_points:
        cv2.circle(center_line_img, (int(point[0]), int(point[1])), 1, 255, -1)

    # 可视化中心线提取结果
    overlay_img = create_color_overlay(original_img, binary_clean, center_line_img)

    # 可视化中心线提取结果
    save_image_comparison(
        [original_img, binary_clean, center_line_img, overlay_img],
        ["原始图像", "二值化处理后", "提取的中心线", "综合叠加效果"],
        "5_中心线提取.png",
        "电线中心线提取过程",
        nrows=2, ncols=2,
        cmaps=['gray', 'binary', 'binary', None]
    )

    # 5. 创建电线中心点的可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(original_img, cmap='gray')

    # 绘制挂点
    ax.plot(left_point[0], left_point[1], 'o', color='red', markersize=12,
            label='挂点', markeredgecolor='white', markeredgewidth=2)
    ax.plot(right_point[0], right_point[1], 'o', color='red', markersize=12,
            markeredgecolor='white', markeredgewidth=2)

    # 绘制中心线点
    x_points = [point[0] for point in wire_center_points]
    y_points = [point[1] for point in wire_center_points]

    # 使用渐变色标记中心点，表示沿线位置
    colors = plt.cm.viridis(np.linspace(0, 1, len(wire_center_points)))
    scatter = ax.scatter(x_points, y_points, c=colors, s=30, alpha=0.8, zorder=3)

    # 添加连线
    ax.plot(x_points, y_points, '-', color='lightblue', linewidth=1.5, alpha=0.7, zorder=2)

    # 绘制两端连线
    ax.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]],
            color='orange', linestyle='--', linewidth=2, label='两端连线', zorder=1)

    # 显示挂点附近区域（不用于拟合的区域）
    exclude_margin = 15
    ax.axvspan(left_point[0], left_point[0] + exclude_margin, alpha=0.2, color='red',
               label=f'挂点附近 ({exclude_margin}像素)')
    ax.axvspan(right_point[0] - exclude_margin, right_point[0], alpha=0.2, color='red')

    ax.set_title('电线中心点提取与挂点识别', fontsize=18)
    ax.legend(loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "6_中心点与挂点.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"电线中心点与挂点可视化已保存为: {output_path}")
    plt.close()

    # 6. 展示不同阶段的数据点密度变化
    # 原始二值图中的点
    binary_points = np.sum(binary == 255)
    # 清洁二值图中的点
    clean_points = np.sum(binary_clean == 255)
    # 中心线点数
    center_points = len(wire_center_points)
    # 挂点区域点数
    valid_indices = np.array([(p[0] > left_point[0] + exclude_margin) &
                              (p[0] < right_point[0] - exclude_margin)
                              for p in wire_center_points])
    fitting_points = np.sum(valid_indices)

    # 创建点数变化柱状图
    plt.figure(figsize=(12, 8))
    labels = ['二值化图像中的点', '形态学处理后的点', '提取的中心线点', '用于拟合的点']
    counts = [binary_points, clean_points, center_points, fitting_points]

    # 对数刻度，因为数值范围较大
    plt.bar(labels, counts, color=[accent_color[i] for i in range(4)])
    plt.yscale('log')

    # 在柱子上标注具体数值
    for i, count in enumerate(counts):
        plt.text(i, count * 1.1, f'{count:,}', ha='center', fontsize=12)

    plt.grid(axis='y', alpha=0.3)
    plt.title('图像处理过程中的数据点数量变化', fontsize=18)
    plt.ylabel('点数 (对数刻度)')
    plt.tight_layout()

    output_path = os.path.join(output_dir, "7_数据点数量变化.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"数据点数量变化图已保存为: {output_path}")
    plt.close()

    print(f"所有图像处理过程可视化结果已保存到 '{output_dir}' 文件夹")


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


def save_image_comparison(images, titles, filename, fig_title=None, nrows=1, ncols=None,
                          cmaps=None, show_colorbar=True):
    """保存图像对比可视化"""
    if ncols is None:
        ncols = len(images)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8 * nrows))

    # 确保axes是数组，即使只有一张图
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()

    # 默认使用灰度图
    if cmaps is None:
        cmaps = ['gray'] * len(images)

    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        ax_idx = i
        if nrows > 1 and ncols > 1:
            ax_idx = (i // ncols, i % ncols)

        im = axes[ax_idx].imshow(img, cmap=cmap)
        axes[ax_idx].set_title(title)
        axes[ax_idx].axis('off')

        if show_colorbar and cmap is not None:
            fig.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)

    if fig_title:
        fig.suptitle(fig_title, fontsize=20, y=0.98)

    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图像对比图已保存为: {output_path}")
    plt.close()


def create_color_overlay(gray_img, binary_img, center_line_img):
    """创建彩色叠加效果图"""
    # 创建一个RGB图像，基础是灰度图
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    # 创建一个透明度图层
    alpha = np.zeros_like(gray_img)

    # 二值化区域设置为半透明
    alpha[binary_img > 0] = 100

    # 中心线区域设置为完全不透明
    alpha[center_line_img > 0] = 255

    # 创建颜色图层
    color_layer = np.zeros_like(rgb_img)

    # 二值化区域设置为蓝色
    color_layer[binary_img > 0] = [30, 144, 255]  # 道奇蓝

    # 中心线区域设置为红色
    color_layer[center_line_img > 0] = [255, 69, 0]  # 橙红色

    # 叠加图层
    alpha_norm = alpha[:, :, np.newaxis] / 255.0
    result = (rgb_img * (1 - alpha_norm) + color_layer * alpha_norm).astype(np.uint8)

    return result


# 使用函数
if __name__ == "__main__":
    image_path = "img.png"  # 替换为您的图像路径

    print("开始生成图像处理过程可视化...")
    print("-" * 50)

    visualize_image_processing(image_path)

    print("\n可视化完成！请查看 '图像处理过程' 文件夹中的图片。")
