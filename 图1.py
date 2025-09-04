import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

# 解决中文字体问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 如果上述字体不可用，使用英文标签
try:
    # 测试中文显示
    fig_test, ax_test = plt.subplots(figsize=(1, 1))
    ax_test.text(0.5, 0.5, '测试', fontsize=12)
    plt.close(fig_test)
    use_chinese = True
except:
    use_chinese = False

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Homography Matrix H Visualization for Perspective Correction' if not use_chinese
             else '单应矩阵H的透视校正可视化效果', fontsize=16, fontweight='bold')

# ===== 子图1: 原始斜拍网格 =====
ax1 = axes[0, 0]

# 创建规则网格
x_regular = np.linspace(0, 8, 9)
y_regular = np.linspace(0, 6, 7)
X_reg, Y_reg = np.meshgrid(x_regular, y_regular)

# 模拟透视变形（斜拍效果）
# 透视变形参数
perspective_factor = 0.15
shear_factor = 0.08

X_distorted = X_reg + Y_reg * perspective_factor + X_reg * Y_reg * 0.01
Y_distorted = Y_reg + X_reg * shear_factor - Y_reg * X_reg * 0.005

# 绘制变形网格
for i in range(len(y_regular)):
    ax1.plot(X_distorted[i, :], Y_distorted[i, :], 'b-', alpha=0.6, linewidth=1)
for j in range(len(x_regular)):
    ax1.plot(X_distorted[:, j], Y_distorted[:, j], 'b-', alpha=0.6, linewidth=1)

# 添加输电线（在变形网格中）
t = np.linspace(1, 7, 60)
cable_y_ideal = 3 + 0.3 * (t - 4)**2  # 抛物线形状
cable_x_distorted = t + cable_y_ideal * perspective_factor + t * cable_y_ideal * 0.01
cable_y_distorted = cable_y_ideal + t * shear_factor - cable_y_ideal * t * 0.005

ax1.plot(cable_x_distorted, cable_y_distorted, 'r-', linewidth=4,
         label='Power Line (Distorted)' if not use_chinese else '输电线（变形）')
ax1.scatter([cable_x_distorted[0], cable_x_distorted[-1]],
           [cable_y_distorted[0], cable_y_distorted[-1]],
           c='red', s=120, marker='s', zorder=5,
           label='Towers' if not use_chinese else '杆塔')

# 添加相机视角示意
camera_cone_x = [0, 2, 8, 6, 0]
camera_cone_y = [0, 1, 7, 6, 0]
ax1.plot(camera_cone_x, camera_cone_y, 'g--', alpha=0.5, linewidth=2)
ax1.fill(camera_cone_x, camera_cone_y, alpha=0.1, color='green')

ax1.set_xlabel('u (pixels)' if not use_chinese else 'u (像素)', fontsize=12)
ax1.set_ylabel('v (pixels)' if not use_chinese else 'v (像素)', fontsize=12)
ax1.set_title('Original Oblique Image\nPerspective Distortion' if not use_chinese
              else '原始斜拍图像\n透视变形效果', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ===== 子图2: 校正后的正射投影网格 =====
ax2 = axes[0, 1]

# 绘制规则网格（校正后）
for i in range(len(y_regular)):
    ax2.plot(X_reg[i, :], Y_reg[i, :], 'g-', alpha=0.7, linewidth=1.5)
for j in range(len(x_regular)):
    ax2.plot(X_reg[:, j], Y_reg[:, j], 'g-', alpha=0.7, linewidth=1.5)

# 校正后的输电线
ax2.plot(t, cable_y_ideal, 'r-', linewidth=4,
         label='Power Line (Corrected)' if not use_chinese else '输电线（校正后）')
ax2.scatter([t[0], t[-1]], [cable_y_ideal[0], cable_y_ideal[-1]],
           c='red', s=120, marker='s', zorder=5,
           label='Towers' if not use_chinese else '杆塔')

# 计算并显示弧垂
endpoints_y = [cable_y_ideal[0], cable_y_ideal[-1]]
straight_line = endpoints_y[0] + (endpoints_y[1] - endpoints_y[0]) * (t - t[0]) / (t[-1] - t[0])
sag_values = cable_y_ideal - straight_line
max_sag_idx = np.argmax(sag_values)
max_sag_value = sag_values[max_sag_idx]

# 绘制最大弧垂
ax2.plot([t[max_sag_idx], t[max_sag_idx]],
         [straight_line[max_sag_idx], cable_y_ideal[max_sag_idx]],
         'purple', linewidth=5, alpha=0.8)
ax2.plot(t, straight_line, 'k--', alpha=0.7, linewidth=2)

ax2.set_xlabel('x\' (meters)' if not use_chinese else 'x\' (米)', fontsize=12)
ax2.set_ylabel('y\' (meters)' if not use_chinese else 'y\' (米)', fontsize=12)
ax2.set_title('After Homography H^(-1)\nOrthogonal Projection' if not use_chinese
              else '经过单应变换H^(-1)\n正射投影结果', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
plt.show()