import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 定义世界坐标系中的输电线点
t = np.linspace(0, 10, 50)
X_w = t
Y_w = 5 + 0.5 * (t - 5)**2  # 抛物线形状
Z_w = np.zeros_like(t)

# 相机位置和姿态
camera_pos = np.array([5, 10, 8])
look_at = np.array([5, 5, 0])
up = np.array([0, 0, 1])

# 绘制输电线
ax.plot(X_w, Y_w, Z_w, 'b-', linewidth=3, label='架空输电线')
ax.scatter([X_w[0], X_w[-1]], [Y_w[0], Y_w[-1]], [Z_w[0], Z_w[-1]],
           c='red', s=100, label='杆塔位置')

# 绘制相机位置
ax.scatter(*camera_pos, c='green', s=200, marker='^', label='相机位置')

# 绘制视线
ax.plot([camera_pos[0], look_at[0]],
        [camera_pos[1], look_at[1]],
        [camera_pos[2], look_at[2]], 'g--', linewidth=2, alpha=0.7)

# 绘制坐标系
origin = np.array([0, 0, 0])
axis_length = 2
# X轴 - 红色
ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0,
          color='red', arrow_length_ratio=0.1, linewidth=2, label='X轴')
# Y轴 - 绿色
ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0,
          color='green', arrow_length_ratio=0.1, linewidth=2, label='Y轴')
# Z轴 - 蓝色
ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length,
          color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z轴')

# 设置图形属性
ax.set_xlabel('X (米)', fontsize=12)
ax.set_ylabel('Y (米)', fontsize=12)
ax.set_zlabel('Z (米)', fontsize=12)
ax.set_title('相机成像模型与世界坐标系\n问题三：相机参数已知的三维重建', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 设置视角
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()
