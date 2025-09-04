import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建3D图形
fig = plt.figure(figsize=(14, 10))
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
ax.plot(X_w, Y_w, Z_w, 'b-', linewidth=4, label='架空输电线')
ax.scatter([X_w[0], X_w[-1]], [Y_w[0], Y_w[-1]], [Z_w[0], Z_w[-1]],
           c='red', s=150, marker='s', label='杆塔位置')

# 绘制相机位置
ax.scatter(*camera_pos, c='green', s=300, marker='^', label='相机位置')

# 绘制视线
ax.plot([camera_pos[0], look_at[0]],
        [camera_pos[1], look_at[1]],
        [camera_pos[2], look_at[2]], 'g--', linewidth=3, alpha=0.8)

# 绘制坐标系
origin = np.array([0, 0, 0])
axis_length = 3
# X轴 - 红色
ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0,
          color='red', arrow_length_ratio=0.1, linewidth=3, label='X轴')
# Y轴 - 绿色
ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0,
          color='green', arrow_length_ratio=0.1, linewidth=3, label='Y轴')
# Z轴 - 蓝色
ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length,
          color='blue', arrow_length_ratio=0.1, linewidth=3, label='Z轴')

# 添加网格平面（用于显示地面参考）
x_grid = np.linspace(-2, 12, 8)
y_grid = np.linspace(0, 10, 6)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = np.zeros_like(X_grid)

# 绘制地面网格
for i in range(len(y_grid)):
    ax.plot(X_grid[i, :], Y_grid[i, :], Z_grid[i, :], 'k-', alpha=0.2, linewidth=0.5)
for j in range(len(x_grid)):
    ax.plot(X_grid[:, j], Y_grid[:, j], Z_grid[:, j], 'k-', alpha=0.2, linewidth=0.5)

# 添加相机视锥
# 相机视锥的四个角点
fov = np.pi/3  # 60度视场角
distance = 4
corners = [
    look_at + distance * np.array([-np.tan(fov/2), -np.tan(fov/2), 0]),
    look_at + distance * np.array([np.tan(fov/2), -np.tan(fov/2), 0]),
    look_at + distance * np.array([np.tan(fov/2), np.tan(fov/2), 0]),
    look_at + distance * np.array([-np.tan(fov/2), np.tan(fov/2), 0])
]

# 绘制视锥边缘
for corner in corners:
    ax.plot([camera_pos[0], corner[0]],
            [camera_pos[1], corner[1]],
            [camera_pos[2], corner[2]], 'g-', alpha=0.3, linewidth=1)

# 设置图形属性
ax.set_xlabel('X 坐标 (米)', fontsize=14)
ax.set_ylabel('Y 坐标 (米)', fontsize=14)
ax.set_zlabel('Z 坐标 (米)', fontsize=14)
ax.set_title('相机成像模型与世界坐标系\n问题三：相机参数已知的三维重建建模',
             fontsize=16, fontweight='bold', pad=20)

# 设置图例
ax.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0, 1))
ax.grid(True, alpha=0.3)

# 设置视角和范围
ax.view_init(elev=25, azim=45)
ax.set_xlim(-2, 12)
ax.set_ylim(0, 12)
ax.set_zlim(-1, 9)

# 添加文字注释
ax.text(camera_pos[0]+0.5, camera_pos[1]+0.5, camera_pos[2]+1,
        '相机\n(已知内外参)', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax.text(5, 8, 2, '弧垂测量目标', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()
