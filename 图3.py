import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def apply_distortion(x, y, k1=-0.3, k2=0.1, p1=0.01, p2=0.01):
    """应用径向和切向畸变"""
    r2 = x ** 2 + y ** 2
    radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2

    x_distorted = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_distorted = y * radial_distortion + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    return x_distorted, y_distorted


def camera_projection(X, Y, Z, camera_matrix, R, T):
    """相机投影变换"""
    # 世界坐标到相机坐标
    world_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
    camera_points = R @ world_points + T.reshape(-1, 1)

    # 透视投影到图像平面
    X_c, Y_c, Z_c = camera_points
    x = X_c / Z_c
    y = Y_c / Z_c

    # 相机内参变换
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    u = fx * x + cx
    v = fy * y + cy

    return u.reshape(X.shape), v.reshape(X.shape), x.reshape(X.shape), y.reshape(X.shape)


# 修改为1×3布局
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('相机内外参数成像效果对比\n畸变模型与网格变形分析', fontsize=16, fontweight='bold')

# 创建3D网格世界
x_world = np.linspace(-4, 4, 9)
y_world = np.linspace(-3, 3, 7)
z_world = 0  # 地面平面
X_world, Y_world = np.meshgrid(x_world, y_world)
Z_world = np.full_like(X_world, z_world)

# 添加输电线
t = np.linspace(-3, 3, 40)
cable_x = t
cable_y = np.zeros_like(t)
cable_z = 0.3 + 0.15 * t ** 2  # 抛物线形状的弧垂

# 相机参数设置
focal_length = 600
cx, cy = 320, 240
camera_matrix = np.array([[focal_length, 0, cx],
                          [0, focal_length, cy],
                          [0, 0, 1]])

# 相机外参
theta_x, theta_y, theta_z = np.deg2rad(15), np.deg2rad(-25), np.deg2rad(8)
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta_x), -np.sin(theta_x)],
               [0, np.sin(theta_x), np.cos(theta_x)]])
Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
               [0, 1, 0],
               [-np.sin(theta_y), 0, np.cos(theta_y)]])
Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
               [np.sin(theta_z), np.cos(theta_z), 0],
               [0, 0, 1]])
R = Rz @ Ry @ Rx
T = np.array([0.5, 0.2, -6])

# ======================== 子图1: 理想投影（无畸变） ========================
ax1 = axes[0]

u_ideal, v_ideal, x_norm, y_norm = camera_projection(X_world, Y_world, Z_world, camera_matrix, R, T)

# 绘制理想投影网格
for i in range(len(y_world)):
    mask = (~np.isnan(u_ideal[i, :])) & (u_ideal[i, :] > 50) & (u_ideal[i, :] < 590)
    if np.sum(mask) > 1:
        ax1.plot(u_ideal[i, mask], v_ideal[i, mask], 'b-', alpha=0.7, linewidth=1.5)
for j in range(len(x_world)):
    mask = (~np.isnan(u_ideal[:, j])) & (u_ideal[:, j] > 50) & (u_ideal[:, j] < 590)
    if np.sum(mask) > 1:
        ax1.plot(u_ideal[mask, j], v_ideal[mask, j], 'b-', alpha=0.7, linewidth=1.5)

# 投影输电线
u_cable, v_cable, _, _ = camera_projection(cable_x.reshape(1, -1),
                                           cable_y.reshape(1, -1),
                                           cable_z.reshape(1, -1),
                                           camera_matrix, R, T)
ax1.plot(u_cable.flatten(), v_cable.flatten(), 'r-', linewidth=4, label='输电线')
ax1.scatter([u_cable.flatten()[0], u_cable.flatten()[-1]],
            [v_cable.flatten()[0], v_cable.flatten()[-1]],
            c='red', s=80, marker='s', label='杆塔')

ax1.set_xlabel('u (像素)', fontsize=12)
ax1.set_ylabel('v (像素)', fontsize=12)
ax1.set_title('理想投影（无畸变）\n规则网格透视变换', fontweight='bold')
ax1.set_xlim(0, 640)
ax1.set_ylim(0, 480)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)
ax1.legend()

# ======================== 子图2: 径向畸变 ========================
ax2 = axes[1]

# 应用径向畸变
x_radial, y_radial = apply_distortion(x_norm, y_norm, k1=-0.35, k2=0.12, p1=0, p2=0)
u_radial = focal_length * x_radial + cx
v_radial = focal_length * y_radial + cy

# 绘制径向畸变网格
for i in range(len(y_world)):
    mask = (~np.isnan(u_radial[i, :])) & (u_radial[i, :] > 50) & (u_radial[i, :] < 590)
    if np.sum(mask) > 1:
        ax2.plot(u_radial[i, mask], v_radial[i, mask], 'g-', alpha=0.7, linewidth=1.5)
for j in range(len(x_world)):
    mask = (~np.isnan(u_radial[:, j])) & (u_radial[:, j] > 50) & (u_radial[:, j] < 590)
    if np.sum(mask) > 1:
        ax2.plot(u_radial[mask, j], v_radial[mask, j], 'g-', alpha=0.7, linewidth=1.5)

# 径向畸变后的输电线
cable_x_norm = (u_cable.flatten() - cx) / focal_length
cable_y_norm = (v_cable.flatten() - cy) / focal_length
cable_x_radial, cable_y_radial = apply_distortion(cable_x_norm, cable_y_norm, k1=-0.35, k2=0.12, p1=0, p2=0)
u_cable_radial = focal_length * cable_x_radial + cx
v_cable_radial = focal_length * cable_y_radial + cy

ax2.plot(u_cable_radial, v_cable_radial, 'r-', linewidth=4, label='输电线')
ax2.scatter([u_cable_radial[0], u_cable_radial[-1]],
            [v_cable_radial[0], v_cable_radial[-1]],
            c='red', s=80, marker='s', label='杆塔')

ax2.set_xlabel('u (像素)', fontsize=12)
ax2.set_ylabel('v (像素)', fontsize=12)
ax2.set_title('径向畸变效果\nk1=-0.35, k2=0.12', fontweight='bold')
ax2.set_xlim(0, 640)
ax2.set_ylim(0, 480)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3)
ax2.legend()

# ======================== 子图3: 完整畸变 ========================
ax3 = axes[2]

# 应用完整畸变（径向+切向）
x_full, y_full = apply_distortion(x_norm, y_norm, k1=-0.28, k2=0.08, p1=0.015, p2=0.012)
u_full = focal_length * x_full + cx
v_full = focal_length * y_full + cy

# 绘制完整畸变网格
for i in range(len(y_world)):
    mask = (~np.isnan(u_full[i, :])) & (u_full[i, :] > 50) & (u_full[i, :] < 590)
    if np.sum(mask) > 1:
        ax3.plot(u_full[i, mask], v_full[i, mask], 'purple', alpha=0.7, linewidth=1.5)
for j in range(len(x_world)):
    mask = (~np.isnan(u_full[:, j])) & (u_full[:, j] > 50) & (u_full[:, j] < 590)
    if np.sum(mask) > 1:
        ax3.plot(u_full[mask, j], v_full[mask, j], 'purple', alpha=0.7, linewidth=1.5)

# 完整畸变后的输电线
cable_x_full, cable_y_full = apply_distortion(cable_x_norm, cable_y_norm,
                                              k1=-0.28, k2=0.08, p1=0.015, p2=0.012)
u_cable_full = focal_length * cable_x_full + cx
v_cable_full = focal_length * cable_y_full + cy

ax3.plot(u_cable_full, v_cable_full, 'r-', linewidth=4, label='输电线')
ax3.scatter([u_cable_full[0], u_cable_full[-1]],
            [v_cable_full[0], v_cable_full[-1]],
            c='red', s=80, marker='s', label='杆塔')

ax3.set_xlabel('u (像素)', fontsize=12)
ax3.set_ylabel('v (像素)', fontsize=12)
ax3.set_title('完整畸变模型\n径向+切向畸变', fontweight='bold')
ax3.set_xlim(0, 640)
ax3.set_ylim(0, 480)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3)
ax3.legend()

# 调整子图间距
plt.tight_layout()
plt.show()
