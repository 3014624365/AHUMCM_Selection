import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# 设置中文字体和高DPI
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 创建更大的画布和子图
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[4, 1])
ax_main = fig.add_subplot(gs[0, :], projection='3d')

# 电力线参数 - 调整为更少的线和更宽的间距
span_length = 300  # 跨距长度
tower_height = 120  # 电塔高度
num_lines = 3  # 减少为3根线(单回路三相)
line_spacing = 15  # 增加线间距离到15米
sag_factor = 0.04  # 垂弧系数

# 计算悬链线参数
a = span_length / (2 * np.arccosh(1 + sag_factor))
sag = a * (np.cosh(span_length / (2 * a)) - 1)

# 生成高精度电力线坐标
x = np.linspace(-span_length / 2, span_length / 2, 1000)

# 导线配置 - 单回路布局，间距更宽
colors = ['#FF2D2D', '#32CD32', '#1E90FF']  # 红绿蓝三相
line_labels = ['A相', 'B相', 'C相']
line_positions = [
    (-15, 0),  # A相，左侧
    (0, -10),  # B相，中间偏下
    (15, 0)  # C相，右侧
]

# 绘制导线束（简化为单根导线，不使用分裂导线）
for i, (y_pos, z_offset) in enumerate(line_positions):
    # 计算悬链线高度
    z_base = tower_height - 15 + z_offset
    z_line = z_base - a + a * np.cosh(x / a)

    # 绘制主导线（更粗的线径）
    y_line = np.full_like(x, y_pos)
    ax_main.plot(x, y_line, z_line, color=colors[i], linewidth=8,
                 label=line_labels[i], alpha=0.9, zorder=5)

    # 添加导线光泽效果（顶部高光）
    z_highlight = z_line + 0.3
    ax_main.plot(x, y_line, z_highlight, color='white', linewidth=2,
                 alpha=0.6, zorder=6)

   

# 增强版电塔绘制函数 - 适配更宽的导线布局
def draw_enhanced_tower(x_pos, height):
    # 塔身参数 - 加宽以适应更大的导线间距
    base_width = 25
    top_width = 6
    leg_height = height * 0.85

    # 四条主腿
    base_points = np.array([
        [x_pos - base_width / 2, -base_width / 2, 0],
        [x_pos + base_width / 2, -base_width / 2, 0],
        [x_pos + base_width / 2, base_width / 2, 0],
        [x_pos - base_width / 2, base_width / 2, 0]
    ])

    leg_top_points = np.array([
        [x_pos - top_width / 2, -top_width / 2, leg_height],
        [x_pos + top_width / 2, -top_width / 2, leg_height],
        [x_pos + top_width / 2, top_width / 2, leg_height],
        [x_pos - top_width / 2, top_width / 2, leg_height]
    ])

    # 绘制主腿
    for i in range(4):
        ax_main.plot([base_points[i, 0], leg_top_points[i, 0]],
                     [base_points[i, 1], leg_top_points[i, 1]],
                     [base_points[i, 2], leg_top_points[i, 2]],
                     'k-', linewidth=6, alpha=0.8)

    # 绘制横向连接
    heights = [height * 0.2, height * 0.4, height * 0.6, height * 0.8]
    for h in heights:
        width_at_h = base_width - (base_width - top_width) * (h / leg_height)
        points_at_h = np.array([
            [x_pos - width_at_h / 2, -width_at_h / 2, h],
            [x_pos + width_at_h / 2, -width_at_h / 2, h],
            [x_pos + width_at_h / 2, width_at_h / 2, h],
            [x_pos - width_at_h / 2, width_at_h / 2, h]
        ])

        for i in range(4):
            next_i = (i + 1) % 4
            ax_main.plot([points_at_h[i, 0], points_at_h[next_i, 0]],
                         [points_at_h[i, 1], points_at_h[next_i, 1]],
                         [points_at_h[i, 2], points_at_h[next_i, 2]],
                         'k-', linewidth=3, alpha=0.7)

    # 绘制斜撑
    for i in range(4):
        for h in heights[1:]:
            width_at_h = base_width - (base_width - top_width) * (h / leg_height)
            ax_main.plot([base_points[i, 0], x_pos + width_at_h / 4 * (-1) ** i],
                         [base_points[i, 1], width_at_h / 4 * (-1) ** (i // 2)],
                         [0, h], 'k-', linewidth=2, alpha=0.5)

    # 绘制横担结构 - 加长以适应更宽的导线间距
    crossarm_configs = [
        (height - 8, 40, 8),  # 上层横担，加长到40米
        (height - 16, 35, 6),  # 下层横担
    ]

    for arm_height, arm_length, arm_width in crossarm_configs:
        # 主横担
        ax_main.plot([x_pos - arm_length / 2, x_pos + arm_length / 2],
                     [0, 0], [arm_height, arm_height],
                     'k-', linewidth=arm_width, alpha=0.9)

        # 横担支撑
        support_points = [-arm_length / 2, -arm_length / 4, 0, arm_length / 4, arm_length / 2]
        for sp in support_points:
            ax_main.plot([x_pos + sp, x_pos + sp], [0, 0],
                         [leg_height, arm_height], 'k-', linewidth=3, alpha=0.7)

    # 绘制避雷针
    lightning_height = height + 15
    ax_main.plot([x_pos, x_pos], [0, 0], [height, lightning_height],
                 'r-', linewidth=4, alpha=0.9)
    ax_main.scatter([x_pos], [0], [lightning_height],
                    s=200, c='red', marker='^', alpha=0.9)

    # 绘制绝缘子串 - 对应三根导线的位置
    insulator_positions = [
        (-15, -3, height - 10),  # A相绝缘子
        (0, -13, height - 18),  # B相绝缘子
        (15, -3, height - 10)  # C相绝缘子
    ]

    for ins_x, ins_y, ins_z in insulator_positions:
        # 绝缘子串
        for j in range(12):  # 12片绝缘子，增加数量
            z_ins = ins_z - j * 0.6
            ax_main.scatter([x_pos + ins_x], [ins_y], [z_ins],
                            s=120, c='white', edgecolors='gray', alpha=0.9)

        # 连接横担到绝缘子的吊线
        ax_main.plot([x_pos, x_pos + ins_x], [0, ins_y],
                     [height - 8, ins_z], 'k-', linewidth=3, alpha=0.7)

        # 绝缘子底部到导线的连接
        wire_z = tower_height - 15 + (0 if ins_x != 0 else -10)
        ax_main.plot([x_pos + ins_x, x_pos + ins_x], [ins_y, ins_y],
                     [ins_z - 7, wire_z], 'k-', linewidth=2, alpha=0.8)


# 绘制增强版电塔
draw_enhanced_tower(-span_length / 2, tower_height)
draw_enhanced_tower(span_length / 2, tower_height)

# 添加地线（架空地线，位置在导线上方）
ground_wire_height = tower_height + 5
ground_wire_x = x
ground_wire_y = np.zeros_like(x)
ground_wire_z = ground_wire_height - 2 + 2 * np.cosh(x / (a * 1.2))  # 地线垂弧更小

ax_main.plot(ground_wire_x, ground_wire_y, ground_wire_z,
             color='silver', linewidth=4, alpha=0.8,
             label='架空地线', zorder=5)

# 创建更真实的地形
terrain_size = 400
terrain_x = np.linspace(-terrain_size, terrain_size, 50)
terrain_y = np.linspace(-terrain_size / 2, terrain_size / 2, 25)
Terrain_X, Terrain_Y = np.meshgrid(terrain_x, terrain_y)

# 添加地形起伏
terrain_noise = 3 * np.sin(Terrain_X / 50) * np.cos(Terrain_Y / 30) + \
                2 * np.sin(Terrain_X / 80) * np.sin(Terrain_Y / 40)
Terrain_Z = terrain_noise

# 绘制地形
ax_main.plot_surface(Terrain_X, Terrain_Y, Terrain_Z,
                     alpha=0.3, cmap='terrain', linewidth=0, antialiased=True)

# 添加道路网络
road_main_x = np.linspace(-terrain_size, terrain_size, 200)
road_main_y = np.zeros_like(road_main_x)
road_main_z = 3 * np.sin(road_main_x / 50) * np.cos(0 / 30) + 0.5

ax_main.plot(road_main_x, road_main_y, road_main_z,
             color='gray', linewidth=6, alpha=0.8, zorder=3)

# 添加道路标线
ax_main.plot(road_main_x, road_main_y - 1, road_main_z + 0.1,
             color='white', linewidth=2, alpha=0.9, zorder=4)
ax_main.plot(road_main_x, road_main_y + 1, road_main_z + 0.1,
             color='white', linewidth=2, alpha=0.9, zorder=4)

# 添加更多环境细节
# 树林
forest_positions = [
    (-200, 80, 15), (-180, 100, 12), (-160, 90, 18),
    (160, -80, 14), (180, -100, 16), (200, -70, 13),
    (-100, 180, 10), (100, -150, 11)
]

for tree_x, tree_y, tree_h in forest_positions:
    tree_z_base = 3 * np.sin(tree_x / 50) * np.cos(tree_y / 30)
    # 树干
    ax_main.plot([tree_x, tree_x], [tree_y, tree_y],
                 [tree_z_base, tree_z_base + tree_h],
                 color='#8B4513', linewidth=5, alpha=0.8)
    # 树冠
    ax_main.scatter([tree_x], [tree_y], [tree_z_base + tree_h + 3],
                    s=1000, c='darkgreen', alpha=0.7, marker='o')

# 添建筑物
building_positions = [(-280, -120), (280, 140)]
for bld_x, bld_y in building_positions:
    bld_z_base = 3 * np.sin(bld_x / 50) * np.cos(bld_y / 30)
    bld_height = 25

    # 简单建筑轮廓
    building_x = [bld_x - 10, bld_x + 10, bld_x + 10, bld_x - 10, bld_x - 10]
    building_y = [bld_y - 8, bld_y - 8, bld_y + 8, bld_y + 8, bld_y - 8]
    building_z_bottom = [bld_z_base] * 5
    building_z_top = [bld_z_base + bld_height] * 5

    # 绘制建筑
    for i in range(4):
        ax_main.plot([building_x[i], building_x[i]],
                     [building_y[i], building_y[i]],
                     [building_z_bottom[i], building_z_top[i]],
                     'gray', linewidth=3, alpha=0.8)

    ax_main.plot(building_x, building_y, building_z_bottom,
                 'gray', linewidth=2, alpha=0.8)
    ax_main.plot(building_x, building_y, building_z_top,
                 'gray', linewidth=2, alpha=0.8)

# 添加天空渐变背景色
ax_main.xaxis.set_pane_color((0.8, 0.9, 1.0, 0.3))
ax_main.yaxis.set_pane_color((0.8, 0.9, 1.0, 0.3))
ax_main.zaxis.set_pane_color((0.9, 0.95, 1.0, 0.3))

# 设置坐标轴标签和范围
ax_main.set_xlabel('跨距方向 (m)', fontsize=14, labelpad=10)
ax_main.set_ylabel('线路横向 (m)', fontsize=14, labelpad=10)
ax_main.set_zlabel('海拔高度 (m)', fontsize=14, labelpad=10)

ax_main.set_xlim(-terrain_size, terrain_size)
ax_main.set_ylim(-terrain_size / 2, terrain_size / 2)
ax_main.set_zlim(-20, tower_height + 30)

# 设置标题
title = ax_main.set_title('220kV单回路架空输电线路三维仿真\n简化设计 - 悬链线力学建模',
                          fontsize=18, pad=30, weight='bold')

# 添加详细的技术参数信息框
info_text = f"""工程技术参数:
━━━━━━━━━━━━━
▪ 电压等级: 220kV
▪ 回路数量: 单回路
▪ 导线型号: LGJ-400/35
▪ 相线数量: {num_lines}根 (三相)
▪ 跨距长度: {span_length}m
▪ 杆塔高度: {tower_height}m
▪ 最大垂弧: {sag:.1f}m
▪ 垂弧率: {sag / span_length * 100:.2f}%
▪ 悬链线参数a: {a:.1f}m
▪ 相间距离: {line_spacing}m
▪ 导线布置: 水平排列
━━━━━━━━━━━━━
▪ 地线: 1根架空地线
▪ 绝缘子: 瓷质悬式绝缘子
▪ 防振: 防振锤"""

# 创建信息面板
ax_info = fig.add_subplot(gs[1, 0])
ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
             verticalalignment='top', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen',
                       alpha=0.8, edgecolor='darkgreen'))
ax_info.axis('off')

# 创建颜色图例
ax_legend = fig.add_subplot(gs[1, 1])
legend_elements = []
for i, (color, label) in enumerate(zip(colors, line_labels)):
    legend_elements.append(plt.Line2D([0], [0], color=color, lw=6, label=label))
legend_elements.append(plt.Line2D([0], [0], color='silver', lw=4, label='架空地线'))

ax_legend.legend(handles=legend_elements, loc='center', fontsize=12,
                 title='导线相别标识', title_fontsize=14)
ax_legend.axis('off')

# 设置最佳观察角度
ax_main.view_init(elev=25, azim=-60)

# 优化网格和轴
ax_main.grid(True, alpha=0.2)
ax_main.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
ax_main.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
ax_main.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)

# 调整整体布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)

# 保存高质量图像
plt.savefig('simplified_power_line_3d.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 显示图像
plt.show()

# 输出工程计算结果
print("\n" + "=" * 50)
print("简化架空输电线路工程计算结果")
print("=" * 50)
print(f"导线数量: {num_lines}根 (三相)")
print(f"相间距离: {line_spacing}m")
print(f"悬链线参数 a = {a:.2f} m")
print(f"最大垂弧 f = {sag:.2f} m")
print(f"垂弧率 f/l = {sag / span_length * 100:.3f}%")
print(f"导线最低点高度 = {tower_height - sag:.1f} m")
print(f"对地安全距离 = {(tower_height - sag) - max(terrain_noise):.1f} m")
print("=" * 50)
