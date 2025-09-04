import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(14, 10))

# 悬链线参数
a_values = [1, 2, 3, 4]  # 不同的参数a值
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
labels = [f'a = {a}' for a in a_values]

# x轴范围
x = np.linspace(-6, 6, 1000)

# 绘制多条悬链线
for i, (a, color, label) in enumerate(zip(a_values, colors, labels)):
    y = a * np.cosh(x / a)
    ax.plot(x, y, color=color, linewidth=3, label=label, alpha=0.8)

    # 在每条曲线上添加一些装饰点
    x_points = np.linspace(-4, 4, 7)
    y_points = a * np.cosh(x_points / a)
    ax.scatter(x_points, y_points, color=color, s=50, alpha=0.7, zorder=5)

# 绘制坐标轴
ax.axhline(y=0, color='black', linewidth=1, alpha=0.3)
ax.axvline(x=0, color='black', linewidth=1, alpha=0.3)

# 添加网格
ax.grid(True, alpha=0.2, linestyle='--')

# 设置坐标轴范围和标签
ax.set_xlim(-6, 6)
ax.set_ylim(0, 12)
ax.set_xlabel('x', fontsize=16, fontweight='bold')
ax.set_ylabel('y', fontsize=16, fontweight='bold')

# 添加标题
title = ax.set_title('悬链线方程', fontsize=20, fontweight='bold', pad=20)

# 添加数学公式
formula_text = r'$y = a \cdot \cosh\left(\frac{x}{a}\right)$'
ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                  edgecolor='gray', alpha=0.9))

# 添加说明文本

# 添加图例
legend = ax.legend(loc='upper right', fontsize=14, frameon=True,
                   fancybox=True, shadow=True, framealpha=0.9)
legend.get_frame().set_facecolor('white')

# 美化坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# 设置刻度样式
ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=3)

# 添加一些物理解释的箭头和标注
# 标注最小值点
ax.annotate('最低点: y = a', xy=(0, a_values[0]), xytext=(1.5, a_values[0] + 1),
            fontsize=12, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 添加渐近线说明


# 设置背景色
ax.set_facecolor('#FAFAFA')

# 调整布局
plt.tight_layout()

# 保存高质量图片
plt.savefig('catenary_curve_2d.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 显示图形
plt.show()

# 输出一些数学性质
print("\n" + "=" * 50)
print("悬链线数学性质")
print("=" * 50)
for a in a_values:
    print(f"当 a = {a} 时:")
    print(f"  - 最低点: y = {a}")
    print(f"  - 在 x = ±{a:.1f} 处: y = {a * np.cosh(1):.2f}")
    print(f"  - 曲率半径(最低点): ρ = {a}")
    print()
