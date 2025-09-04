import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import warnings
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore')

# 设置更美观的样式
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 10
})

# 使用更美观的调色板
colors = plt.cm.viridis(np.linspace(0, 1, 6))
accent_color = plt.cm.Set2(np.linspace(0, 1, 8))

# 给定数据
x_data = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
y_data = np.array([30.1, 29.31, 30.15, 31.33, 35.81, 34.62, 44.52, 50.82, 59.16, 73.14, 80])

print("输电线路数据点：")
for i in range(len(x_data)):
    print(f"点{i + 1}: ({x_data[i]}, {y_data[i]})")

# 两端挂点
x1, y1 = x_data[0], y_data[0]  # 起点
x2, y2 = x_data[-1], y_data[-1]  # 终点
L = x2 - x1  # 总长度

print(f"\n挂点信息：")
print(f"起点：({x1}, {y1})")
print(f"终点：({x2}, {y2})")
print(f"水平跨距：{L}m")


# 定义拟合函数：y(x) = a*cosh((x-x0)/a) + c + δ(x)
# 其中 δ(x) = b1*x + b2*x² + b3*sin(π*x/L)
def catenary_model(x, a, x0, c, b1, b2, b3):
    """
    悬链线模型 + 修正项
    a: 悬链线参数
    x0: 水平偏移
    c: 垂直偏移
    b1, b2, b3: 修正项系数
    """
    # 避免数值溢出
    arg = (x - x0) / a
    arg = np.clip(arg, -50, 50)  # 限制参数范围

    # 主要悬链线项
    y_catenary = a * np.cosh(arg) + c

    # 修正项
    delta = b1 * x + b2 * x ** 2 + b3 * np.sin(np.pi * x / L)

    return y_catenary + delta


# 初始参数估计
# 对于悬链线，最低点通常在中间附近
x0_init = L / 2
a_init = L / 4  # 经验估计
c_init = np.min(y_data) - a_init
b1_init = (y2 - y1) / L  # 线性趋势
b2_init = 0
b3_init = 0

initial_params = [a_init, x0_init, c_init, b1_init, b2_init, b3_init]

try:
    # 拟合模型
    popt, pcov = curve_fit(catenary_model, x_data, y_data,
                           p0=initial_params,
                           maxfev=5000)

    a_fit, x0_fit, c_fit, b1_fit, b2_fit, b3_fit = popt

    print(f"\n拟合参数：")
    print(f"a = {a_fit:.4f}")
    print(f"x0 = {x0_fit:.4f}")
    print(f"c = {c_fit:.4f}")
    print(f"b1 = {b1_fit:.6f}")
    print(f"b2 = {b2_fit:.8f}")
    print(f"b3 = {b3_fit:.4f}")

    # 生成拟合曲线
    x_fit = np.linspace(0, 500, 1000)
    y_fit = catenary_model(x_fit, *popt)

    # 计算拟合优度
    y_pred = catenary_model(x_data, *popt)
    r_squared = 1 - np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
    rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))

    print(f"\n拟合质量：")
    print(f"R² = {r_squared:.4f}")
    print(f"RMSE = {rmse:.4f}m")

except Exception as e:
    print(f"拟合失败: {e}")

    # 使用简化模型
    def simple_catenary(x, a, c, b1):
        return a * np.cosh(x / a) + c + b1 * x

    popt, _ = curve_fit(simple_catenary, x_data, y_data, p0=[100, 0, 0])
    a_fit, c_fit, b1_fit = popt
    x_fit = np.linspace(0, 500, 1000)
    y_fit = simple_catenary(x_fit, *popt)


# 计算弧垂
def calculate_sag_distance(x, y, x1, y1, x2, y2):
    """计算点(x,y)到两端连线的垂直距离"""
    # 两端连线方程: (y-y1)/(y2-y1) = (x-x1)/(x2-x1)
    # 即: (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x1 = 0
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x1

    # 点到直线距离公式
    distance = np.abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
    return distance


# 计算两端连线方程: y = mx + c
m = (y2 - y1) / L  # 斜率
line_c = y1  # y轴截距
line_eq = lambda x: m * x + line_c


# 计算所有点的弧垂
x_dense = np.linspace(0, 500, 10000)
sag_distances = [calculate_sag_distance(x, catenary_model(x, *popt), x1, y1, x2, y2) for x in x_dense]
sag_distances = np.array(sag_distances)

# 找到最大弧垂
max_sag_idx = np.argmax(sag_distances)
max_sag = sag_distances[max_sag_idx]
max_sag_x = x_dense[max_sag_idx]
max_sag_y = catenary_model(max_sag_x, *popt)

print(f"\n=== 最大弧垂结果 ===")
print(f"最大弧垂值：{max_sag:.4f} m")
print(f"最大弧垂位置：x = {max_sag_x:.2f} m")
print(f"该点高度：y = {max_sag_y:.4f} m")

# 计算两端连线方程上对应点的高度
line_y_at_max_sag = line_eq(max_sag_x)
print(f"两端连线在该位置的高度：{line_y_at_max_sag:.4f} m")
print(f"实际下垂距离：{line_y_at_max_sag - max_sag_y:.4f} m")

# 计算拟合误差
residuals = y_data - catenary_model(x_data, *popt)

# ---------- 图1: 主图 - 输电线路拟合与最大弧垂 ----------
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制背景区域
ax.fill_between(x_fit,
                np.ones_like(x_fit) * (min(y_data) - 5),
                line_eq(x_fit),
                color='lightblue', alpha=0.3, label='_nolegend_')

# 绘制拟合曲线
ax.plot(x_fit, y_fit, color=colors[0], linewidth=3, label='悬链线拟合曲线')

# 绘制端点连线
ax.plot([x1, x2], [y1, y2], color=colors[2], linestyle='--', linewidth=2, label='两端连线')

# 绘制最大弧垂线
ax.plot([max_sag_x, max_sag_x],
        [line_eq(max_sag_x), max_sag_y],
        color='red', linewidth=2.5, label='最大弧垂')

# 绘制原始数据点
ax.scatter(x_data, y_data, color=accent_color[1], s=80, zorder=5, label='实测数据点')

# 突出显示最大弧垂点
ax.scatter(max_sag_x, max_sag_y, color='red', s=150, marker='*', zorder=6, label='最大弧垂点')

# 添加标注
ax.annotate(f'最大弧垂: {max_sag:.2f}m',
            xy=(max_sag_x, (max_sag_y + line_eq(max_sag_x)) / 2),
            xytext=(max_sag_x + 50, (max_sag_y + line_eq(max_sag_x)) / 2 + 5),
            arrowprops=dict(facecolor='black', arrowstyle='fancy', connectionstyle="arc3,rad=.3"),
            fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

# 设置标题和标签
ax.set_title('输电线路悬链线拟合与最大弧垂分析', fontweight='bold', pad=15)
ax.set_xlabel('水平距离 x (m)')
ax.set_ylabel('高度 y (m)')

# 设置y轴范围
y_min, y_max = min(y_data) - 8, max(y_data) + 8
ax.set_ylim(y_min, y_max)

# 显示网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax.legend(loc='upper left', framealpha=0.9)

# 添加关键数据标签
for i, (x, y) in enumerate(zip(x_data, y_data)):
    if i == 0 or i == len(x_data) - 1:  # 只标注两端点
        ax.annotate(f"({x}, {y})",
                    xy=(x, y),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
try:
    plt.savefig('输电线路悬链线拟合分析.png', dpi=300, bbox_inches='tight')
    print("主图保存成功！")
except Exception as e:
    print(f"保存图片时出错: {e}")
plt.show()

# ---------- 图2: 弧垂分布图 ----------
fig, ax = plt.subplots(figsize=(12, 6))

# 创建渐变色填充
x_points = np.linspace(0, 500, 1000)
y_points = [calculate_sag_distance(x, catenary_model(x, *popt), x1, y1, x2, y2) for x in x_points]
points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, max_sag)
lc = mpl.collections.LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(np.array(y_points))
lc.set_linewidth(3)
line = ax.add_collection(lc)

# 添加颜色条
cbar = fig.colorbar(line, ax=ax)
cbar.set_label('弧垂大小 (m)', fontsize=12)

# 标注最大弧垂位置
ax.axvline(x=max_sag_x, color='red', linestyle='--', alpha=0.7)
ax.scatter(max_sag_x, max_sag, color='red', s=120, marker='*')

# 添加标注
ax.annotate(f'最大弧垂: {max_sag:.2f}m',
            xy=(max_sag_x, max_sag),
            xytext=(max_sag_x - 150, max_sag + 1),
            arrowprops=dict(facecolor='black', arrowstyle='fancy', connectionstyle="arc3,rad=.3"),
            fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

# 设置坐标轴
ax.set_xlim(0, 500)
ax.set_ylim(0, max_sag * 1.1)
ax.xaxis.set_major_locator(MaxNLocator(10))
ax.yaxis.set_major_locator(MaxNLocator(10))

# 设置标题和标签
ax.set_title('输电线路弧垂分布图', fontweight='bold')
ax.set_xlabel('水平距离 x (m)')
ax.set_ylabel('弧垂大小 (m)')
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
try:
    plt.savefig('输电线路弧垂分布图.png', dpi=300, bbox_inches='tight')
    print("弧垂分布图保存成功！")
except Exception as e:
    print(f"保存图片时出错: {e}")
plt.show()

# ---------- 图3: 拟合误差分析 ----------
fig, ax = plt.subplots(figsize=(12, 6))

# 创建渐变色柱状图
bars = ax.bar(x_data, residuals, width=20, alpha=0.8)

# 为柱状图设置不同的颜色
for i, bar in enumerate(bars):
    if residuals[i] >= 0:
        bar.set_color(accent_color[0])
    else:
        bar.set_color(accent_color[1])

    # 添加数值标签
    height = residuals[i]
    ax.text(x_data[i], height + np.sign(height) * 0.1,
            f"{height:.2f}",
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=9)

# 添加误差均值和标准差线
mean_error = np.mean(residuals)
std_error = np.std(residuals)
ax.axhline(y=mean_error, color='red', linestyle='-', linewidth=1.5, label=f'均值: {mean_error:.2f}m')
ax.axhline(y=mean_error + std_error, color='orange', linestyle='--', linewidth=1,
           label=f'标准差: {std_error:.2f}m')
ax.axhline(y=mean_error - std_error, color='orange', linestyle='--', linewidth=1)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 添加RMSE指标
ax.text(0.02, 0.95, f"均方根误差(RMSE): {rmse:.4f}m",
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

# 设置标题和标签
ax.set_title('悬链线拟合误差分析', fontweight='bold')
ax.set_xlabel('水平距离 x (m)')
ax.set_ylabel('误差 (m)')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

# 调整y轴范围以便更好地显示误差
max_abs_error = max(abs(max(residuals)), abs(min(residuals)))
ax.set_ylim(-max_abs_error * 1.2, max_abs_error * 1.2)

plt.tight_layout()
try:
    plt.savefig('输电线路拟合误差分析.png', dpi=300, bbox_inches='tight')
    print("拟合误差分析图保存成功！")
except Exception as e:
    print(f"保存图片时出错: {e}")
plt.show()


# ---------- 图4: 弧垂沿线分布示意图 ----------
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制悬链线曲线
ax.plot(x_fit, y_fit, color=colors[0], linewidth=3, label='悬链线')

# 绘制端点连线
ax.plot([x1, x2], [y1, y2], color=colors[2], linestyle='--', linewidth=2, label='端点连线')

# 创建弧垂示意图
step = 50  # 每隔50m显示一个弧垂线
cmap = plt.cm.plasma
for i, x_pos in enumerate(np.arange(0, 501, step)):
    if x_pos == 0 or x_pos == 500:  # 跳过端点
        continue

    y_curve = catenary_model(x_pos, *popt)
    y_line = line_eq(x_pos)
    sag = calculate_sag_distance(x_pos, y_curve, x1, y1, x2, y2)

    # 使用渐变色表示弧垂大小
    color = cmap(sag / max_sag)

    # 绘制弧垂线
    ax.plot([x_pos, x_pos], [y_line, y_curve], color=color, linewidth=2)

    # 添加弧垂大小标签
    if sag > 1:  # 只为较大的弧垂添加标签
        ax.text(x_pos + 5, (y_curve + y_line) / 2, f"{sag:.2f}m", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

# 突出显示最大弧垂
ax.plot([max_sag_x, max_sag_x],
        [line_eq(max_sag_x), max_sag_y],
        color='red', linewidth=3, label=f'最大弧垂: {max_sag:.2f}m')
ax.scatter(max_sag_x, max_sag_y, color='red', s=120, marker='*')

# 添加彩色标签，表示弧垂的大小
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_sag))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('弧垂大小 (m)')

# 设置标题和标签
ax.set_title('输电线路弧垂沿线分布示意图', fontweight='bold')
ax.set_xlabel('水平距离 x (m)')
ax.set_ylabel('高度 y (m)')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left')

# 设置y轴范围
y_min, y_max = min(y_data) - 5, max(y_data) + 5
ax.set_ylim(y_min, y_max)

plt.tight_layout()
try:
    plt.savefig('输电线路弧垂沿线分布.png', dpi=300, bbox_inches='tight')
    print("弧垂沿线分布图保存成功！")
except Exception as e:
    print(f"保存图片时出错: {e}")
plt.show()

# 输出模型方程
print(f"\n=== 拟合模型方程 ===")
print(f"y(x) = {a_fit:.4f} × cosh((x - {x0_fit:.2f})/{a_fit:.4f}) + {c_fit:.4f}")
print(f"     + {b1_fit:.6f}×x + {b2_fit:.8f}×x² + {b3_fit:.4f}×sin(π×x/{L})")
