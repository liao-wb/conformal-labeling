import matplotlib.pyplot as plt
import numpy as np

# 设置图形尺寸为4.5x3.5
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.set_xlim(0.7, 2.1)
ax.set_ylim(-0.4, 0.5)
ax.axis('off')

# 进一步减小点之间的间距
calib_positions = [1.0, 1.5, 2.0]  # 间距从0.6减小到0.5
test_positions = [1.25, 1.75]  # 间距从0.6减小到0.5

# 绘制点和线
ax.scatter(calib_positions, [0]*len(calib_positions), c='blue', s=500, zorder=3, alpha=0.6)
ax.plot(calib_positions, [0]*len(calib_positions), 'b-', linewidth=2.5, alpha=0.3)

ax.scatter(test_positions, [0]*len(test_positions), c='orange', s=500, zorder=3, edgecolors='black', linewidth=1.2)
ax.plot(test_positions, [0]*len(test_positions), 'orange', linewidth=2.5, alpha=0.3)

# 调整文本位置和大小
ax.text(1.25, -0.2, r"$\frac{\mathbf{1}}{\mathbf{n_0+1}}$", ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(1.75, -0.2, r"$\frac{\mathbf{2}}{\mathbf{n_0+1}}$", ha='center', va='top', fontsize=20, fontweight='bold')

all_positions = sorted(calib_positions + test_positions)
ax.plot(all_positions, [0]*len(all_positions), 'k-', linewidth=1.8, alpha=0.5)

# 文字标签 - 调整位置和大小
ax.text(0.8, -0.05, r'$\hat{s}(x)$', ha='center', va='bottom', fontsize=20, fontweight='bold')
ax.text(0.8, -0.22, r'$\hat{p}(x)$', ha='center', va='top', fontsize=20, fontweight='bold')

# 图例 - 调整位置和大小使其更紧凑
mov = 1.31  # 减小移动距离
y_mov = 0.2
ax.add_patch(plt.Rectangle((2.2 - mov, 0.35 - y_mov), 1.2, 0.22, fill=False, edgecolor='black', linewidth=1.2))
ax.text(2.4 - mov, 0.45-y_mov, 'calibration', ha='left', va='center', fontsize=16, fontweight='bold')
ax.scatter(2.3 - mov, 0.45-y_mov, c='blue', s=300, alpha=0.6)

ax.text(3.13 - mov, 0.45-y_mov, 'test', ha='left', va='center', fontsize=16, fontweight='bold')
ax.scatter(3.03 - mov, 0.45-y_mov, c='orange', s=300)

# 设置全局字体大小
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'stix'

# 调整边距
plt.tight_layout(pad=0.1)
plt.subplots_adjust(left=0.07, right=0.93, bottom=0.05, top=0.95)

# 保存为高DPI图片
plt.savefig('conformal_plot.pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05,
            facecolor='white', edgecolor='none')

plt.show()