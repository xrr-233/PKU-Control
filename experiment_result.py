import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

res = np.array([
    [103.852,    99.72,      133.018,    169.35,     94.752],
    [116.644,    257.388,    190.218,    432.262,    322.3],
    [101.836,    246.132,    342.448,    402.892,    344.758],
    [160.372,    225.986,    407.938,    398.23,     249.902],
    [306.076,    289.7,      382.672,    223.37,     375.392],
])
x_ticks = np.arange(1, 5 + 1, 1)
y_ticks = [9, 18, 36, 48, 72]  # 自定义横纵轴
ax = sns.heatmap(res, xticklabels=x_ticks, yticklabels=y_ticks)
ax.set_xlabel('Layers')
ax.set_ylabel('Dim')
plt.show()
figure = ax.get_figure()
figure.savefig('heatmap.png')
