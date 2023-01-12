#!env python3
# Usage:
#   python 训练准确率 0.605 0.656 0.683
import matplotlib.pyplot as plt
import sys

plt.rcParams['font.sans-serif']=['SimHei']

fig, ax = plt.subplots()

title = sys.argv[1]

items = ['第一次', '第二次', '第三次', '平均']
val = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
val.append(sum(val) / len(val))

p = ax.bar(items, val)

ax.set_ylabel('比率')
ax.set_title(title)
ax.bar_label(p)

plt.savefig(f"{title}.svg")
