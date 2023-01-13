#!env python3
# Usage:
#   python bar 0.678 0.673 0.683
import matplotlib.pyplot as plt
import sys

plt.rcParams['font.sans-serif']=['SimHei']

fig, ax = plt.subplots()

title = sys.argv[1]

items = ['1', '2', '3', 'Average']
val = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
val.append(sum(val) / len(val))

p = ax.bar(items, val)

ax.set_ylabel('Accuracy')
ax.bar_label(p)

plt.savefig(f"{title}.svg")
