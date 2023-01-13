#!env python3
# Usage:
#   python {TITLE} {YLABEL} {DATA_FILE}

import matplotlib.pyplot as plt
import sys

plt.rcParams['font.sans-serif']=['SimHei']

title = sys.argv[1]
ylabel = sys.argv[2]
data = sys.argv[3]

with open(data) as f:
  train = [float(x) for x in f.readline().split()]
  val = [float(x) for x in f.readline().split()]

assert(len(train) == len(val))

epochs = range(1, len(train) + 1)

plt.title(title)
plt.plot(epochs, train, 'bo', label='Training')
plt.plot(epochs, val, 'b', label='Validation')
plt.xlabel('Epochs')
plt.ylabel(ylabel)
plt.legend()

plt.savefig(f"{title}.svg")
