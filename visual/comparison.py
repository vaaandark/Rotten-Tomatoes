#!env python3
# Usage:
#   python {TITLE} {YLABEL} {DATA_FILE}

import matplotlib.pyplot as plt
import sys

title = sys.argv[1]
ylabel = sys.argv[2]
data = sys.argv[3]

with open(data) as f:
  cross_entropy_loss_train = [float(x) for x in f.readline().split()]
  cross_entropy_loss_val = [float(x) for x in f.readline().split()]
  nll_loss_train = [float(x) for x in f.readline().split()]
  nll_loss_val = [float(x) for x in f.readline().split()]

assert(len(cross_entropy_loss_train) == len(cross_entropy_loss_val))

epochs = range(1, len(cross_entropy_loss_val) + 1)

plt.plot(epochs, cross_entropy_loss_train, 'bo', label='CrossEntropyLoss Training')
plt.plot(epochs, cross_entropy_loss_val, 'b', label='CrossEntropyLoss Validation')
plt.plot(epochs, nll_loss_train, 'go', label='NLLLoss Training')
plt.plot(epochs, nll_loss_val, 'g', label='NLLLoss Validation')
plt.xlabel('Epochs')
plt.ylabel(ylabel)
plt.legend()

plt.savefig(f"{title}.svg")
