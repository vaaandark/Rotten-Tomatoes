#!/bin/sh

# Epochs: 1 | Train Loss:  0.490 | Train Accuracy:  0.605 | Test Loss:  0.439 | Test Accuracy:  0.638
# Epochs: 2 | Train Loss:  0.420 | Train Accuracy:  0.656 | Test Loss:  0.415 | Test Accuracy:  0.659
# Epochs: 3 | Train Loss:  0.391 | Train Accuracy:  0.683 | Test Loss:  0.405 | Test Accuracy:  0.666
# 
# Average accuracy of 3 times:  0.654

svg2png() {
  filename="$(basename "$1" .svg)"
  convert -density 1200 "$filename".svg "$filename".png
}

python figure.py Loss Loss loss.data
python figure.py Accuracy Accuracy acc.data

for i in ./*.svg; do
  svg2png "$i"
  rm "$i"
done
