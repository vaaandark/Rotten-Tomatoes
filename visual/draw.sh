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

python bar.py 训练损失   0.490 0.420 0.391
python bar.py 训练准确率 0.605 0.656 0.683
python bar.py 测试损失   0.439 0.415 0.405
python bar.py 测试准确率 0.638 0.659 0.666

for i in ./*.svg; do
  svg2png "$i"
  rm "$i"
done
