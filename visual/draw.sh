#!/bin/sh

svg2png() {
  filename="$(basename "$1" .svg)"
  convert -density 1200 "$filename".svg "$filename".png
}

python figure.py Loss Loss loss.data
python figure.py Accuracy Accuracy acc.data
python bar.py bar 0.678 0.673 0.683
python comparison.py Loss_Comparison loss loss_comparison.data
python comparison.py Accuracy_Comparison Accuracy accuracy_comparison.data

for i in ./*.svg; do
  svg2png "$i"
  rm "$i"
done
