#!/bin/sh
for i in ./*.png; do
  convert "$i" -resize 50% "$i"
done
