#!/bin/bash -l

datasets=("Soldeg" "SoldegH" "SoldegHNEB" "cSi")
cur="$(pwd)"
echo -n "" > label.txt
echo -n "" > data.txt
for dataset in ${datasets[@]}; do
	cd ../data/$dataset/src/
	echo $(pwd)
	python dataset.py
	cat label.txt >> $cur/label.txt
	cat data.txt >> $cur/data.txt
	cd $cur
done
