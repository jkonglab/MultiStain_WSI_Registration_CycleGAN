#!/bin/bash
# https://github.com/junyanz/CycleGAN/blob/master/datasets/download_dataset.sh

FILE=$1

if [[ $FILE != "He2Ki67" && $FILE != "He2PHH3" ]]; then
	    echo "Available datasets are: HE2Ki67, HE2PHH3"
	        exit 1
	fi

	URL=   #https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
	ZIP_FILE=./data/$FILE.zip
	TARGET_DIR=./data/$FILE/
	wget -N $URL -O $ZIP_FILE
	mkdir -p $TARGET_DIR
	unzip $ZIP_FILE -d ./data/
	rm $ZIP_FILE
