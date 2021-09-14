#!/bin/bash

file="${HOME}/.dataset/hdfs/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

cd $file
zipfile=HDFS_1.tar.gz?download=1
wget https://zenodo.org/record/3227177/files/${zipfile} -P $file
tar -xvzf $zipfile