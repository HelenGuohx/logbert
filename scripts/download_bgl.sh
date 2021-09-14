#!/bin/bash

file="${HOME}/.dataset/bgl/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

cd $file
zipfile=BGL.tar.gz?download=1
wget https://zenodo.org/record/3227177/files/${zipfile} -P $file
tar -xvzf $zipfile