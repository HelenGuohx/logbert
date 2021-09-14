#!/bin/bash

file="${HOME}/.dataset/bgl_2k/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

wget https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log -P $file