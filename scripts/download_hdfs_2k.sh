#!/bin/bash

file="${HOME}/.dataset/hdfs_2k/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

wget https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log -P $file