#!/bin/bash

file="${HOME}/.dataset/tbird/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

cd $file
zipfile=tbird2.gz
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/${zipfile}
gunzip -k $zipfile