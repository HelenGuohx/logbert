file="../output/hdfs"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

file="../output/hdfs/bert"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi