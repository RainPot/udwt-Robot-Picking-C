cd ext/nms/
make
cd ../../data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/2018origin.zip
unzip -q 2018origin.zip
cd ..
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth
mv scripts/RRNet/train.py ./
mv scripts/RRNet/eval.py ./

