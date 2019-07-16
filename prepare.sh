cd ext/nms/
make
cd ../../data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/UNDERWATER_split.zip
unzip UNDERWATER_split.zip
cd ..
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth
mv scripts/RRNet/train.py ./
