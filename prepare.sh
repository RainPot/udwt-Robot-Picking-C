cd ext/nms/
make
cd ../../data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/2019origin.zip
unzip -q 2019origin.zip
cd ..
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth
mv ./backbones/hourglass.py ./backbones/hourglass_old.py
mv ./backbones/hourglass_flip.py ./backbones/hourglass.py
#hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/hourglass_UNDER_39000.pth
#mv ./hourglass_UNDER_39000.pth ./hourglass.pth
mv scripts/RRNet/train.py ./
mv scripts/RRNet/eval.py ./

mkdir log
mkdir results
mkdir deepsort_holothurian
mkdir deepsort_scallop
mkdir deepsort_echinus
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/original_ckpt.t7
cd log
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/UNDERantialiased/ckp-64999.pth

