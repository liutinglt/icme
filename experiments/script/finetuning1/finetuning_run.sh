CAFFE=/home/amax/caffe-master/build/tools/caffe 
$CAFFE train --solver /home/amax/caffe-master/models/crossmedia/finetuning1/solver.prototxt --weights /home/amax/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --gpu=all 2>&1 |tee /home/amax/caffe-master/models/crossmedia/finetuning1/log/imagenet_512.log 

