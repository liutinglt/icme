CAFFE=/home/amax/caffe-master/build/tools/caffe 

$CAFFE train --solver /home/amax/pascal_sentence/finetuning0/solver.prototxt --weights /home/amax/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --gpu=all 2>&1 |tee /home/amax/pascal_sentence/finetuning0/log/imagenet.log
 
$CAFFE train --solver /home/amax/pascal_sentence/finetuning0/solver_text.prototxt --gpu=0 2>&1 | tee  /home/amax/pascal_sentence/finetuning0/log/textnet.log 