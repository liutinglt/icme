# ICME cross-media retrieval

## Introduction

This is an implementation of "Enhanced isomorphic semantic representation for cross-media retrieval" in caffe. We add a bin_data_layer for reading the binary file.  

## Usage

We conduct experiments on three publicly available datasets, i.e., Wikipedia, Pascal Sentence and Pascal VOC 2007. All the experimental data are put in the folder 'experiments'. The extracted text feature and label vector, which are saved as binary file, are put in the folder 'experiments/data'. In the binary file, the first two positions are the number and dimension of features with 'int', respectively. The rest are features with the type of 'float'.

In the folder 'experiments/script', the prototxt and scripts are provided. The implementation about the paper **"Modality-dependent cross-media retrieval"** is in 'experiments/script/finetuning0'. In 'experiments/script/finetuning1', it is the implementation of the approach proposed in "Enhanced isomorphic semantic representation for cross-media retrieval".   

1. Install caffe following the instructions of [caffe](https://github.com/BVLC/caffe). And download the pre-trained model *bvlc_reference_caffenet*.
2. Download the three datasets, i.e., [Wikipedia](http://www.svcl.ucsd.edu/projects/crossmodal/), [Pascal Sentence](http://vision.cs.uiuc.edu/pascal-sentences/),[Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
3. Change all the path in the script, and then run the *finetuning_run.sh* for training on your own dataset.

The extracted features used for retrieval are provided in the folder 'experiments/feature'. You can directly utilize those features for retrieval task.

For more details, please refer to "Modality-dependent cross-media retrieval" and "Enhanced isomorphic semantic representation for cross-media retrieval".

## Citation

If this code is helpful for your research, please cite the following paper:

@inproceedings{liu2017cross,
  author = {Ting Liu and
            Yao Zhao and
            Shikui Wei and
            Yunchao Wei and
            Lixin Liao},
  title = {Enhanced isomorphic semantic representation for cross-media retrieval},
  booktitle = {IEEE International Conference on Multimedia and Expo(ICME)},
  year = {2017}
}