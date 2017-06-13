#ifndef CAFFE_BIN_DATA_LAYER_HPP_
#define CAFFE_BIN_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BinDataLayer:public Layer<Dtype>{
	public:
	explicit BinDataLayer(const LayerParameter& param):Layer<Dtype>(param){};
	 virtual ~BinDataLayer();
	 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top);
	 virtual inline const char* type() const{ return "BinData";}
	 virtual inline int ExactNumBottomBlobs() const { return 0; }
	 virtual inline int ExactNumTopBlobs() const { return 1; }
	 virtual inline bool ShareInParallel() const { return true; }
	 // Data layers have no bottoms, so reshaping is trivial.
	 virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		   const vector<Blob<Dtype>*>& top) {}
	 int fea_dim() { return fea_dim_; }

	protected:
	   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	       const vector<Blob<Dtype>*>& top);
	   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	   virtual void LoadBinFileData(const char* filename);
	  int fea_dim_,batch_size_,totalcount_;//Bin特征维度与批处理个数
	  size_t pos_;             //当前读取位置
	  Blob<Dtype> data_; 
	  std::string fea_file_; 
	  size_t current_row_;

	  std::vector<unsigned int> data_permutation_;        //数据排列
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
