#ifdef USE_OPENCV

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/bin_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp" 

namespace caffe {

template <typename Dtype>
BinDataLayer<Dtype>::~BinDataLayer(){}

template <typename Dtype>
void BinDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{
	  const BinDataParameter& param = this->layer_param_.bin_data_param();

	  fea_file_ = param.source(); 
	  batch_size_ = param.batch_size();

      CHECK_GT(batch_size_, 0) << "Positive batch size required";

      LoadBinFileData(fea_file_.c_str());

      vector<int> dataShape(2);
      dataShape[0] = batch_size_;
      dataShape[1] = fea_dim_;
      top[0]->Reshape(dataShape);
 

      DLOG(INFO) << "Successully loaded " << top[0]->shape(0)
				<< "Binary features";
}

template <typename Dtype>
void BinDataLayer<Dtype>::LoadBinFileData(const char* filename)
{
	  LOG(INFO) << "Opening file " << fea_file_;

          FILE* infile1;

	  if((infile1=fopen(fea_file_.c_str(),"rb+"))==NULL)
	  {
		  LOG(ERROR)<<"Can not open file.\n";
	  }

     int totalnum1 = 0;
     int featuredim = 0;

     CHECK(fread(&totalnum1,sizeof(int),1,infile1)>0)<<"Read data count failed";
     CHECK(fread(&featuredim,sizeof(int),1,infile1)>0)<<"Read data feature dim failed";
 

     CHECK_GT(totalnum1,0)<<"The size of Bin feature number should greater than 0"; 
     CHECK_GT(featuredim,0)<<"The dimensions of Bin feature should greater than 0";  

	 pos_ = 0;
	 totalcount_ = totalnum1;
	 fea_dim_ = featuredim; 
	  
	  vector<int> dataShape(2);
	  dataShape[0] = totalcount_;
	  dataShape[1] = fea_dim_;
	  data_.Reshape(dataShape);

	   
	  data_.cpu_data(); 

	  Dtype* pdata = data_.mutable_cpu_data(); 

	  size_t actual1 = fread(pdata,sizeof(Dtype),totalcount_*fea_dim_,infile1); 
	  CHECK_EQ(actual1,totalcount_*fea_dim_)<<"Can't read enough feature data"; 

	  LOG(INFO) << "A total of " << totalcount_ << " text features.";

	  fclose(infile1); 

      data_permutation_.resize(totalcount_);
	  for(int i=0;i<totalcount_;i++)
		  data_permutation_[i]=i; 
}
 
template <typename Dtype>
void BinDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK(data_.cpu_data()) << "BinDataLayer needs to be initalized by calling DataLayerSetUp"; 

	vector<int> dataShape(2);
	dataShape[0] = batch_size_;
	dataShape[1] = fea_dim_;
    top[0]->Reshape(dataShape); 

	CHECK_EQ(data_permutation_.size(),totalcount_)<<"The size of data_permutation_ is not equal with totalcount_";

	int data_dim = top[0]->count() / top[0]->shape(0); 
	if(pos_+batch_size_ <= totalcount_)
    {
		 for(int i=0;i<batch_size_;i++)
		{
			caffe_copy(data_dim, &data_.cpu_data()[data_permutation_[pos_+i]*data_dim], &top[0]->mutable_cpu_data()[i* data_dim]); 

		}
		 pos_ += batch_size_;
    }else{
    	int firstpart = totalcount_-pos_;
    	int lastpart = pos_+batch_size_ - totalcount_;
    	for(int i=0;i<firstpart;i++)
    	{
    		caffe_copy(data_dim, &data_.cpu_data()[data_permutation_[pos_+i]*data_dim], &top[0]->mutable_cpu_data()[i* data_dim]); 
    	}
        pos_ = 0;
        DLOG(INFO) << "Restarting data from start."; 
    	for(int i=0;i<lastpart;i++)
    	{
    		caffe_copy(data_dim, &data_.cpu_data()[data_permutation_[pos_+i]*data_dim], &top[0]->mutable_cpu_data()[(i+firstpart)* data_dim]); 
    	 }
    	pos_ += lastpart;
    }
}


INSTANTIATE_CLASS(BinDataLayer);
REGISTER_LAYER_CLASS(BinData);

}  // namespace caffe
#endif  // USE_OPENCV
