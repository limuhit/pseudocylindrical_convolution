#include "d_extract_v2.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void d_extract_opt2::init(){
    init_base();
}

void d_extract_opt2::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    h_out_ = height_*npart_;
    w_out_ = width_;
    cpn_ = channel_ / nchannel_;
    pidx_ = 0;
    mod_ = h_out_ + w_out_ + nchannel_ - 2;
    top_num_ = at::zeros({1},at::kInt);
    std::vector<at::Tensor> tmp = ctx_->produce_param_group(height, width);
    index_ = tmp[0];
    start_idx_ = tmp[1];
    //printf("%d,%d,%d,%d,%d,%d,%d,%d\n",num_,channel_,height_,width_,h_out_,w_out_,cpn_,mod_);
}

void d_extract_opt2::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_/npart_, cpn_, h_out_, w_out_});
    reshape_top_base(option,shapes);
}


template <typename scalar_t>
__global__ void d_extract2_forward_kernel(const int num, const scalar_t * const input,
    const int * index, scalar_t * const output, const int start_idx, const int len_idx,
    const int height, const int width, const int channel,  const int cpn, const int npart, const int psum) {
    CUDA_KERNEL_LOOP(i, num) {
        int ci = i % cpn;
        int tl = (i / cpn)  % len_idx;
        int tn = i / cpn / len_idx;
        int thw = index[tl + start_idx];
        int tw = thw % width;
        int tha = thw / width;
        int tg = tha / height;
        int th = tha % height;
        int tc = psum - tw - tha;
        int pidx = (((tn*npart + tg)*channel + tc*cpn + ci)*height + th) * width + tw;
        //printf("%d,%d,%d,%d,%d,%d\n",tn,tg,tc,ci,th,tw);
        output[i] = input[pidx];
    }

}

std::vector<at::Tensor>  d_extract_opt2::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
    int * tn = top_num_.data_ptr<int>();
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
    const int* start_idx = start_idx_.data_ptr<int>();
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "d_extract2_forward_cuda", 
			([&] {
                
                const scalar_t * bottom = bottom_data.data_ptr<scalar_t>();
                scalar_t * top_data = top_data_[0].data_ptr<scalar_t>();
                if (label_) {
                    if(psum<mod_){
                        int st = psum - nchannel_ + 1 < 0 ? 0 : psum - nchannel_ + 1;
                        int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
                        int len_idx = start_idx[end] - start_idx[st];
                        int count = len_idx * num_ / npart_ * cpn_;
                        tn[0] = count / cpn_;
                        if(count>0){
                            d_extract2_forward_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom, index_.data_ptr<int>(), top_data, start_idx[st], len_idx, height_, width_, channel_, cpn_, npart_, psum);
                        }
                    }
                }
                else {
                    
                    if (psum == 0) {
                        //printf("%d,%d,%d,%d,%d\n",mod_, psum, num_, h_out_, num_ / npart_ *  cpn_ * h_out_ * w_out_);
                        caffe_gpu_set(stream_, num_ / npart_ *  cpn_ * h_out_ * w_out_, scalar_t(0), top_data);
                    }
                    else if(psum<=mod_){
                        //printf("%d,%d,%d\n",num_,cpn_,npart_);
                        psum -= 1;
                        int st = psum - nchannel_ + 1 < 0 ? 0 : psum - nchannel_ + 1;
                        int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
                        int len_idx = start_idx[end] - start_idx[st];
                        int count = len_idx * num_ / npart_ * cpn_;
                        tn[0] = count / cpn_;
                        if(count>0){
                            d_extract2_forward_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom, index_.data_ptr<int>(), top_data, start_idx[st], len_idx, height_, width_, channel_, cpn_, npart_, psum);
                        }
                    }
                }
                CUDA_POST_KERNEL_CHECK;
   			})
    );
    return {top_data_[0],top_num_};
}


template <typename scalar_t>
__global__ void d_extract2_batch_forward_kernel(const int num, const scalar_t * const input,
    const int * index, scalar_t * const output, const int start_idx, const int len_idx,
    const int height, const int width, const int channel,  const int cpn, const int npart, 
    const int psum, const int stride, const int inner_shape) {
    CUDA_KERNEL_LOOP(i, num) {
        int ps = i % inner_shape;
        int pn = i / inner_shape;
        int ci = i % cpn;
        int tl = (i / cpn)  % len_idx;
        int tn = i / cpn / len_idx;
        int thw = index[tl + start_idx];
        int tw = thw % width;
        int tha = thw / width;
        int tg = tha / height;
        int th = tha % height;
        int tc = psum - tw - tha;
        int pidx = (((tn*npart + tg)*channel + tc*cpn + ci)*height + th) * width + tw;
        //printf("%d,%d,%d,%d,%d,%d\n",tn,tg,tc,ci,th,tw);
        int qidx = pn*stride + ps;
        output[qidx] = input[pidx];
    }

}

std::vector<at::Tensor>  d_extract_opt2::forward_batch_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
    int * tn = top_num_.data_ptr<int>();
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
    int nout = num_ / npart_ / 3;
    const int* start_idx = start_idx_.data_ptr<int>();
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "d_extract2_forward_cuda", 
			([&] {
                
                const scalar_t * bottom = bottom_data.data_ptr<scalar_t>();
                scalar_t * top_data = top_data_[0].data_ptr<scalar_t>();
                if(psum<mod_){
                    int st = psum - nchannel_ + 1 < 0 ? 0 : psum - nchannel_ + 1;
                    int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
                    int len_idx = start_idx[end] - start_idx[st];
                    //printf("%d %d %d %d\n",psum,len_idx, end, st);
                    int count = len_idx * num_ / npart_ * cpn_;
                    tn[0] = nout*len_idx;
                    if(count>0){
                        d_extract2_batch_forward_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom, index_.data_ptr<int>(), top_data, start_idx[st], len_idx, height_, width_, channel_, cpn_, npart_,
                                psum, cpn_ * h_out_ * w_out_*nout, len_idx*cpn_*nout);
                    }
                }
                CUDA_POST_KERNEL_CHECK;
   			})
    );
    return {top_data_[0],top_num_};
}