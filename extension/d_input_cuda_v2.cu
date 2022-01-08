#include "d_input_v2.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "math_functions.hpp"

void d_input_opt2::init(){
    init_base();
}

void d_input_opt2::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    pidx_ = 0;
    h_out_ = height_*npart_;
    w_out_ = width_;
    mod_ = h_out_ + w_out_ + channel_ - 2;
    std::vector<at::Tensor> tmp = ctx_->produce_param_group(height, width);
    index_ = tmp[0];
    start_idx_ = tmp[1];
}

void d_input_opt2::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({rep_*num_, channel_, height_ + 2*pad_, width_+ 2*pad_});
    reshape_top_base(option,shapes);
}


template <typename scalar_t>
__global__ void d_input2_forward_kernel(const int num, const scalar_t * const input,
    const int * index, scalar_t * const output, const int start_idx, const int len_idx,
    const int height, const int width, const int channel, const int npart, const int psum, 
    const int hout, const int wout, const int pad, float bias, const int rep, const int stride_out) {
    CUDA_KERNEL_LOOP(i, num) {
        int tl = i  % len_idx;
        int tn = i / len_idx;
        int thw = index[tl + start_idx];
        int tw = thw % width;
        int tha = thw / width;
        int tg =  tha / height;
        int th = tha % height;
        int tc = psum - tw - tha;
        int pidx = (((tn*npart+tg)*channel+tc)*hout + th + pad)*wout + tw + pad;//(tn*nchannel + tc)*height*width + thw;
        scalar_t tmp = input[i] + bias;
        for(int j = 0; j< rep; j++){
            output[pidx+j*stride_out] = tmp;
        }
    }

}


std::vector<at::Tensor>  d_input_opt2::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0)*npart_, channel_, bottom_data.size(2)/npart_, bottom_data.size(3));
    reshape_top(bottom_data.options());
    const int* start_idx = start_idx_.data_ptr<int>();
    int psum = pidx_;
    pidx_ = pidx_ + 1;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "d_input2_forward_cuda", 
			([&] {
                    int stride_out = num_*channel_*(width_+2*pad_)*(height_+2*pad_);
                    if (psum == 0) {
                        caffe_gpu_set(stream_, rep_*stride_out, scalar_t(0), top_data_[0].data_ptr<scalar_t>());
                    }
                    else if(psum<=mod_){
                        psum -= 1;
                        int st = psum - channel_ + 1 < 0 ? 0 : psum - channel_ + 1;
                        int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
                        int len_idx = start_idx[end] - start_idx[st];
                        int count = len_idx*num_/npart_;
                        if(count>0) {
                            d_input2_forward_kernel << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom_data.data_ptr<scalar_t>(), index_.data_ptr<int>(), top_data_[0].data_ptr<scalar_t>(), 
                                    start_idx[st], len_idx, height_, width_,  channel_, npart_, psum, height_+2*pad_, width_+2*pad_, 
                                    pad_, bias_, rep_, stride_out);
                        }
                    }
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}