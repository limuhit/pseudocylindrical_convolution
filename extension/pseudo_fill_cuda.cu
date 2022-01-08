#include "pseudo_fill.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void pseudo_fill_opt::init(){
    init_base();
}

void pseudo_fill_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    switch(context_version_){
        case 0:
            hindex_ = pctx_->produce_param_fill(height,width);
            break;
        case 1:
            hindex_ = pectx_->produce_param_fill(height,width);
            break;
        default:
            hindex_ = ectx_->produce_param_fill(height,width); 
            break;
    }
    //printf("hindex addr: %p\n", hindex_.data_ptr<int>());
}

template <typename scalar_t>
__global__ void pseudo_fill_forward_kernel(const int nthreads,   scalar_t * const data, 
    const int * hindex, const int width, const int height, const int channel, const int npart, 
    const int pad, const int trim, scalar_t fvalue) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        int pg = (index / width / height / channel) % npart;
        if(ph<pad-trim || ph>=height-pad+trim){
            data[index] = fvalue;
        }else{
            if(pw<pad-trim || pw>=pad+hindex[pg]+trim){
                data[index] = fvalue;
            }
        }
    }
}


std::vector<at::Tensor>  pseudo_fill_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "pseudo_fill_forward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    pseudo_fill_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), hindex_.data_ptr<int>(), width_, height_, channel_, npart_, pad_, trim_, static_cast<scalar_t>(fvalue_));
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return {bottom_data};
}


std::vector<at::Tensor>  pseudo_fill_opt::backward_cuda(at::Tensor  top_diff) 
{
    int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "pseudo_fill_backward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    pseudo_fill_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, top_diff.data_ptr<scalar_t>(), hindex_.data_ptr<int>(), width_, height_, channel_, npart_, pad_, trim_, static_cast<scalar_t>(0.));
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return {top_diff};
}