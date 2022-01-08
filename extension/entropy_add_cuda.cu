#include "entropy_add.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void entropy_add_opt::init(){
    init_base();
}

void entropy_add_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    num_out_ = num / npart_;
    h_out_ = height_*npart_;
    w_out_ = width_;
    pidx_ = 0;
    mod_ = h_out_ + w_out_ + ngroup_ - 2;
    std::vector<at::Tensor> tmp = ctx_->produce_param_group(height, width);
    index_mat_ = tmp[0];
    plan_idx_ = tmp[1];
}


template <typename scalar_t>
__global__ void entropy_add_forward_kernel(const int count, scalar_t * output, const scalar_t * input, 
    const int * mindex, const int group_out, const int start_idx, const int psum, 
	const int height, const int width, const int nout, const int num, const int npart, const int pad, const int inner_shape) {
	CUDA_KERNEL_LOOP(index, count) {
		int pn = index % num;
        int pp = index / num;
        int pb = pp % inner_shape;
        int hw = mindex[pb + start_idx];
        int tw = hw % width;
        int hp = hw / width;
        int tg = hp / height;
        int th = hp % height;
		int tc =  psum - tw - hp;
		int og = pp / inner_shape;
		int pout = (tc * group_out + og);
        int qn = pn*npart + tg;
		int out_idx = ((qn*nout+pout)*(height+2*pad)+th+pad)*(width+2*pad) + tw + pad;
		output[out_idx] = output[out_idx] + input[out_idx];
	}
}


std::vector<at::Tensor>  entropy_add_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor bottom_data2) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2)-2*pad_, bottom_data.size(3)-2*pad_);
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
    if(psum<=mod_){
        const int* start_idx = plan_idx_.data_ptr<int>();
        int st = psum - ngroup_ + 1 < 0 ? 0 : psum - ngroup_ + 1;
        int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
        int len_idx = start_idx[end] - start_idx[st];
        AT_DISPATCH_FLOATING_TYPES(
            bottom_data.scalar_type(), "entropy_add_forward_cuda", 
                ([&] {
                        timer_->start();
                        count = cpg_ * len_idx * num_out_;
                        if(count>0){
                            entropy_add_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom_data.data_ptr<scalar_t>(), bottom_data2.data_ptr<scalar_t>(),  index_mat_.data_ptr<int>(), 
                                    cpg_, start_idx[st], psum, height_, width_, channel_, num_out_, npart_, pad_, len_idx);
                             CUDA_POST_KERNEL_CHECK;
                        }
                        timer_->stop("kernel 1");
                    }
                )
        );
    }
    return {bottom_data};
}