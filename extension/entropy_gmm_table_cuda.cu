#include "entropy_gmm_table.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void entropy_gmm_table_opt::init(){
    init_base();
}

void entropy_gmm_table_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    assert((num_gaussian_<=16)&&"the number of Gaussian Distribution in GMM should be less than 16");
}

void entropy_gmm_table_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*height_*width_, nstep_+1});
    reshape_top_base(option,shapes);
}

void entropy_gmm_table_opt::reshape_top_batch(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*height_*width_/3, nstep_+1});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void entropy_gmm_table_weight_kernel(const int nthreads, scalar_t* const weight, const int addr_start, const int w) {
    scalar_t tmp[16];
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pbase = addr_start + index*w;
        scalar_t mval = -1e10, psum = 0;
        for(int i = 0; i<w; i++){
            tmp[i] = weight[pbase+i];
            if(mval<tmp[i])
                mval = tmp[i];
        }
        for(int i = 0; i<w; i++){
            tmp[i] = exp(tmp[i] - mval);
            psum += tmp[i];
        }
        for(int i = 0; i<w; i++){
            weight[pbase+i] = tmp[i] / psum;
        }
    }
}

template <typename scalar_t>
__global__ void entropy_gmm_table_delta_kernel(const int nthreads, scalar_t* const delta, const int addr_start, scalar_t beta) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        scalar_t tmp = delta[addr_start+index];
        tmp = tmp < 0 ? beta: tmp + beta;
        delta[addr_start+index] = tmp;
    }
}

template <typename scalar_t>
__global__ void entropy_gmm_table_forward_kernel(const int nthreads, const scalar_t* const weight, 
    const scalar_t* const delta,  const scalar_t* const mean, scalar_t * const output, 
    const int w, const int ntable, const scalar_t total, const scalar_t bias, const scalar_t s2) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pt = index % ntable;
        int pn = index / ntable;
        if(pt==0){
            output[index] = 0;
        }else if(pt==ntable-1){
            output[index] = static_cast<int>(total);
        }else{
            scalar_t v = pt - 1 - bias + 0.5, ps=0, f;
            for(int i = 0; i<w;i++){
                f = 0.5+0.5*erf(s2*(v-mean[pn*w+i])/delta[pn*w+i]);
                ps = ps + weight[pn*w+i]*f;
                //printf("%d %d %f %f %f\n",index,i,v,f,weight[pn*w+i]);
            } 
            output[index] = static_cast<int>(total*ps+0.5);
        }
    }
}


template <typename scalar_t>
__global__ void entropy_gmm_table_check_kernel(const int count, scalar_t * const output, const int ngroup) {
	CUDA_KERNEL_LOOP(index, count) {
		scalar_t bias = 0;
		scalar_t mval = 0;
		int midx = 0;
		for (int i = 0; i < ngroup; i++) {
			if (output[index*(ngroup+1) + i +1] <= output[index*(ngroup+1) + i])
			{
				bias += 1;
			}
            output[index*(ngroup+1) + i +1] += bias;
			if (output[index*(ngroup+1) + i+1] - output[index*(ngroup+1) + i] > mval) {
					mval = output[index*(ngroup+1) + i + 1] - output[index*(ngroup+1) + i];
					midx = i;
			}
		}
		if (bias > 0) {
			for (int i = midx; i < ngroup; i++) {
				output[index*(ngroup+1) + i+1] -= bias;
			}
		}	
	}
}

std::vector<at::Tensor>  entropy_gmm_table_opt::forward_cuda(at::Tensor weight, at::Tensor delta, at::Tensor mean, at::Tensor tnum) 
{
    reshape(weight.size(0), weight.size(1), weight.size(2), weight.size(3));
    reshape_top(weight.options());
    int tn = tnum.data_ptr<int>()[0];
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		weight.scalar_type(), "entropy_gmm_table_forward_cuda", 
			([&] {
                    entropy_gmm_table_weight_kernel<< <CAFFE_GET_BLOCKS(tn), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (tn, weight.data_ptr<scalar_t>(), 0, num_gaussian_);
                    count = tn*num_gaussian_;
                    entropy_gmm_table_delta_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, delta.data_ptr<scalar_t>(), 0, scalar_t(beta_));
                    count = tn*(nstep_+1);
                    scalar_t s2 = 1. / sqrt(2.0);
                    entropy_gmm_table_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, weight.data_ptr<scalar_t>(),  delta.data_ptr<scalar_t>(),  mean.data_ptr<scalar_t>(), 
                            top_data_[0].data_ptr<scalar_t>(),  num_gaussian_, nstep_+1, scalar_t(total_region_), scalar_t(bias_), s2);
                    entropy_gmm_table_check_kernel<< <CAFFE_GET_BLOCKS(tn), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (tn, top_data_[0].data_ptr<scalar_t>(), nstep_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void entropy_gmm_table_batch_forward_kernel(const int nthreads, const scalar_t* const data, scalar_t * const output, 
    const int w, const int ntable, const scalar_t total, const scalar_t bias, const scalar_t s2, const int stride) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pt = index % ntable;
        int pn = index / ntable;
        if(pt==0){
            output[index] = 0;
        }else if(pt==ntable-1){
            output[index] = static_cast<int>(total);
        }else{
            scalar_t v = pt - 1 - bias + 0.5, ps=0;
            for(int i = 0; i<w;i++){
                ps = ps + data[pn*w+i]*(0.5+0.5*erf(s2*(v-data[2*stride+pn*w+i])/data[stride+pn*w+i]));
            } 
            output[index] = static_cast<int>(total*ps+0.5);
        }
    }
}

std::vector<at::Tensor>  entropy_gmm_table_opt::forward_batch_cuda(at::Tensor data, at::Tensor tnum) 
{
    reshape(data.size(0), data.size(1), data.size(2), data.size(3));
    reshape_top_batch(data.options());
    int tn = tnum.data_ptr<int>()[0];
	int count;
    int stride = num_ * channel_ * width_ * height_ / 3; 
    if(tn>0){
        AT_DISPATCH_FLOATING_TYPES(
            data.scalar_type(), "entropy_gmm_table_forward_cuda", 
                ([&] {
                        entropy_gmm_table_weight_kernel<< <CAFFE_GET_BLOCKS(tn), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (tn, data.data_ptr<scalar_t>(), 0, num_gaussian_);
                        count = tn*num_gaussian_;
                        entropy_gmm_table_delta_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, data.data_ptr<scalar_t>(), stride, scalar_t(beta_));
                        count = tn*(nstep_+1);
                        scalar_t s2 = 1. / sqrt(2.0);
                        entropy_gmm_table_batch_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, data.data_ptr<scalar_t>(),  top_data_[0].data_ptr<scalar_t>(),  
                                num_gaussian_, nstep_+1, scalar_t(total_region_), scalar_t(bias_), s2, stride);
                        entropy_gmm_table_check_kernel<< <CAFFE_GET_BLOCKS(tn), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (tn, top_data_[0].data_ptr<scalar_t>(), nstep_);
                        CUDA_POST_KERNEL_CHECK;
                       }
                )
        );
    }
	
    return top_data_;
}
