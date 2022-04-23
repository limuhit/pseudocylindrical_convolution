#include "sphere_slice.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void sphere_slice_opt::init(){
    init_base();
}

template <typename scalar_t>
__global__ void init_slice_param_kernel(const int nthreads, const int npart, const int width, const int * hindex, scalar_t * param){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ti = index % width;
        int tp = index / width;
        int tw = hindex[npart+tp];
        if(ti<tw){
            scalar_t nidx = (ti + 0.5) / tw * width - 0.5 + 1e-9;
            nidx = (nidx<0) ?  nidx+width : nidx;
            scalar_t nint = static_cast<scalar_t>(static_cast<int>(nidx));
            scalar_t t = nidx - nint;
            scalar_t t2 = t*t;
            scalar_t t3 = t*t2;
            param[index*5] = nint;
            param[index*5+1] = (-t+2*t2-t3)/2; 
            param[index*5+2] = (2-5*t2+3*t3)/2;
            param[index*5+3] = (t+4*t2-3*t3)/2;
            param[index*5+4] = (-t2+t3)/2;   
        }
    }
}

template <typename scalar_t>
__global__ void init_inv_param_kernel(const int nthreads, const int npart, const int width, const int stride_inv, int * inv_idx, 
    const scalar_t * param, const int * hindex){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = hindex[npart+index];
        for(int i = 0, j=0; i<tw; i++){
           int pidx = static_cast<int>(param[(index*width+i)*5]+0.1);
           for(int k = -1; k<3; k++){
                int base = (index*width + (pidx + k + width)%width)*stride_inv;
                for(j=0; j<stride_inv; j+=2){
                    if(inv_idx[base+j]==-1) break;
                }
                if(j==stride_inv) printf("stack_full\n");
                inv_idx[base+j] = i;
                inv_idx[base+j+1] = k+2;
           } 
        }
    }
}


void sphere_slice_opt::reshape(int num, int channel, int height, int width){
    bool hflag = (height_==height);
    if (!reshape_base(num, channel, height, width)) return;
	w_out_ = width_;
    n_out_ = num_ * npart_;	
    if(hflag) return;
    hindex_ = at::zeros({2, npart_}, at::kInt);
    hinv_ = at::zeros({2, height_}, at::kInt);
    h_out_ = sphere_cal_npart_hw_v2(height_, width_, npart_, weight_, hindex_.data_ptr<int>(), hinv_.data_ptr<int>());
    rt_ = 4;
    hindex_ = hindex_.to(torch::TensorOptions().device(torch::kCUDA, device_));
    hinv_ = hinv_.to(torch::TensorOptions().device(torch::kCUDA, device_));
    inv_param_ = at::zeros({npart_,width_,rt_*10}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_));
    caffe_gpu_set(stream_, npart_*width_*rt_*10, -1, inv_param_.data_ptr<int>());
    init_param_ = true;
}

void sphere_slice_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({n_out_, channel_, h_out_+2*pad_, w_out_+2*pad_});
    reshape_top_base(option,shapes);
    if(init_param_) resize_param_ = torch::zeros({npart_,width_,5},option);
}

void sphere_slice_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(option,shapes);
}


template <typename scalar_t>
__global__ void sphere_slice_forward_kernel(const int nthreads, const scalar_t* const input,  
    scalar_t * const output, const scalar_t * param, const int * hindex, const int width, 
    const int height, const int height_in, const int channel, const int npart, const int pad, 
    const int stride_h, const int stride_w) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int tw = index % width;
       int th = (index / width) % height;
       int tc = (index / width / height) % channel;
       int tn = index / width / height / channel;
       int oidx = ((tn*channel + tc)*stride_h + th + pad)*stride_w + tw + pad;
       int pn = tn / npart;
       int pt = tn % npart;
       int ph = pt>0? th + hindex[pt-1] : th;
       if(tw>=hindex[pt+npart]){
           output[oidx] = 0;
           continue;
       } 
       int base = (pt*width+tw)*5;
       int pw = static_cast<int>(param[base]);
       int pidx = ((pn*channel + tc)*height_in + ph) * width;
       
       if(pw>0 && pw < width-2){
           output[oidx] = param[base+1]*input[pidx+pw-1] + param[base+2]*input[pidx+pw] +
                           param[base+3]*input[pidx+pw+1] + param[base+4]*input[pidx+pw+2];
       }else{
           output[oidx] = param[base+1]*input[pidx+(pw-1+width)%width] + param[base+2]*input[pidx+pw] +
                           param[base+3]*input[pidx+(pw+1)%width] + param[base+4]*input[pidx+(pw+2)%width];
       }
   }
}


std::vector<at::Tensor>  sphere_slice_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "sphere_slice_forward_cuda", 
			([&] {
                count = width_ * npart_;
                if(init_param_){
                    init_slice_param_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, npart_, width_, hindex_.data_ptr<int>(), resize_param_.data_ptr<scalar_t>());
                    init_inv_param_kernel<< <CAFFE_GET_BLOCKS(npart_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (npart_, npart_, width_, rt_*10, inv_param_.data_ptr<int>(), resize_param_.data_ptr<scalar_t>(), hindex_.data_ptr<int>());
                    init_param_ = false;
                }
                count = n_out_ * channel_ * w_out_ * h_out_;
                sphere_slice_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),resize_param_.data_ptr<scalar_t>(),
                        hindex_.data_ptr<int>(), width_, h_out_, height_, channel_, npart_, pad_, h_out_+2*pad_, width_+2*pad_);
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

/*
template <typename scalar_t>
__global__ void sphere_slice_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output, const int* inv_idx, const int* hindex, const int* hinv, const scalar_t* resize_param,
     const int channel, const int height, const int width, const int h_out, const int npart, const int stride_inv, const int pad,
     const int stride_h, const int stride_w) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int th = (index / width) % height;
        int tc = (index / width / height) % channel;
        int tn = index / width / height / channel;
        int tp = hinv[th];
        int ph = hinv[th+height];
        int pn = tn*tp;
        scalar_t sum = 0;
        int base = (tp*width+tw)*stride_inv;
        for(int i = 0; i<stride_inv; i+=2){
            if(inv_idx[base+i]==-1) break;
            sum = sum + output[((pn*channel+tc)*stride_h+ph+pad)*stride_w+inv_idx[base+i]+pad]*resize_param[(tp*width+inv_idx[base+i])*5+inv_idx[base+i+1]];
        }
        input[index] = sum;
    }
}

std::vector<at::Tensor>  sphere_slice_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_slice_backward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    sphere_slice_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), inv_param_.data_ptr<int>(),
                            hindex_.data_ptr<int>(), hinv_.data_ptr<int>(), resize_param_.data_ptr<scalar_t>(), channel_, height_,
                             width_, h_out_, npart_, rt_*10, pad_, h_out_+2*pad_, width_+2*pad_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return {bottom_diff_[0], inv_param_};
}*/


template <typename scalar_t>
__global__ void sphere_slice_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output, const scalar_t * param, const int * hindex, const int width, 
    const int height, const int height_in, const int channel, const int npart, const int pad, 
    const int stride_h, const int stride_w) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int tw = index % width;
       int th = (index / width) % height;
       int tc = (index / width / height) % channel;
       int tn = index / width / height / channel;
       int oidx = ((tn*channel + tc)*stride_h + th + pad)*stride_w + tw + pad;
       int pn = tn / npart;
       int pt = tn % npart;
       int ph = pt>0? th + hindex[pt-1] : th;
       if(tw>=hindex[pt+npart]){
           continue;
       } 
       int base = (pt*width+tw)*5;
       int pw = static_cast<int>(param[base]);
       int pidx = ((pn*channel + tc)*height_in + ph) * width;
       
       if(pw>0 && pw < width-2){
            atomicAdd(input+pidx+pw-1, output[oidx]*param[base+1]);
            atomicAdd(input+pidx+pw, output[oidx]*param[base+2]);
            atomicAdd(input+pidx+pw+1, output[oidx]*param[base+3]);
            atomicAdd(input+pidx+pw+2, output[oidx]*param[base+4]);
       }else{
            atomicAdd(input+pidx+(pw-1+width)%width, output[oidx]*param[base+1]);
            atomicAdd(input+pidx+pw, output[oidx]*param[base+2]);
            atomicAdd(input+pidx+(pw+1)%width, output[oidx]*param[base+3]);
            atomicAdd(input+pidx+(pw+2)%width, output[oidx]*param[base+4]);
       }
   }
}

std::vector<at::Tensor>  sphere_slice_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_slice_backward_cuda", 
			([&] {
                    count = n_out_ * channel_ * w_out_ * h_out_;
                    caffe_gpu_set(stream_, num_*channel_*height_*width_, 0, bottom_diff_[0].data_ptr<scalar_t>());
                    sphere_slice_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), resize_param_.data_ptr<scalar_t>(),
                            hindex_.data_ptr<int>(), width_, h_out_, height_, channel_, npart_, pad_, h_out_+2*pad_, width_+2*pad_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}