#include "pseudo_entropy_pad.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void pseudo_entropy_pad_opt::init(){
    init_base();
}

void pseudo_entropy_pad_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    h_out_ = height_ + 2*pad_;
    w_out_ = width_ + 2*pad_;
    std::vector<at::Tensor> tmp = ctx_->produce_param(channel_, height_, width_, pad_);
    param_ = tmp[0];
    inv_param_ = tmp[1];
    hindex_  = tmp[2];
    hindex2_ = tmp[3];
    stride_inv_ = tmp[4];
    init_param_ = true;
}

void pseudo_entropy_pad_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_, h_out_, w_out_});
    reshape_top_base(option,shapes);
}

void pseudo_entropy_pad_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(option,shapes);
}


template <typename scalar_t>
__global__ void pseudo_entropy_pad_copy_forward_kernel(const int nthreads, const scalar_t* const input, 
    scalar_t* const output, const int * hindex, const int h_out, const int w_out, const int height, 
    const int width, const int channel, const int npart, const int pad){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % w_out;
        int ph = (index / w_out) % h_out;
        int ps = index / w_out / h_out;
        int pg = (ps / channel) % npart;
        if(pw<pad || pw>=hindex[pg]+pad || ph<pad || ph>=height+pad){
            output[index] = 0;
            continue;
        }
        int tidx = (ps*height+ph-pad)*width+pw-pad;
        output[index] = input[tidx];
    }
}

template <typename scalar_t>
__global__ void pseudo_entropy_pad_forward_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const scalar_t * param, const int * hindex, const int * hindex2, 
     const int inner_shape, const int astride, const int astride_out, const int bstride, const int bstride_out,
      const int width, const int channel, const int npart, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ps = index % inner_shape;
        int pc = (index / inner_shape) % channel;
        int pn = index / inner_shape / channel;
        int tn = pn / npart;
        int tg = pn % npart;
        if (pw>=hindex[tg]) continue;
        int base = (tg * inner_shape + ps)*4;
        int qg = hindex2[base/4/width];
        int pbase = static_cast<int>(param[base]+1e-6) + tn*astride_out + pc*astride;
        if(qg==-1){
            output[pbase + pw + pad] = 0;
            continue;
        }
        int qbase = static_cast<int>(param[base+1]+1e-6)+ tn*bstride_out + pc*bstride;
        int qw = param[base+2]>=0 ? static_cast<int>(param[base+2]+1e-6) : -1;
        scalar_t qdata = (qw == -1) ?  0 : input[qbase + qw];
        scalar_t t = param[base+3];
        int qww = (qw + 1) % hindex[qg];
        //printf("%d %d %d %d %d %d\n", index,  base, pbase, qbase, qw, qg);
        output[pbase + pw + pad] =  qdata * t + input[qbase + qww ]*(1-t);
    }
}

template <typename scalar_t>
__global__ void pseudo_entropy_pad_circle_forward_kernel(const int nthreads, scalar_t * const output, const int h_out, 
    const int w_out, const int * hindex, const int channel, const int pad, const int pad2, const int npart) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % pad2;
        int pn = index / pad2 / h_out / channel;
        int pg = pn % npart;
        int pwa = pw % pad;
        int pwb = pw / pad;
        int wl = hindex[pg];
        int qw = pwb*(wl + pad) + pwa;
        int base = index / pad2 * w_out;
        if(pwb<1){
            output[base+qw] = 0;
        }else{
            output[base+qw] = output[base+(qw-pad+wl)%wl+pad];
        }
        
    }
}

std::vector<at::Tensor>  pseudo_entropy_pad_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "pseudo_entropy_pad_forward_cuda", 
			([&] {
                count = num_*channel_*h_out_*w_out_;
                //printf("here! %d %d %d %d %d %d\n",  height_, width_, channel_, npart_, pad_, count );
                pseudo_entropy_pad_copy_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_data.data_ptr<scalar_t>(),top_data_[0].data_ptr<scalar_t>(), hindex_.data_ptr<int>(), 
                        h_out_, w_out_, height_, width_, channel_, npart_, pad_);
                count = num_ * channel_ * width_ * pad_ * 2;
                pseudo_entropy_pad_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),param_.data_ptr<scalar_t>(), 
                    hindex_.data_ptr<int>(), hindex2_.data_ptr<int>(), 2*pad_*width_, h_out_*w_out_, h_out_*w_out_*channel_*npart_, 
                    height_*width_, height_*width_*channel_*npart_, width_, channel_, npart_, pad_);
                count = num_* channel_ * h_out_ * pad_ * 2;
                pseudo_entropy_pad_circle_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, top_data_[0].data_ptr<scalar_t>(), h_out_, w_out_, hindex_.data_ptr<int>(), channel_, pad_, pad_*2, npart_);
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void pseudo_entropy_pad_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output, const int * hindex, const scalar_t * inv_param, const int * stride_inv, 
    const int stride, const int width, const int height, const int channel,
    const int inner_shape, const int astride_out, const int astride_inner, 
    const int npart, const int pad){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index % inner_shape) / width;
        int pc = (index / inner_shape) % channel;
        int pn = index / inner_shape / channel;
        int pg = ph / height;
        int ps = (pn*npart+pg)*channel+pc;
        int pph = ph%height;
        int pbase =  (ps*height+pph)*width+pw;
        int ppbase = (ps*(height+2*pad)+pph+pad)*(width+2*pad)+pw+pad;
        if(pw>=hindex[pg]){
            input[pbase] = 0;
            continue;
        }
        input[pbase] = output[ppbase]; 
        int base = ph*stride + pw*stride_inv[pg];
        int nn = static_cast<int>(inv_param[base]+1e-6);
        for(int i=0, sbase; i<nn; i++){
            sbase = static_cast<int>(inv_param[base+i*2+2]+1e-6) + pn*astride_out + pc*astride_inner;
            input[pbase] += output[sbase]*inv_param[base+i*2+3];
        }
    }
}


template <typename scalar_t>
__global__ void pseudo_entropy_pad_circle_backward_kernel(const int nthreads, scalar_t * const output, const int h_out, 
    const int w_out, const int * hindex, const int channel, const int pad, const int pad2, const int npart) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % pad2;
        int pn = index / pad2 / h_out / channel;
        int pg = pn % npart;
        int pwa = pw % pad;
        int pwb = pw / pad;
        int wl = hindex[pg];
        int qw = pwb*(wl + pad) + pwa;
        int base = index / pad2 * w_out;
        if(pwb>0){
            output[base+(qw-pad+wl)%wl+pad] += output[base+qw];
            output[base+qw] = 0.;
        }
    }
}

template <typename scalar_t>
__global__ void pseudo_entropy_pad_circle_backward_lfour_kernel(const int nthreads, scalar_t * const output, const int h_out, 
    const int w_out, const int * hindex, const int channel, const int pad,  const int npart) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pn = index / h_out / channel;
        int pg = pn % npart;
        int wl, qw, base;
        for(int pwa=0; pwa<pad; pwa++){
            wl = hindex[pg];
            qw = wl + pad + pwa;
            base = index * w_out;
            output[base+(qw-pad)%wl+pad] += output[base+qw];
            output[base+qw] = 0.;
        }
    }
}

std::vector<at::Tensor>  pseudo_entropy_pad_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
    if(init_param_){
        at::Tensor x = hindex_.to(torch::TensorOptions().device(torch::kCPU));
        int * xi = x.data_ptr<int>();
        lfour_ = false;
        for(int i=0; i<npart_; i++)
        {
            if(xi[i]<2*pad_)
                lfour_ = true;
        }
        init_param_ = false;
    }
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "ppseudo_entropy_pad_backward_cuda", 
			([&] {
                if(lfour_){
                    //printf("lfour\n");
                    count = num_ * channel_ * h_out_;
                    pseudo_entropy_pad_circle_backward_lfour_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, top_diff.data_ptr<scalar_t>(), h_out_, w_out_, hindex_.data_ptr<int>(), channel_, pad_,  npart_);
                }else{
                    count = num_ * channel_ * h_out_ * pad_ * 2;
                    pseudo_entropy_pad_circle_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, top_diff.data_ptr<scalar_t>(), h_out_, w_out_, hindex_.data_ptr<int>(), channel_, pad_, pad_*2, npart_);
                }
                count = num_ * channel_ * width_ * height_;
                int stride = inv_param_.size(2);
                pseudo_entropy_pad_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), hindex_.data_ptr<int>(), 
                    inv_param_.data_ptr<scalar_t>(), stride_inv_.data_ptr<int>(), stride, width_, height_, channel_, 
                    npart_*height_*width_, npart_*channel_*h_out_*w_out_, h_out_*w_out_, npart_, pad_);
                
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}