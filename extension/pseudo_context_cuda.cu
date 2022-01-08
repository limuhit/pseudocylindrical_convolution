#include "pseudo_context.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void pseudo_context_opt::init(){
    init_base();
}

bool pseudo_context_opt::reshape_hw(int height, int width){
    height_ = height;
    width_ = width;
    //printf("pcontext width %d\n", width_);
    if(width_dict_.count(width)>0) return false;
    //printf("pcontext new width %d\n", width_);
    width_dict_[width] = 1;
    h_out_ = height_ * npart_;
    w_out_ = width_;
    at::Tensor hindex_tmp = at::zeros({npart_}, at::kInt);
    at::Tensor stride_inv_tmp = at::zeros({npart_}, at::kInt);
    int * sw = hindex_tmp.data_ptr<int>();
    sphere_cal_npart_hw_v3(h_out_, w_out_, npart_, weight_, sw);
    int * sidx = stride_inv_tmp.data_ptr<int>();
    for(int i=0;i<npart_;i++){
        sidx[i] = rt_ * width_ / sw[i];
    }
    stride_inv_[width_] = stride_inv_tmp.to(torch::Device(torch::kCUDA, device_));
    hindex_[width_] = hindex_tmp.to(torch::Device(torch::kCUDA, device_));
    return true;
}

bool pseudo_context_opt::reshape_channel_pad(int channel, int pad){
    channel_ = channel; 
    pad_ = pad;
    //printf("pcontext channel %d pad %d\n", channel_,pad_);
    assert((channel<1000)&&(pad<10)&&"the channel number should be less than 1000 and the pad size should be less than 10");
    cp_ = width_*10000 + channel_*10 + pad_;
    if(pad_channel_dict_.count(cp_)>0) return false;
    //printf("pcontext new channel %d pad %d\n", channel_,pad_);
    pad_channel_dict_[cp_] = 1;
    
    hindex2_[cp_] = at::zeros({npart_,2*pad_}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_));
    param_[cp_] = at::zeros({npart_,2*pad_, width_, 4}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, device_));
    param2_[cp_] = at::zeros({npart_, height_, width_*rt_}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, device_));
    return true;
}


__global__ void pseudo_context_forward_kernel(const int nthreads, float * const param, const int * hindex, int * hindex2,
    const int channel, const int height, const int width, const int npart, const int pad, const int stride) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int tg = index / width / pad / 2;
        if(tw>=hindex[tg]) continue;
        int ph, pg;
        float pw;
        if(tl==0){
            ph = tg*height - pad + tp;
            if(ph<0){
                ph = -ph - 1;
                float nw;
                nw = tw +  hindex[tg] / 2.;
                nw = (nw >= hindex[tg]) ? nw - hindex[tg] : nw; 
                pg = ph / height;
                pw = (nw + 0.5) / hindex[tg] * hindex[pg] - 0.5 + 1e-9;
            }else{
                pg = ph / height;
                pw = (tw + 0.5) / hindex[tg] * hindex[pg] - 0.5 + 1e-9;
            }
            

        }else{
            ph = (tg+1)*height + tp;
            if(ph>=height*npart){
                ph = 2*height*npart - ph - 1;
                float nw;
                nw = tw + hindex[tg] / 2.;
                nw = (nw >= hindex[tg]) ? nw - hindex[tg] : nw; 
                pg = ph / height;
                pw = (nw + 0.5) / hindex[tg] * hindex[pg] - 0.5 + 1e-9;

            }else{
                pg = ph / height;
                pw = (tw + 0.5) / hindex[tg] * hindex[pg] - 0.5 + 1e-9;
            }
            
        }
        pw = (pw<0) ?  pw + hindex[pg] : pw;

        int pidx = static_cast<int> (pw);
        param[index*stride + 2] = pidx;
        param[index*stride + 3] = pidx+1-pw;
        param[index*stride] =  (tl == 0) ? tg *channel*(height+pad*2) + tp : tg *channel*(height+pad*2) + pad + height + tp;
        param[index*stride] *= (width+pad*2);
        param[index*stride+1] = (pg*channel*height + ph % height) * width ; 
        if(tw==0){
            hindex2[(tg*2 + tl)*pad + tp] = pg;
        }
    }
}

__global__ void pseudo_context_backward_kernel(const int nthreads, float * const inv_param, float * const param, const int * hindex,
    const int channel, const int height, const int width, const int npart, const int pad, const int stride, const int stride_out,
    const int * stride_inv) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int tg = index / width / pad / 2;
        if(tw>=hindex[tg]) continue;
        int pbase = static_cast<int>(param[index*stride+1]+1e-6);
        int pw = static_cast<int>(param[index*stride+2]+1e-6);
        int sbase = static_cast<int>(param[index*stride]+1e-6);
        float t = param[index*stride+3];
        int qh = pbase / width % height;
        int qg = pbase / width / height / channel;
        int qbase = (qg*height+qh)*stride_out;
        int idx = qbase + pw*stride_inv[qg];
        //printf("%d %d %d %d\n", qg, qh, qbase, idx);
        int ti = atomicAdd(inv_param + idx,  1.); 
        if(ti*2+4>stride_inv[qg]) printf("%d %d overflow!\n", tg, ti);
        if(ti==0)  inv_param[idx+1] = pbase + pw ;
        inv_param[idx+ti*2+2] = sbase + tw + pad;
        inv_param[idx+ti*2+3] = t;
        
        int pww = (pw+1) % hindex[qg];
        idx = qbase + pww*stride_inv[qg];
        ti = atomicAdd(inv_param + idx,  1.); 
        if(ti*2+4>stride_inv[qg]) printf("%d %d overflow!\n", tg, ti);
        if(ti==0) inv_param[idx+1] = pbase + pww;
        inv_param[idx+ti*2+2] = sbase + tw + pad;
        inv_param[idx+ti*2+3] = 1-t;
    }
}

std::vector<at::Tensor> pseudo_context_opt::produce_param(int channel, int height, int width, int pad)
{
    reshape_hw(height,width);
    bool change_flag = reshape_channel_pad(channel,pad);
	int count;
	if(change_flag){
        //printf("produce pcontext %d %d %d %d\n",channel_,height_,width_,pad_);
        //printf("produce pctx %d\n",cp_);
        count = npart_  * width_ * pad_ * 2;
        caffe_gpu_set(stream_,count*4,float(0),param_[cp_].data_ptr<float>());
        caffe_gpu_set(stream_,npart_*height_*width_*rt_,float(0),param2_[cp_].data_ptr<float>());
        pseudo_context_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
            (count, param_[cp_].data_ptr<float>(), hindex_[width_].data_ptr<int>(), hindex2_[cp_].data_ptr<int>(),
                channel_, height_, width_, npart_, pad_, 4);
        
        pseudo_context_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
            (count, param2_[cp_].data_ptr<float>(), param_[cp_].data_ptr<float>(), hindex_[width_].data_ptr<int>(),
                channel_, height_, width_, npart_, pad_, 4, rt_*width_, stride_inv_[width_].data_ptr<int>());
        CUDA_POST_KERNEL_CHECK;
    }
    return {param_[cp_], param2_[cp_], hindex_[width_], hindex2_[cp_], stride_inv_[width_]};
}

at::Tensor pseudo_context_opt::produce_param_fill(int height, int width) 
{
    reshape_hw(height,width);
    return hindex_[width_];
}