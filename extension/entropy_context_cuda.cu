
#include "entropy_context.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void entropy_context::init(){
    init_base();
}

bool entropy_context::reshape_hw(int height, int width){
    height_ = height;
    width_ = width;
    if(width_dict_.count(width)>0) return false;
    h_out_ = height_ * npart_;
    w_out_ = width_;
    width_dict_[width_] = 1;
    at::Tensor hindex_tmp = at::zeros({npart_}, at::kInt);
    int * hindex = hindex_tmp.data_ptr<int>();
    sphere_cal_npart_hw_v3(h_out_, w_out_, npart_, weight_, hindex);
    at::Tensor index_tmp = at::zeros({h_out_, w_out_},at::kInt);
    start_idx_[width_] = at::zeros({h_out_+w_out_},at::kInt);
    int * idx = index_tmp.data_ptr<int>();
    int * start_idx = start_idx_[width_].data_ptr<int>();
    int index = 0, jidx=0;
    for (int ps = 0; ps < h_out_ + w_out_ - 1; ps++) {
        start_idx[jidx] = index;
        jidx ++;
        for (int i = 0; i < h_out_; i++) {
            int j = ps - i;
            if (j < 0 || j >= hindex[i/height_])
                continue;
            idx[index] = i*w_out_ + j;
            index++;
        }
    }
    start_idx[jidx] = index;
    index_[width_] = index_tmp.to(torch::Device(torch::kCUDA, device_));
    //for(int i = 0;i<npart_;i++) printf("%d ", hindex[i]);
    hindex_[width_] = hindex_tmp.to(torch::Device(torch::kCUDA, device_));
    //printf("hw:%d %d %d %d %d %d\n",height_,width_,h_out_,w_out_,npart_,hindex_[width].size(0));
    return true;
}

bool entropy_context::reshape_channel_pad(int channel, int pad){
    channel_ = channel; 
    pad_ = pad;
    assert((channel<1000)&&(pad<10)&&"the channel number should be less than 1000 and the pad size should be less than 10");
    cp_ = width_*10000 + channel_*10 + pad_;
    if(pad_channel_dict_.count(cp_)>0) return false;
    pad_channel_dict_[cp_] = 1;
    
    stride_entropy_ = (npart_*2+2)*pad_*3;
    hindex2_[cp_] = at::zeros({npart_,2*pad_}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_));
    param2_[cp_] = at::zeros({h_out_+ w_out_ + pad_*2, stride_entropy_}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_));
    param_[cp_] = at::zeros({npart_,2*pad_, width_, 4}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, device_));
    pad_idx_[cp_] = at::zeros({h_out_+w_out_+ pad_*2},at::kInt);
    //printf("cp:%d %d %d %d %d %d %d\n",channel_,pad_,width_,npart_,h_out_,w_out_,cp_);
    return true;
}

__global__ void entropy_context_step1(const int nthreads, int * const output, 
    const float * param, const int * hindex, const int * hindex2, 
     const int width, const int height, const int pad, const int stride_entropy, const int stride_param) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int tg = index / width / pad / 2;
        if(tw>=hindex[tg]) continue;
        if(hindex2[(tg*2 + tl)*pad + tp]<0) continue;
        if(param[index*stride_param + 2] < 0 && param[index*stride_param + 3] >= 1-1e-6) continue;
        int ph = (tl==0) ? tg*height - pad + tp : (tg+1)*height + tp;
        int plan_idx = (ph+tw) * stride_entropy;
        int ti = atomicAdd(output + plan_idx + stride_entropy-1,  1.);
        if(ti*3+3>=stride_entropy) printf("plane idxs overflow!\n");
        output[plan_idx+ti*3] = index;
        output[plan_idx+ti*3+1] = -1;
        output[plan_idx+ti*3+2] = ph + tw;
    }
}
//template <typename scalar_t>
__global__ void entropy_context_step2(const int nthreads, int * const output, const int * hindex,
     const int width, const int height, const int channel, const int npart, const int pad, const int stride_entropy) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int qh = height+2*pad;
        int tw = index % pad;
        int th = (index / pad) % qh;
        int tg = index / pad / qh;
        int ph = tg*height + th - pad;
        if(ph<0 || ph >= height*npart) continue;
        int wp = hindex[tg];
        int plan_idx = (ph+tw+wp) * stride_entropy;
        int base =  (tg *channel*qh + th)*(width+2*pad) + tw + pad;
        int ti = atomicAdd(output + plan_idx + stride_entropy-1,  1.);
        if(ti*3+3>=stride_entropy) printf("plane idxs overflow!\n");
        output[plan_idx+ti*3] = base + wp;
        output[plan_idx+ti*3+1] = base;
        output[plan_idx+ti*3+2] = ph+tw+wp;
    }
}

__global__ void entropy_context_kernel(const int nthreads, float * const param, const int * hindex, int * hindex2,
    const int channel, const int height, const int width, const int npart, const int pad, const int stride) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int tg = index / width / pad / 2;
        if(tw>=hindex[tg]) continue;
        int ph, pg;
        float pw;
        bool bound = false;
        //printf("%d %d\n",index,hindex[tg]);
        if(tl==0){
            ph = tg*height - pad + tp;
            if(ph<0){
                bound = true;
            }else{
                pg = ph / height;
                pw = (tw + 0.5) / hindex[tg] * hindex[pg] - 0.5 + 1e-9;
            }
        }else{
            ph = (tg+1)*height + tp;
            if(ph>=height*npart){
                bound = true;
            }else{
                pg = ph / height;
                pw = (tw + 0.5) / hindex[tg] * hindex[pg] - 0.5 + 1e-9;
            }
        }
        
        if(bound){
            param[index*stride] =  (tl == 0) ? tg *channel*(height+pad*2) + tp : tg *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            if(tw==0){
                hindex2[(tg*2 + tl)*pad + tp] = -1;
            }
        }else{
            param[index*stride] =  (tl == 0) ? tg *channel*(height+pad*2) + tp : tg *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            //param[index*stride+1] = (pg*channel*height + ph % height) * width; 
            param[index*stride+1] = (pg*channel*(height+pad*2) + pad + ph % height) * (width+pad*2);
            int pidx = pw<0? -1 : static_cast<int> (pw);
            if(pidx>tw){
                param[index*stride + 2] = -1;
                param[index*stride + 3] = 1.;
            }else if(pidx+1>tw){
                param[index*stride + 2] = pidx;
                param[index*stride + 3] = 1.;
            }else{
                param[index*stride + 2] = pidx;
                param[index*stride + 3] = pidx+1-pw;
                if(pidx==-1){
                    param[index*stride + 3] = 0.;
                }
            }
            if(tw==0){
                hindex2[(tg*2 + tl)*pad + tp] = pg;
            }
        }
    }
}


std::vector<at::Tensor> entropy_context::produce_param(int channel, int height, int width, int pad) 
{
    reshape_hw(height,width);
    bool change_flag = reshape_channel_pad(channel,pad);
	int count;
    if(change_flag){
        count = npart_  * width_ * pad_ * 2;
        caffe_gpu_set(stream_, count*4, float(0), param_[cp_].data_ptr<float>());
        caffe_gpu_set(stream_, (h_out_+ w_out_ + pad_*2)*stride_entropy_, 0, param2_[cp_].data_ptr<int>());
        entropy_context_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
            (count, param_[cp_].data_ptr<float>(), hindex_[width_].data_ptr<int>(), hindex2_[cp_].data_ptr<int>(),
                channel_, height_, width_, npart_, pad_, 4);
        count = npart_  * width_ * pad_ * 2;
        entropy_context_step1<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
            (count, param2_[cp_].data_ptr<int>(), param_[cp_].data_ptr<float>(), hindex_[width_].data_ptr<int>(),
                hindex2_[cp_].data_ptr<int>(), width_, height_, pad_, stride_entropy_, 4);
        count = npart_  * (height_+2*pad_)* pad_;
        entropy_context_step2<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
            (count, param2_[cp_].data_ptr<int>(), hindex_[width_].data_ptr<int>(), width_, height_, channel_, npart_, pad_, stride_entropy_);
        at::Tensor tmp = param2_[cp_].to(torch::Device(torch::kCPU));
        int * pid = pad_idx_[cp_].data_ptr<int>();
        int * prm = tmp.data_ptr<int>();
        int ti = 1;
        pid[0] = 0;
        for(int i=0; i < h_out_ + w_out_ + pad_-1;i++,ti++){
            pid[ti] = pid[ti-1] + prm[i*stride_entropy_+stride_entropy_-1];
        }
        for(int i=1; i < h_out_ + w_out_ + pad_-1;i++){
            for(int j=0;j<prm[i*stride_entropy_+stride_entropy_-1];j++){
                int pra = (pid[i]+j)*3;
                int prb = i*stride_entropy_+j*3;
                prm[pra] = prm[prb];
                prm[pra+1] = prm[prb+1];
                prm[pra+2] = prm[prb+2];
            }
        }
        param2_[cp_] = tmp.to(torch::Device(torch::kCUDA, device_));
        CUDA_POST_KERNEL_CHECK;
    }
    //printf("here2: %d %d %d %d %d %d\n", width_,height_,pad_,channel_,npart_,stride_entropy_);
    return {hindex_[width_],  hindex2_[cp_], param_[cp_], param2_[cp_], pad_idx_[cp_]};
}

std::vector<at::Tensor> entropy_context::produce_param_group(int height, int width) 
{
    reshape_hw(height,width);
    return {index_[width_], start_idx_[width_]};
}

at::Tensor entropy_context::produce_param_fill(int height, int width) 
{
    reshape_hw(height,width);
    return hindex_[width_];
}

