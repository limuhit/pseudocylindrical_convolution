#include "entropy_ctx_pad_run2.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void entropy_ctx_pad_run2_opt::init(){
    init_base();
}

void entropy_ctx_pad_run2_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    h_out_ = height_ + 2*pad_;
    w_out_ = width_ + 2*pad_;
    cpn_ = channel_ / ngroup_;
    mod_ = height_ * npart_ + width_ + pad_ + ngroup_ - 2;
    pidx_ = 0;
    stride_entropy_ = (npart_*2+2)*pad_*2;
    n_out_ = num / npart_;
    //printf("%d %d %d %d\n",channel,height,width,pad_);
    std::vector<at::Tensor> tmp = ctx_->produce_param(channel, height, width, pad_);
    hindex_ = tmp[0];  
    hindex2_ = tmp[1]; 
    param_ = tmp[2];
    param2_ = tmp[3]; 
    pad_idx_ = tmp[4];
    //printf("%d\n",hindex_.size(0));
    //printf("%d %d\n",hindex2_.size(0),hindex2_.size(1));
    //printf("%d %d %d %d\n",param_.size(0),param_.size(1),param_.size(2),param_.size(3));
}

template <typename scalar_t>
__global__ void entropy_ctx_pad_run2_forward_kernel(const int nthreads, const scalar_t* const input,  scalar_t * const output, 
    const scalar_t * param, const int * param2, const int * hindex, const int * hindex2,
    const int psum, const int start_idx,  const int ntile, const int cpn,
     const int astride, const int astride_out, const int width, const int pad) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int pa = index % ntile;
       int abase = (start_idx + pa)*3;
       int ppc = (index /  ntile) % cpn;
       int tn = index /  ntile / cpn;
       int pc = (psum - param2[abase+2])*cpn + ppc;
       int pbase,qbase;
       if(param2[abase+1]<0){
            int tbase = param2[abase];
            int base = tbase*4;
            int pw = tbase % width;
            int qg = hindex2[tbase/width];
            pbase = static_cast<int>(param[base]+1e-6) + tn*astride_out + pc*astride;
            qbase = static_cast<int>(param[base+1]+1e-6)+ tn*astride_out + pc*astride;
            int qw = param[base+2]>=0 ? static_cast<int>(param[base+2]+1e-6) : -1;
            scalar_t qdata = (qw == -1) ?  0 : input[qbase + qw +pad];
            scalar_t t = param[base+3];
            int qww = (qw + 1) % hindex[qg];
            //printf("%d %d %d %d %d %d\n", index,  base, pbase, qbase, qw, qg);
            //printf("%d %d %d\n", pbase+pw+pad, qbase+qw+pad, qbase+qww+pad);
            output[pbase + pw + pad] =  qdata * t + input[qbase + qww + pad]*(1-t);
       }else{     
           pbase = static_cast<int>(param2[abase]+1e-6)  + tn*astride_out + pc*astride;
           qbase = static_cast<int>(param2[abase+1]+1e-6)  + tn*astride_out + pc*astride;
           output[pbase] = output[qbase];
       }
       
   }
}

template <typename scalar_t>
__global__ void entropy_ctx_pad_run2_zero_kernel(const int nthreads,  scalar_t * const output, 
    const int * hindex, const int wout, const int hout, const int width, const int height, 
    const int channel, const int npart, const int pad){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ph = (index / wout) % hout;
        if(ph<pad || ph >= pad + height){
            output[index] = 0;
            continue;
        }
        int pw = index % wout;
        int pg = (index / wout / hout / channel) % npart;
        if(pw<pad || pw >= pad + hindex[pg]){
            output[index] = 0;
        }
    }
}


std::vector<at::Tensor>  entropy_ctx_pad_run2_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2)-2*pad_, bottom_data.size(3)-2*pad_);
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "entropy_ctx_pad_run_forward_cuda", 
			([&] {
                    if(input_){
                        psum = psum - 1;
                    }
                    if(psum>=0 && psum<mod_){
                        int * id_list = pad_idx_.data_ptr<int>();
                        int st = psum - ngroup_ + 1 < 0 ? 0 : psum - ngroup_ + 1;
                        int end = psum < height_*npart_ + width_ + pad_ - 2 ? psum + 1 : height_*npart_ + width_ + pad_ - 1;
                        int ntile = id_list[end] - id_list[st];
                        if(ntile>0){
                            count = n_out_ * cpn_ * ntile;
                            //printf("%d %d %d %d\n", n_out_, cpn_, stride_entropy_, end-st);
                            entropy_ctx_pad_run2_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom_data.data_ptr<scalar_t>(),  bottom_data.data_ptr<scalar_t>(), 
                                    param_.data_ptr<scalar_t>(), param2_.data_ptr<int>(), hindex_.data_ptr<int>(), hindex2_.data_ptr<int>(),
                                    psum, id_list[st], ntile, cpn_, h_out_*w_out_, h_out_*w_out_*channel_*npart_,  width_, pad_);
                            CUDA_POST_KERNEL_CHECK;
                        }
                    }
   			    }
			)
    );
    return {bottom_data};
}
