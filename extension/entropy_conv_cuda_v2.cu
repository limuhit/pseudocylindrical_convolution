#include "entropy_conv_v2.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void entropy_conv_opt2::init(){
    init_base();
}

void entropy_conv_opt2::reshape(int num, int channel, int height, int width){
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

void entropy_conv_opt2::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,nout_,height_+2*pad_out_,width_+2*pad_out_});
    reshape_top_base(option,shapes);
}

template<class T>
struct SharedMemory
{
    __device__ inline operator  T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};


struct idx_p3{
    int a,b,c;
};
__device__ inline idx_p3 produce3(int n, int n1, int n2)
{
    idx_p3 x;
    int p = n;
    x.a = p % n1;
    p = p / n1;
    x.b = p % n2;
    x.c = p / n2;
    return x;
}
//channel*kernel_size*num*group_out*len*kernel_size, blockSize=128
template <typename scalar_t,int blockSize>
__global__ void entropy_conv2_data_to_col_gpu_v3(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int kernel_size, const int skernel,  const int half_kernel, const int group_in, 
    const int group_out, const int height, const int width, const int start_idx,  const int psum, const int inner_shape, 
    const int channel, const int nout, const int npart, const int pad_in, const int pad_out, 
	const int index_stride, const int constrain) {
	scalar_t * sdata = SharedMemory<scalar_t>();
	scalar_t sum = 0;
	int tid = threadIdx.x;
	int pout, qn, th, tw;
	for(int index = tid; index < nblock; index += blockSize){
		idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid
		int hw = mindex[oid.a + start_idx];
		tw = hw % width;
		int hp = hw / width;
		int tg = hp / height;
		th = hp % height;
		int ph = th - half_kernel + iid.b;
		int qh = hp - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
		qn = oid.c*npart + tg;
		int tc =  psum - tw - hp; 
		int nchannel = constrain==5?(psum - qh - pw) * group_in:(psum - qh - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
			int weight_base =  (pout * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (qn * channel* (height+2*pad_in) + ph + pad_in) * (width+2*pad_in) + pw + pad_in;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
			}
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
		int out_idx = ((qn*nout+pout)*(height+2*pad_out)+th+pad_out)*(width+2*pad_out) + tw + pad_out;
		output[out_idx] = sum+bias[pout];
	}
}

std::vector<at::Tensor>  entropy_conv_opt2::forward_cuda(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2)-2*pad_in_, bottom_data.size(3)-2*pad_in_);
    reshape_top(bottom_data.options());
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
    const int* start_idx_ = plan_idx_.data_ptr<int>();
    int st = psum - ngroup_ + 1 < 0 ? 0 : psum - ngroup_ + 1;
    int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
    int len_idx = start_idx_[end] - start_idx_[st];
    
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "entropy_conv2_forward_cuda", 
			([&] {
                if(psum<mod_ && len_idx>0){
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*(height_+2*pad_out_)*(width_+2*pad_out_)*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_out_ * group_out_ * len_idx;
                    entropy_conv2_data_to_col_gpu_v3<scalar_t,blockSize><<<count, blockSize, blockSize*sizeof(scalar_t), stream_>>>
                        (skernel_*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), kernel_size_, skernel_,  kernel_size_/2, group_in_, 
                            group_out_, height_, width_, start_idx_[st],  psum, len_idx, channel_,  nout_, npart_, pad_in_, pad_out_, 
                            (height_+2*pad_in_)*(width_+2*pad_in_), constrain_);
                    CUDA_POST_KERNEL_CHECK;
                }
   			})
    );
    return top_data_;
}

template <typename scalar_t,int blockSize>
__global__ void entropy_conv2_data_to_col_gpu_v3_act(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int kernel_size, const int skernel,  const int half_kernel, const int group_in, 
    const int group_out, const int height, const int width, const int start_idx,  const int psum, const int inner_shape, 
    const int channel, const int nout, const int npart, const int pad_in, const int pad_out, 
	const int index_stride, const int constrain,  const scalar_t* act_param) {
	scalar_t * sdata = SharedMemory<scalar_t>();
	scalar_t sum = 0;
	int tid = threadIdx.x;
	int pout, qn, th, tw;
	for(int index = tid; index < nblock; index += blockSize){
		idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid
		int hw = mindex[oid.a + start_idx];
		tw = hw % width;
		int hp = hw / width;
		int tg = hp / height;
		th = hp % height;
		int ph = th - half_kernel + iid.b;
		int qh = hp - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
		qn = oid.c*npart + tg;
		int tc =  psum - tw - hp; 
		int nchannel = constrain==5?(psum - qh - pw) * group_in:(psum - qh - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
			int weight_base =  (pout * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (qn * channel* (height+2*pad_in) + ph + pad_in) * (width+2*pad_in) + pw + pad_in;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
			}
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
		int out_idx = ((qn*nout+pout)*(height+2*pad_out)+th+pad_out)*(width+2*pad_out) + tw + pad_out;
		sum = sum+bias[pout];
        if(sum<0) sum = sum * act_param[pout];
        output[out_idx] = sum;
	}
}

std::vector<at::Tensor>  entropy_conv_opt2::forward_act_cuda(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2)-2*pad_in_, bottom_data.size(3)-2*pad_in_);
    reshape_top(bottom_data.options());
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
    const int* start_idx_ = plan_idx_.data_ptr<int>();
    int st = psum - ngroup_ + 1 < 0 ? 0 : psum - ngroup_ + 1;
    int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
    int len_idx = start_idx_[end] - start_idx_[st];
    
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "entropy_conv2_forward_act_cuda", 
			([&] {
                if(psum<mod_&& len_idx>0){
                    count = skernel_ * group_out_ * len_idx * num_out_;
                    //cudaMemset(tmp_.data_ptr<scalar_t>(), scalar_t(0.0), count*sizeof(scalar_t));
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*(height_+2*pad_out_)*(width_+2*pad_out_)*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_out_ * group_out_ * len_idx;
                    entropy_conv2_data_to_col_gpu_v3_act<scalar_t,blockSize><<<count, blockSize, blockSize*sizeof(scalar_t), stream_>>>
                        (skernel_*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), kernel_size_, skernel_,  kernel_size_/2, group_in_, 
                            group_out_, height_, width_, start_idx_[st],  psum, len_idx, channel_,  nout_, npart_, pad_in_, pad_out_, 
                            (height_+2*pad_in_)*(width_+2*pad_in_), constrain_, act.data_ptr<scalar_t>());
                    CUDA_POST_KERNEL_CHECK;
                }
   			})
    );
    return top_data_;
}



template <typename scalar_t,int blockSize>
__global__ void entropy_conv2_data_to_col_gpu_v3_batch(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int kernel_size, const int skernel,  const int half_kernel, const int group_in, 
    const int group_out, const int height, const int width, const int start_idx,  const int psum, const int inner_shape, 
    const int channel, const int nout, const int npart, const int pad_in, const int pad_out, 
	const int index_stride, const int constrain, const int num_per_batch) {
	scalar_t * sdata = SharedMemory<scalar_t>();
	scalar_t sum = 0;
	int tid = threadIdx.x;
	int pout, qn, th, tw, nbatch;
	for(int index = tid; index < nblock; index += blockSize){
		idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid
        nbatch = oid.c/num_per_batch;
		int hw = mindex[oid.a + start_idx];
		tw = hw % width;
		int hp = hw / width;
		int tg = hp / height;
		th = hp % height;
		int ph = th - half_kernel + iid.b;
		int qh = hp - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
		qn = oid.c*npart + tg;
		int tc =  psum - tw - hp; 
		int nchannel = constrain==5?(psum - qh - pw) * group_in:(psum - qh - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
			int weight_base =  ((nbatch * nout + pout) * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (qn * channel* (height+2*pad_in) + ph + pad_in) * (width+2*pad_in) + pw + pad_in;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
			}
            //printf("%d %d %d %d %d %d %d %d\n",oid.a,oid.b,oid.c,iid.a,iid.b,iid.c, data_base, weight_base);
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
		int out_idx = ((qn*nout+pout)*(height+2*pad_out)+th+pad_out)*(width+2*pad_out) + tw + pad_out;
        int bidx = nbatch*nout + pout;
		output[out_idx] = sum + bias[bidx];
        //printf()
	}
}

std::vector<at::Tensor>  entropy_conv_opt2::forward_cuda_batch(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2)-2*pad_in_, bottom_data.size(3)-2*pad_in_);
    reshape_top(bottom_data.options());
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
    const int* start_idx_ = plan_idx_.data_ptr<int>();
    int st = psum - ngroup_ + 1 < 0 ? 0 : psum - ngroup_ + 1;
    int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
    int len_idx = start_idx_[end] - start_idx_[st];
    int num_per_batch = num_out_ / weight.size(0);
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "entropy_conv2_forward_cuda_batch", 
			([&] {
                if(psum<mod_ && len_idx>0){
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*(height_+2*pad_out_)*(width_+2*pad_out_)*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_out_ * group_out_ * len_idx;
                    entropy_conv2_data_to_col_gpu_v3_batch<scalar_t,blockSize><<<count, blockSize, blockSize*sizeof(scalar_t), stream_>>>
                        (skernel_*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), kernel_size_, skernel_,  kernel_size_/2, group_in_, 
                            group_out_, height_, width_, start_idx_[st],  psum, len_idx, channel_,  nout_, npart_, pad_in_, pad_out_, 
                            (height_+2*pad_in_)*(width_+2*pad_in_), constrain_, num_per_batch);
                    CUDA_POST_KERNEL_CHECK;
                }
   			})
    );
    return top_data_;
}

template <typename scalar_t,int blockSize>
__global__ void entropy_conv2_data_to_col_gpu_v3_act_batch(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int kernel_size, const int skernel,  const int half_kernel, const int group_in, 
    const int group_out, const int height, const int width, const int start_idx,  const int psum, const int inner_shape, 
    const int channel, const int nout, const int npart, const int pad_in, const int pad_out, 
	const int index_stride, const int constrain,  const scalar_t* act_param, const int num_per_batch) {
	scalar_t * sdata = SharedMemory<scalar_t>();
	scalar_t sum = 0;
	int tid = threadIdx.x;
	int pout, qn, th, tw, nbatch;
	for(int index = tid; index < nblock; index += blockSize){
		idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid
        nbatch = oid.c/num_per_batch;
		int hw = mindex[oid.a + start_idx];
		tw = hw % width;
		int hp = hw / width;
		int tg = hp / height;
		th = hp % height;
		int ph = th - half_kernel + iid.b;
		int qh = hp - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
		qn = oid.c*npart + tg;
		int tc =  psum - tw - hp; 
		int nchannel = constrain==5?(psum - qh - pw) * group_in:(psum - qh - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
			int weight_base =  ((nbatch * nout + pout) * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (qn * channel* (height+2*pad_in) + ph + pad_in) * (width+2*pad_in) + pw + pad_in;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
			}
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
		int out_idx = ((qn*nout+pout)*(height+2*pad_out)+th+pad_out)*(width+2*pad_out) + tw + pad_out;
        int bidx = nbatch*nout + pout;
		sum = sum+bias[bidx];
        if(sum<0) sum = sum * act_param[bidx];
        output[out_idx] = sum;
	}
}

std::vector<at::Tensor>  entropy_conv_opt2::forward_act_cuda_batch(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2)-2*pad_in_, bottom_data.size(3)-2*pad_in_);
    reshape_top(bottom_data.options());
	int count;
    int psum = pidx_;
    pidx_ = pidx_ + 1;
    const int* start_idx_ = plan_idx_.data_ptr<int>();
    int st = psum - ngroup_ + 1 < 0 ? 0 : psum - ngroup_ + 1;
    int end = psum < h_out_ + w_out_ - 2 ? psum + 1 : h_out_ + w_out_ - 1;
    int len_idx = start_idx_[end] - start_idx_[st];
    int num_per_batch = num_out_ / weight.size(0);
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "entropy_conv2_forward_act_cuda_batch", 
			([&] {
                if(psum<mod_&& len_idx>0){
                    count = skernel_ * group_out_ * len_idx * num_out_;
                    //cudaMemset(tmp_.data_ptr<scalar_t>(), scalar_t(0.0), count*sizeof(scalar_t));
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*(height_+2*pad_out_)*(width_+2*pad_out_)*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_out_ * group_out_ * len_idx;
                    entropy_conv2_data_to_col_gpu_v3_act_batch<scalar_t,blockSize><<<count, blockSize, blockSize*sizeof(scalar_t), stream_>>>
                        (skernel_*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), kernel_size_, skernel_,  kernel_size_/2, group_in_, 
                            group_out_, height_, width_, start_idx_[st],  psum, len_idx, channel_,  nout_, npart_, pad_in_, pad_out_, 
                            (height_+2*pad_in_)*(width_+2*pad_in_), constrain_, act.data_ptr<scalar_t>(), num_per_batch);
                    CUDA_POST_KERNEL_CHECK;
                }
   			})
    );
    return top_data_;
}
