#include "projects.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"
__global__ void projects_init_xyz_kernel(int num, float * data, int height, int width, float w_stride, float h_stride, float c_x, float c_y){
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        float x = 1.;
        float y = (w - c_x)*w_stride;
        float z = (h - c_y)*h_stride;
        float r = sqrt(x*x + y*y + z*z);
        data[i*3] = x/r;
        data[i*3+1] = y/r;
        data[i*3+2] = -z/r;
    }
}
void projects_mrod(float* x, float* y, float* z, float* data){
    float norm, tx,ty,tz,c,s;
    int base;
    for(int i = 0; i < 14; i++){
        base = i * 9;
        norm = sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
        if(norm == 0)
        {
            data[base] = 1.;
            data[base+4] = 1.;
            data[base+8] = 1.;
            continue;
        }
        tx = x[i] / norm;
        ty = y[i] / norm;
        tz = z[i] / norm;
        c = cos(norm);
        s = sin(norm);
        data[base + 0] = c + (1-c)*tx*tx;
        data[base + 1] =     (1-c)*tx*ty - s*tz;
        data[base + 2] =     (1-c)*tx*tz + s*ty;
        data[base + 3] =     (1-c)*ty*tx + s*tz;
        data[base + 4] = c + (1-c)*ty*ty;
        data[base + 5] =     (1-c)*ty*tz - s*tx;
        data[base + 6] =     (1-c)*tz*tx - s*ty;
        data[base + 7] =     (1-c)*tz*ty + s*tx;
        data[base + 8] = c + (1-c)*tz*tz;
    }
    return ;
}
__global__ void projects_cal_xyz_kernel(int num, float * const xyz, float * tf,  int height, int width, float hx, float hy, float pi){
    CUDA_KERNEL_LOOP(i, num) {
        //int w = i % width;
        //int h = (i / width) % height;
        float lat = asin(xyz[i*3+2]);
        float tx = xyz[i*3];
        float ty = xyz[i*3+1];
        float theta = atan(ty/tx);
        if (tx<=0){
            if(ty>0){
                theta = theta + pi;
            }else{
                theta = theta - pi;
            }
        }
        tf[i*2] = theta / pi * hx + hx;
        tf[i*2+1] = -2 * lat / pi * hy + hy; 
    }
}
__global__ void gmm_kernel(int num, const float * const x, const float * y, float *z, const int m, const int k, const int n, const int batch){
    CUDA_KERNEL_LOOP(i, num) {
        int tm = (i / n) % m;
        int tn = i % n;
        int tb = i / n / m;
        int base_x =  tb * m * k;
        int base_y = tb * k * n;
        int base_z = tb * m * n;
        float sum = 0;
        for(int j = 0; j<k; j++){
            sum += x[base_x + tm*k + j]*y[base_y + j*n + tn];
        }
        z[base_z + tm*n + tn] = sum;
    }
}
__global__ void gmm_transpose_kernel(int num, float * const x, const float * y, const int m, const int batch){
    CUDA_KERNEL_LOOP(i, num) {
        int tb = i / m;
        int tm = i % m;
        int base_x =  tb * m * 3;
        int base_y = tb * 3 * 3;
        float xa = x[base_x+tm*3];
        float xb = x[base_x+tm*3 + 1];
        float xc = x[base_x+tm*3 + 2];
        x[base_x+tm*3] = xa * y[base_y] + xb * y[base_y+1] + xc * y[base_y+2];
        x[base_x+tm*3 + 1] = xa * y[base_y+3] + xb * y[base_y+4] + xc * y[base_y+5];
        x[base_x+tm*3 + 2] = xa * y[base_y+6] + xb * y[base_y+7] + xc * y[base_y+8];
    }
}
void projects_opt::init(){
    init_base();
    height_ = -1;
    width_ = -1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_).requires_grad(false);
    xyz_ =  at::zeros({14, h_out_ * w_out_, 3},options);
    tf_ = at::zeros({14, h_out_*w_out_, 2},options);
    float hfov = fov_ * h_out_ / w_out_ /2;
    float wfov = fov_ / 2;
    c_x_ = (w_out_-1) / 2.0;
    c_y_ = (h_out_-1) / 2.0;
    float pi_2 = pi_ / 2;
    float wangle = pi_2 - wfov;
    float hangle = pi_2 - hfov;
    w_stride_ = 2 * sin(wfov) / sin(wangle) / (w_out_ - 1);
    h_stride_ = 2 * sin(hfov) / sin(hangle) / (h_out_ - 1);
    int count = 14* h_out_*w_out_;
    projects_init_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>> >
        (count, xyz_.data_ptr<float>(),h_out_,w_out_,w_stride_,h_stride_, c_x_, c_y_);
    CUDA_POST_KERNEL_CHECK;
    r1_ = at::zeros({14,3,3},at::kFloat);
    r2_ = at::zeros({14,3,3},at::kFloat);
    r_ = at::zeros({14,3,3}, options);
    float * r1 = r1_.data_ptr<float>();
    float * r2 = r2_.data_ptr<float>();
    float xa[14],ya[14],za[14];
    for(int i = 0; i<14; i++){
        xa[i] = 0;
        ya[i] = 0;
        za[i] = theta_[i];
    }
    projects_mrod(xa,ya,za,r1);
    for(int i = 0; i<14; i++){
        xa[i] = r1[i*9+1]*(-phi_[i]);
        ya[i] = r1[i*9+4]*(-phi_[i]);
        za[i] = r1[i*9+7]*(-phi_[i]);
    }
    projects_mrod(xa,ya,za,r2);
    count = 14*9;
    r1_ = r1_.to(torch::Device(torch::kCUDA, device_));
    r2_ = r2_.to(torch::Device(torch::kCUDA, device_));
    gmm_kernel<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
        (count, r2_.data_ptr<float>(), r1_.data_ptr<float>(), r_.data_ptr<float>(),3,3,3, 14);
    CUDA_POST_KERNEL_CHECK;
    count = 14 * h_out_ * w_out_;
    gmm_transpose_kernel<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
        (count, xyz_.data_ptr<float>(), r_.data_ptr<float>(),h_out_*w_out_, 14);
}
void projects_opt::update(){
    int count = 14* h_out_*w_out_;
    float hx = (width_ - 1) / 2.0;
    float hy = (height_ - 1) / 2.0;
    //printf("%d %f %f\n", count, hx,hy);
    projects_cal_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
        (count, xyz_.data_ptr<float>(),tf_.data_ptr<float>(), h_out_,w_out_,hx, hy, pi_);
    CUDA_POST_KERNEL_CHECK;
    //printf("k4\n");
}
void projects_opt::reshape(int num, int channel, int height, int width){
   if(height_ != height || width_ != width){
        height_ = height;
        width_ = width;
        update();
   }
   channel_ = channel;
   num_ = num;
}

void projects_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*14,channel_,h_out_,w_out_});
    reshape_top_base(options,shapes);
}

void projects_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void projects_forward_kernel(const int nthreads, const scalar_t* const input,  
    const float * tf, scalar_t * const output, const int inner_shape, const int hs, const int ws, const int out_shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int tn = (index / inner_shape) % out_shape;
        int tb = index / inner_shape / out_shape;
        int base = tb*2*inner_shape;
        int tw = static_cast<int>(floor(tf[base + 2*ps]));
        int th = static_cast<int>(floor(tf[base + 2*ps + 1]));
        int pw = (tw + 1) % ws;
        int ph = th + 1 >= hs ?  hs-1 : th + 1; 
        float tx = tf[base + 2*ps] - tw;
        float ty = tf[base + 2*ps+1] - th;
        float ntx = 1. - tx;
        float nty = 1. - ty;
        output[index] = input[(tn*hs+th)*ws + tw]*ntx*nty + input[(tn*hs+th)*ws + pw]*tx*nty +  input[(tn*hs+ph)*ws + tw]*ntx*ty + input[(tn*hs+ph)*ws + pw]*tx*ty; 
    }
}
template <typename scalar_t>
__global__ void projects_forward_kernel_nearest(const int nthreads, const scalar_t* const input,  
    const float * tf, scalar_t * const output, const int inner_shape, const int hs, const int ws, const int out_shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int tn = (index / inner_shape) % out_shape;
        int tb = index / inner_shape / out_shape;
        int base = tb*2*inner_shape;
        int tw = static_cast<int>(floor(tf[base + 2*ps]+0.5)) % ws;
        int th = static_cast<int>(floor(tf[base + 2*ps + 1]+0.5));
        th = th >= hs ?  hs-1 : th; 
        output[index] = input[(tn*hs+th)*ws + tw]; 
    }
}

template <typename scalar_t>
__global__ void projects_copy_kernel(const int nthreads, const float* const input,  
     scalar_t * const output) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        output[index] = input[index]; 
    }
}


std::vector<at::Tensor>  projects_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
    int count;
    
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "projects_forward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * w_out_ * h_out_ * 14;
                    if(near_){
                        projects_forward_kernel_nearest<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(), tf_.data_ptr<float>(), 
                             top_data_[0].data_ptr<scalar_t>(), h_out_*w_out_, height_,width_, num_*channel_);
                    }else{
                        projects_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(), tf_.data_ptr<float>(), 
                            top_data_[0].data_ptr<scalar_t>(), h_out_*w_out_, height_,width_, num_*channel_);
                    }
                    
                    /*
                    count = 2*w_out_*h_out_;
                    projects_copy_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count,  tf_.data_ptr<float>(), top_data_[1].data_ptr<scalar_t>());*/
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void projects_backward_kernel_nearest(const int nthreads, scalar_t* const input,  scalar_t * const count,
    const float * tf, const scalar_t * const output,  const int inner_shape, const int hs, const int ws, const int out_shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int mod = (out_shape * 14);
        int ps = index / mod;
        int tn = (index % mod) % out_shape;
        int tb = (index % mod) / out_shape; 
        int base = tb*2*inner_shape;
        int pidx = (index % mod) * inner_shape + ps;
        int tw = static_cast<int>(floor(tf[base + 2*ps]+0.5)) % ws;
        int th = static_cast<int>(floor(tf[base + 2*ps + 1]+0.5));
        th = th >= hs ?  hs-1 : th; 
        atomicAdd(input+(tn*hs+th)*ws + tw, output[pidx]);
        atomicAdd(count+(tn*hs+th)*ws + tw, 1.);
    }
}
template <typename scalar_t>
__global__ void projects_backward_kernel(const int nthreads, scalar_t* const input,  scalar_t* const count,
    const float * tf, const scalar_t * const output, const int inner_shape, const int hs, const int ws, const int out_shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int mod = (out_shape * 14);
        int ps = index / mod;
        int tn = (index % mod) % out_shape;
        int tb = (index % mod) / out_shape; 
        int base = tb*2*inner_shape;
        int pidx = (index % mod) * inner_shape + ps;
        int tw = static_cast<int>(floor(tf[base + 2*ps]));
        int th = static_cast<int>(floor(tf[base + 2*ps + 1]));
        int pw = (tw + 1) % ws;
        int ph = th + 1 >= hs ?  hs-1 : th + 1; 
        scalar_t tx = tf[base + 2*ps] - tw;
        scalar_t ty = tf[base + 2*ps+1] - th;
        scalar_t ntx = 1. - tx;
        scalar_t nty = 1. - ty;
        atomicAdd(input+(tn*hs+th)*ws + tw,ntx*nty* output[pidx]);
        atomicAdd(count+(tn*hs+th)*ws + tw,ntx*nty);
        atomicAdd(input+(tn*hs+th)*ws + pw,tx*nty* output[pidx]);
        atomicAdd(count+(tn*hs+th)*ws + pw,tx*nty);
        atomicAdd(input+(tn*hs+ph)*ws + tw,ntx*ty* output[pidx]);
        atomicAdd(count+(tn*hs+ph)*ws + tw,ntx*ty);
        atomicAdd(input+(tn*hs+ph)*ws + pw,tx*ty* output[pidx]);
        atomicAdd(count+(tn*hs+ph)*ws + pw,tx*ty);
    }
}
std::vector<at::Tensor>  projects_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
    int count;
    //printf("backward...\n");
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "projects_backward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    caffe_gpu_set(stream_, count,0,bottom_diff_[0].data_ptr<scalar_t>());
                    caffe_gpu_set(stream_, count,0,bottom_diff_[1].data_ptr<scalar_t>());
                    count = num_ * channel_ * w_out_ * h_out_*14;
                    if(near_){
                        projects_backward_kernel_nearest<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_diff_[0].data_ptr<scalar_t>(), bottom_diff_[1].data_ptr<scalar_t>(),
                            tf_.data_ptr<float>(), top_diff.data_ptr<scalar_t>(), h_out_*w_out_, height_, width_, num_*channel_);
                    }else{
                        projects_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_diff_[0].data_ptr<scalar_t>(), bottom_diff_[1].data_ptr<scalar_t>(),
                            tf_.data_ptr<float>(), top_diff.data_ptr<scalar_t>(), h_out_*w_out_, height_, width_, num_*channel_);
                    }
                    
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return bottom_diff_;
}