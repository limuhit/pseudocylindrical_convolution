#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "math.h"
#include "base_opt.hpp"
class projects_opt: public base_opt{
	public:
		projects_opt(int h_out, int w_out, std::vector<float> theta, std::vector<float> phi, float fov = 0.33333, bool near=false, int device =  0, bool timeit=false){
			pi_ = acos(-1.0);
			for(int i=0;i<14;i++){
				theta_[i] = theta[i]*pi_;
				phi_[i] = phi[i]*pi_;
			}
			h_out_ = h_out;
			w_out_ = w_out; 
			fov_ = fov*pi_;
			near_ = near;
			base_opt_init(device, timeit);
		}
		~projects_opt(){}
		void init();
		void update();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		at::Tensor xyz_,tf_, r1_, r2_, r_;
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		bool  near_;
		float fov_;
		float theta_[14], phi_[14];
		float c_x_, c_y_, w_stride_, h_stride_, pi_;
};
