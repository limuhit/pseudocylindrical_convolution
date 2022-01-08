#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class sphere_uslice_opt: public base_opt{
	public:
		sphere_uslice_opt(int npart, int interp_type, int pad, std::vector<float> weight, int device = 0, bool timeit=false){
			npart_ = npart;
			interp_type_ = interp_type;
			weight_ = new float[npart_];
			for(int i = 0; i<weight.size();i++)
				weight_[i] = weight[i];
			pad_ = pad;
			base_opt_init(device,timeit);
		}
		~sphere_uslice_opt(){delete weight_;}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		at::Tensor hindex_, resize_param_, inv_param_, hinv_, stride_inv_;
		int npart_;
		int interp_type_;
		float * weight_;
		int pad_;
		int n_out_, rt_;
		bool init_param_ , init_inv_;
};
