#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class entropy_gmm_table_opt: public base_opt{
	public:
		entropy_gmm_table_opt(int nstep, float bias, int num_gaussian, float total_region, float beta=1e-6, int device = 0, bool timeit=false){
			nstep_ = nstep;
			bias_ = bias;
			beta_ = beta;
			num_gaussian_ = num_gaussian;
			total_region_ = total_region;
			base_opt_init(device,timeit);
		}
		~entropy_gmm_table_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_top_batch(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor weight, at::Tensor delta, at::Tensor mean, at::Tensor tnum);
		std::vector<at::Tensor>  forward_batch_cuda(at::Tensor  bottom_data, at::Tensor tnum);
		int nstep_;
		float bias_,beta_;
		int num_gaussian_;
		float total_region_;
};
