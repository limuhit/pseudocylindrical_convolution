#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "pseudo_context.hpp"
#include "string2class.hpp"
#include <string>
class pseudo_quant_opt:public base_opt{
	public:
		pseudo_quant_opt(int channel, int bin_num, int npart, float weight_decay, int check_iters, int ntop, float top_alpha, std::string addr, int device=0, bool timeit=false){
			channel_ = channel;
			bin_num_ = bin_num;
			mod_ = check_iters;
			ntop_ = ntop;
			weight_decay_ = weight_decay;
			top_alpha_ = top_alpha;
			npart_ = npart;
			ctx_ = FromStringPseudo(addr);
			base_opt_init(device,timeit);
		}
		~pseudo_quant_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void update_weight(at::Tensor weight, at::Tensor);
		void reshape_bottom(at::TensorOptions options);
		at::Tensor count_data_, quant_, weight_;
		std::vector<at::Tensor>  quant_forward_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor ncount, bool train);
		std::vector<at::Tensor>  quant_backward_cuda(std::vector<at::Tensor>  top_diff, at::Tensor bottom_data, at::Tensor hindex);
		float weight_decay_, top_alpha_;
		int bin_num_;
		int iter_, mod_, ntop_;
		int npart_;
		at::Tensor hindex_;
		pseudo_context_opt * ctx_;
};
