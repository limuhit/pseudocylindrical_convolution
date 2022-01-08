#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "pseudo_context.hpp"
#include "string2class.hpp"
#include <string>

class pseudo_dquant_opt: public base_opt{
	public:
		pseudo_dquant_opt(int npart, int channel, int bin_num, std::string addr,int device = 0, bool timeit=false){
			nchannel_ = channel;
			bin_num_ = bin_num;
			npart_ = npart;
			ctx_ = FromStringPseudo(addr);
			base_opt_init(device,timeit);
		}
		~pseudo_dquant_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor weight_old);
		int nchannel_;
		int bin_num_;
		int npart_;
		at::Tensor weight_, hindex_;
		pseudo_context_opt * ctx_;
};
