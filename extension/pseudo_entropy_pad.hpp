#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "pseudo_entropy_context.hpp"
#include "string2class.hpp"
#include <string>
class pseudo_entropy_pad_opt: public base_opt{
	public:
		pseudo_entropy_pad_opt(int pad, int npart, std::string ctx_addr, int device = 0, bool timeit=false){
			pad_ = pad;
			npart_ = npart;
			ctx_ = FromStringPseudoEntropy(ctx_addr);
			base_opt_init(device,timeit);
		}
		~pseudo_entropy_pad_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int pad_;
		int npart_;
		bool init_param_, lfour_;
		at::Tensor hindex_,  hindex2_, param_, inv_param_, stride_inv_;
		pseudo_entropy_context_opt * ctx_;
};
