#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "entropy_context.hpp"
#include "string2class.hpp"
class entropy_ctx_pad_run2_opt: public base_opt{
	public:
		entropy_ctx_pad_run2_opt(int pad, int npart, int ngroup, bool input, std::string ctx_addr, int device = 0, bool timeit=false){
			pad_ = pad;
			npart_ = npart;
			ngroup_ = ngroup;
			input_ = input;
			//printf("you are here!\n");
			ctx_ = FromString(ctx_addr);
			base_opt_init(device,timeit);
		}
		~entropy_ctx_pad_run2_opt(){}
		void init();
		void restart(){pidx_=0;}
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options){};
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff){return {};}
		int pad_;
		int npart_, ngroup_, cpn_;
		int mod_, pidx_;
		int stride_entropy_;
		int n_out_;
		bool input_;
		at::Tensor hindex_,  hindex2_, param_, param2_, pad_idx_;
		entropy_context * ctx_;
};
