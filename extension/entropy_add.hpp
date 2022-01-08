#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "entropy_context.hpp"
#include "string2class.hpp"

class entropy_add_opt: public base_opt{
	public:
		entropy_add_opt(int npart, int channel, int ngroup, int pad, std::string ctx_addr, int device = 0, bool timeit=false){
			npart_ = npart;
			channel_ = channel;
			ngroup_ = ngroup;
			cpg_ = channel_ / ngroup_;
			pad_ = pad;
			ctx_ = FromString(ctx_addr);
			base_opt_init(device,timeit);
		}
		~entropy_add_opt(){}
		void init();
		void restart() {pidx_ = 0;}
		void reshape(int num, int channel, int height, int width);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor bottom_data2);
		int npart_;
		int channel_;
		int ngroup_,cpg_;
		int pad_;
		entropy_context * ctx_;
		at::Tensor index_mat_, plan_idx_;
		int num_out_;
		int pidx_,mod_;
};
