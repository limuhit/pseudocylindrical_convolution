#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "entropy_context.hpp"
#include "string2class.hpp"
class d_extract_opt2: public base_opt{
	public:
		d_extract_opt2(int npart, int nchannel, bool label, std::string ctx_addr, int device = 0, bool timeit=false){
			npart_ = npart;
			nchannel_ = nchannel;
			label_ = label;
			ctx_ = FromString(ctx_addr);
			base_opt_init(device,timeit);
		}
		~d_extract_opt2(){}
		void init();
		void restart(){pidx_=0;}
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  forward_batch_cuda(at::Tensor  bottom_data);
		int npart_;
		int nchannel_;
		bool label_;
		int cpn_,pidx_,mod_;
		at::Tensor top_num_, index_, start_idx_;
		entropy_context * ctx_;
};
