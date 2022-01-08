#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "entropy_context.hpp"
#include "string2class.hpp"
class d_input_opt2: public base_opt{
	public:
		d_input_opt2(int nchannel, int npart, int pad, float bias, int replicate, std::string ctx_addr, int device = 0, bool timeit=false){
			channel_ = nchannel;
			npart_ = npart;
			pad_ = pad;
			bias_ = bias;
			rep_ = replicate;
			ctx_ = FromString(ctx_addr);
			base_opt_init(device,timeit);
		}
		~d_input_opt2(){}
		void init();
		void restart(){pidx_=0;}
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		int pidx_, mod_, npart_, pad_;
		at::Tensor index_, start_idx_;
		entropy_context * ctx_;
		float bias_;
		int rep_;
};
