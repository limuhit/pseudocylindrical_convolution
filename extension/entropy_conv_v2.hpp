#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "entropy_context.hpp"
#include "string2class.hpp"
class entropy_conv_opt2: public base_opt{
	public:
		entropy_conv_opt2(int npart, int channel, int ngroup, int nout, int kernel_size, int constrain, int pad_in, int pad_out, std::string ctx_addr, int device = 0, bool timeit=false){
			npart_ = npart;
			channel_ = channel;
			ngroup_ = ngroup;
			nout_ = nout;
			kernel_size_ = kernel_size;
			constrain_ = constrain;
			pad_in_ = pad_in;
			pad_out_ = pad_out;
			group_in_ = channel / ngroup;
			group_out_ = nout / ngroup;
			skernel_ = kernel_size*kernel_size;
			ctx_ = FromString(ctx_addr);
			base_opt_init(device,timeit);
		}
		~entropy_conv_opt2(){}
		void init();
		void restart(){pidx_=0;}
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias);
		std::vector<at::Tensor>  forward_act_cuda(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act);
		std::vector<at::Tensor>  forward_cuda_batch(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias);
		std::vector<at::Tensor>  forward_act_cuda_batch(at::Tensor bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act);
		int channel_, npart_;
		int ngroup_;
		int nout_;
		int num_out_;
		int pad_in_, pad_out_;
		int kernel_size_;
		int group_in_, group_out_;
		int constrain_;
		int pidx_,mod_;
		int skernel_;
		entropy_context * ctx_;
		at::Tensor index_mat_, plan_idx_;
};
