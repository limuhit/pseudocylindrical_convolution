#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include <map>
#include <sstream> //for std::stringstream 
#include <string>  //for std::string
class pseudo_entropy_context_opt: public base_opt{
	public:
		pseudo_entropy_context_opt(int npart, int rt, int context_version, std::vector<float> weight, int device = 0, bool timeit=false){
			npart_ = npart;
			rt_ = rt;
			weight_ = new float[npart_];
			for(int i = 0; i<weight.size();i++)
				weight_[i] = weight[i];
            context_version_ = context_version;
			base_opt_init(device,timeit);
		}
		~pseudo_entropy_context_opt(){delete weight_;}
		void init();
		bool reshape_hw(int height, int width);
        bool reshape_channel_pad(int channel, int pad);
        void start_context(int width)
        {
            if(width!=data_width_){
                param_.clear();
                param2_.clear();
                hindex_.clear();
                hindex2_.clear();
                stride_inv_.clear();
                width_dict_.clear();
                pad_channel_dict_.clear();
            }
            data_width_ = width;
        }
		std::vector<at::Tensor> produce_param(int channel, int height, int width, int pad);
        at::Tensor produce_param_fill(int height, int width);
        std::map<int, at::Tensor> param_, param2_, hindex_, hindex2_, index_,  stride_inv_;
        std::map<int,int> pad_channel_dict_, width_dict_;
        at::Tensor wbase_;
		int npart_, pad_, rt_;
		float * weight_;
        int data_width_=-1;
        int cp_;
        int context_version_;
};


class pseudo_entropy_context_shell{
    public:
        pseudo_entropy_context_shell(int npart, int rt, int context_version, std::vector<float> weight, int device = 0, bool timeit=false){
            ctx = new pseudo_entropy_context_opt(npart,rt, context_version, weight,device,timeit);
            const void * address = static_cast<const void*>(ctx);
            std::stringstream ss;
            ss << address;  
            addr = ss.str(); 
        }
        ~pseudo_entropy_context_shell(){delete ctx;}
        void to(int device){
            ctx->to(device);
        }
        void start_context(int width){
            ctx->start_context(width);
        }
        std::string get_pointer(){
            return addr;
        }
        pseudo_entropy_context_opt * ctx;
        std::string addr;
};