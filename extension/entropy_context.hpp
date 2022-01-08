#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include <map>
#include "base_opt.hpp"
#include <sstream> //for std::stringstream 
#include <string>  //for std::string


class entropy_context: public base_opt{
	public:
		entropy_context(int npart, int rt,  std::vector<float> weight, int device = 0, bool timeit=false){
			npart_ = npart;
			rt_ = rt;
			weight_ = new float[npart_];
			for(int i = 0; i<weight.size();i++)
				weight_[i] = weight[i];
			base_opt_init(device,timeit);
		}
		~entropy_context(){delete weight_;}

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
                index_.clear();
                start_idx_.clear();
                pad_idx_.clear();
                width_dict_.clear();
                pad_channel_dict_.clear();
            }
            data_width_ = width;
        }
        std::vector<at::Tensor> produce_param(int channel, int height, int width, int pad);
        at::Tensor produce_param_fill(int height, int width);
        std::vector<at::Tensor> produce_param_group(int height, int width);
        std::map<int, at::Tensor> param_, param2_, hindex_, hindex2_, index_, start_idx_, pad_idx_;
        std::map<int,int> pad_channel_dict_, width_dict_;
		int npart_, pad_;
		int rt_, stride_entropy_;
		float* weight_;
        int data_width_=-1;
        int cp_;
};

class entropy_context_shell{
    public:
        entropy_context_shell(int npart, int rt, std::vector<float> weight, int device = 0, bool timeit=false){
            ctx = new entropy_context(npart,rt,weight,device,timeit);
            const void * address = static_cast<const void*>(ctx);
            std::stringstream ss;
            ss << address;  
            addr = ss.str(); 
        }
        ~entropy_context_shell(){delete ctx;}
        void to(int device){
            ctx->to(device);
        }
        void start_context(int width){
            ctx->start_context(width);
        }
        std::string get_pointer(){
            return addr;
        }
        entropy_context * ctx;
        std::string addr;
        

};


	