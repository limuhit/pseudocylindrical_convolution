#include "main.hpp"
#include <torch/extension.h>
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    py::class_<projects_opt>(m,"ProjectsOp")
        .def(py::init<int, int , std::vector<float>, std::vector<float>, float, bool, int, bool>())
        .def("to", &projects_opt::to)
        .def("forward", &projects_opt::forward_cuda)
        .def("backward", &projects_opt::backward_cuda);

    py::class_<dtow_opt>(m,"DtowOp")
        .def(py::init<int, bool, int, bool>())
        .def("to", &dtow_opt::to)
        .def("forward", &dtow_opt::forward_cuda)
        .def("backward", &dtow_opt::backward_cuda);

    py::class_<context_reshape_opt>(m,"ContextReshapeOp")
        .def(py::init<int, int, bool>())
        .def("to", &context_reshape_opt::to)
        .def("forward", &context_reshape_opt::forward_cuda)
        .def("backward", &context_reshape_opt::backward_cuda);

    py::class_<entropy_gmm_opt>(m,"EntropyGmmOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &entropy_gmm_opt::to)
        .def("forward", &entropy_gmm_opt::forward_cuda)
        .def("backward", &entropy_gmm_opt::backward_cuda);

    py::class_<mask_constrain_opt>(m,"MaskConstrainOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &mask_constrain_opt::to)
        .def("forward", &mask_constrain_opt::forward_cuda)
        .def("backward", &mask_constrain_opt::backward_cuda);


    py::class_<sphere_slice_opt>(m,"SphereSliceOp")
        .def(py::init<int, int, int, std::vector<float>, int, bool>())
        .def("to", &sphere_slice_opt::to)
        .def("forward", &sphere_slice_opt::forward_cuda)
        .def("backward", &sphere_slice_opt::backward_cuda);

    py::class_<sphere_uslice_opt>(m,"SphereUsliceOp")
        .def(py::init<int, int, int, std::vector<float>, int, bool>())
        .def("to", &sphere_uslice_opt::to)
        .def("forward", &sphere_uslice_opt::forward_cuda)
        .def("backward", &sphere_uslice_opt::backward_cuda);

    py::class_<entropy_gmm_table_opt>(m,"EntropyGmmTableOp")
        .def(py::init<int, float, int, float, float, int, bool>())
        .def("to", &entropy_gmm_table_opt::to)
        .def("forward", &entropy_gmm_table_opt::forward_cuda)
        .def("forward_batch", &entropy_gmm_table_opt::forward_batch_cuda);

    py::class_<entropy_context_shell>(m,"EntropyContextOp")
        .def(py::init<int, int, std::vector<float>, int, bool>())
        .def("to", &entropy_context_shell::to)
        .def("start_context", &entropy_context_shell::start_context)
        .def("addr",&entropy_context_shell::get_pointer);

    py::class_<entropy_ctx_pad_run2_opt>(m,"EntropyCtxPadRun2Op")
        .def(py::init<int, int, int, bool, std::string, int, bool>())
        .def("to", &entropy_ctx_pad_run2_opt::to)
        .def("restart",&entropy_ctx_pad_run2_opt::restart)
        .def("forward", &entropy_ctx_pad_run2_opt::forward_cuda)
        .def("backward", &entropy_ctx_pad_run2_opt::backward_cuda);
    
    py::class_<d_extract_opt2>(m,"DExtract2Op")
        .def(py::init<int, int, bool, std::string, int, bool>())
        .def("to", &d_extract_opt2::to)
        .def("restart",&d_extract_opt2::restart)
        .def("forward", &d_extract_opt2::forward_cuda)
        .def("forward_batch", &d_extract_opt2::forward_batch_cuda);

    py::class_<d_input_opt2>(m,"DInput2Op")
        .def(py::init<int, int, int, float, int, std::string, int, bool>())
        .def("to", &d_input_opt2::to)
        .def("restart", &d_input_opt2::restart)
        .def("forward", &d_input_opt2::forward_cuda);

    py::class_<entropy_conv_opt2>(m,"EntropyConv2Op")
        .def(py::init<int, int, int, int, int, int, int, int, std::string, int, bool>())
        .def("to", &entropy_conv_opt2::to)
        .def("restart",&entropy_conv_opt2::restart)
        .def("forward", &entropy_conv_opt2::forward_cuda)
        .def("forward_act", &entropy_conv_opt2::forward_act_cuda)
        .def("forward_batch", &entropy_conv_opt2::forward_cuda_batch)
        .def("forward_act_batch", &entropy_conv_opt2::forward_act_cuda_batch);

    py::class_<pseudo_context_shell>(m,"PseudoContextOp")
        .def(py::init<int, int, std::vector<float>, int, bool>())
        .def("to", &pseudo_context_shell::to)
        .def("start_context", &pseudo_context_shell::start_context)
        .def("addr",&pseudo_context_shell::get_pointer)
        .def("produce_fill_param",&pseudo_context_shell::produce_fill_param);

    py::class_<pseudo_pad_opt>(m,"PseudoPadOp")
        .def(py::init<int, int, std::string, int, bool>())
        .def("to", &pseudo_pad_opt::to)
        .def("forward", &pseudo_pad_opt::forward_cuda)
        .def("backward", &pseudo_pad_opt::backward_cuda);

    py::class_<pseudo_fill_opt>(m,"PseudoFillOp")
        .def(py::init<int, int, int, int, std::string, int, int, bool>())
        .def("to", &pseudo_fill_opt::to)
        .def("forward", &pseudo_fill_opt::forward_cuda)
        .def("backward", &pseudo_fill_opt::backward_cuda);

    py::class_<pseudo_entropy_context_shell>(m,"PseudoEntropyContextOp")
        .def(py::init<int, int, int, std::vector<float>, int, bool>())
        .def("to", &pseudo_entropy_context_shell::to)
        .def("start_context", &pseudo_entropy_context_shell::start_context)
        .def("addr",&pseudo_entropy_context_shell::get_pointer);

    py::class_<pseudo_entropy_pad_opt>(m,"PseudoEntropyPadOp")
        .def(py::init<int, int, std::string, int, bool>())
        .def("to", &pseudo_entropy_pad_opt::to)
        .def("forward", &pseudo_entropy_pad_opt::forward_cuda)
        .def("backward", &pseudo_entropy_pad_opt::backward_cuda);

    py::class_<pseudo_quant_opt>(m,"PseudoQuantOp")
        .def(py::init<int, int, int, float, int, int, float, std::string, int, bool>())
        .def("to", &pseudo_quant_opt::to)
        .def("forward", &pseudo_quant_opt::quant_forward_cuda)
        .def("backward", &pseudo_quant_opt::quant_backward_cuda);
    
    py::class_<pseudo_dquant_opt>(m,"PseudoDQuantOp")
        .def(py::init<int, int, int, std::string,int, bool>())
        .def("to", &pseudo_dquant_opt::to)
        .def("forward", &pseudo_dquant_opt::forward_cuda);

    py::class_<entropy_add_opt>(m,"EntropyAddOp")
        .def(py::init<int, int, int, int, std::string, int, bool>())
        .def("restart",&entropy_add_opt::restart)
        .def("to", &entropy_add_opt::to)
        .def("forward", &entropy_add_opt::forward_cuda);
};