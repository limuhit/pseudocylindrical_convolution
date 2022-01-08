#include "string2class.hpp"
entropy_context * FromString ( const std::string &Text )
{
    std::stringstream ss(Text);
    void * result;
    ss >> result;
    return reinterpret_cast<entropy_context *>(result);
}
pseudo_context_opt * FromStringPseudo ( const std::string &Text )
{
    std::stringstream ss(Text);
    void * result;
    ss >> result;
    return reinterpret_cast<pseudo_context_opt *>(result);
}
pseudo_entropy_context_opt * FromStringPseudoEntropy ( const std::string &Text )
{
    std::stringstream ss(Text);
    void * result;
    ss >> result;
    return reinterpret_cast<pseudo_entropy_context_opt *>(result);
}