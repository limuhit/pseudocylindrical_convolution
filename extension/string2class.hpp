#pragma once
#include "entropy_context.hpp"
#include "pseudo_context.hpp"
#include "pseudo_entropy_context.hpp"
entropy_context * FromString ( const std::string &Text );
pseudo_context_opt * FromStringPseudo ( const std::string &Text );
pseudo_entropy_context_opt * FromStringPseudoEntropy ( const std::string &Text );
