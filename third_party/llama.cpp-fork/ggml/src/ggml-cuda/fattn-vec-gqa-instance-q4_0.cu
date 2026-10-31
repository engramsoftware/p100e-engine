// Explicit instances of the opt-in GQA vector FlashAttention kernel (GGML_P100_FA_GQA), see fattn-vec-gqa.cuh.
// Kept in the ggml-cuda directory: template-instances/fattn-vec*.cu are only globbed with GGML_CUDA_FA_ALL_QUANTS.

#include "fattn-vec-gqa.cuh"

DECL_FATTN_VEC_GQA_CASE(256, 1, 4, GGML_TYPE_Q4_0, false);
DECL_FATTN_VEC_GQA_CASE(256, 1, 6, GGML_TYPE_Q4_0, false);
DECL_FATTN_VEC_GQA_CASE(256, 1, 8, GGML_TYPE_Q4_0, false);
DECL_FATTN_VEC_GQA_CASE(256, 2, 3, GGML_TYPE_Q4_0, false);
DECL_FATTN_VEC_GQA_CASE(256, 2, 4, GGML_TYPE_Q4_0, false);
DECL_FATTN_VEC_GQA_CASE(256, 4, 2, GGML_TYPE_Q4_0, false);
