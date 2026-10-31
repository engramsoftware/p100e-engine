// ggml type table (block sizes come from the vendored ggml-common.h structs, so they can't drift)
// plus the glue the vendored convert.cu expects (device info, abort).
#include "gguf.h"
#include "../third_party/ggml/common.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace p100e {

static const qtype_info k_unknown = {"unknown", 0, 0, false};

const qtype_info & get_type_info(uint32_t type) {
    static const qtype_info table[GGML_TYPE_COUNT] = {
        /*  0 F32     */ {"f32",     1,       4,                       true},
        /*  1 F16     */ {"f16",     1,       2,                       true},
        /*  2 Q4_0    */ {"q4_0",    QK4_0,   sizeof(block_q4_0),      true},
        /*  3 Q4_1    */ {"q4_1",    QK4_1,   sizeof(block_q4_1),      true},
        /*  4         */ {"q4_2(removed)", 0, 0, false},
        /*  5         */ {"q4_3(removed)", 0, 0, false},
        /*  6 Q5_0    */ {"q5_0",    QK5_0,   sizeof(block_q5_0),      true},
        /*  7 Q5_1    */ {"q5_1",    QK5_1,   sizeof(block_q5_1),      true},
        /*  8 Q8_0    */ {"q8_0",    QK8_0,   sizeof(block_q8_0),      true},
        /*  9 Q8_1    */ {"q8_1",    QK8_1,   sizeof(block_q8_1),      false},
        /* 10 Q2_K    */ {"q2_K",    QK_K,    sizeof(block_q2_K),      true},
        /* 11 Q3_K    */ {"q3_K",    QK_K,    sizeof(block_q3_K),      true},
        /* 12 Q4_K    */ {"q4_K",    QK_K,    sizeof(block_q4_K),      true},
        /* 13 Q5_K    */ {"q5_K",    QK_K,    sizeof(block_q5_K),      true},
        /* 14 Q6_K    */ {"q6_K",    QK_K,    sizeof(block_q6_K),      true},
        /* 15 Q8_K    */ {"q8_K",    QK_K,    sizeof(block_q8_K),      false},
        /* 16 IQ2_XXS */ {"iq2_xxs", QK_K,    sizeof(block_iq2_xxs),   true},
        /* 17 IQ2_XS  */ {"iq2_xs",  QK_K,    sizeof(block_iq2_xs),    true},
        /* 18 IQ3_XXS */ {"iq3_xxs", QK_K,    sizeof(block_iq3_xxs),   true},
        /* 19 IQ1_S   */ {"iq1_s",   QK_K,    sizeof(block_iq1_s),     true},
        /* 20 IQ4_NL  */ {"iq4_nl",  QK4_NL,  sizeof(block_iq4_nl),    true},
        /* 21 IQ3_S   */ {"iq3_s",   QK_K,    sizeof(block_iq3_s),     true},
        /* 22 IQ2_S   */ {"iq2_s",   QK_K,    sizeof(block_iq2_s),     true},
        /* 23 IQ4_XS  */ {"iq4_xs",  QK_K,    sizeof(block_iq4_xs),    true},
        /* 24 I8      */ {"i8",      1,       1,                       false},
        /* 25 I16     */ {"i16",     1,       2,                       false},
        /* 26 I32     */ {"i32",     1,       4,                       false},
        /* 27 I64     */ {"i64",     1,       8,                       false},
        /* 28 F64     */ {"f64",     1,       8,                       false},
        /* 29 IQ1_M   */ {"iq1_m",   QK_K,    sizeof(block_iq1_m),     true},
        /* 30 BF16    */ {"bf16",    1,       2,                       true},
        /* 31         */ {"q4_0_4_4(removed)", 0, 0, false},
        /* 32         */ {"q4_0_4_8(removed)", 0, 0, false},
        /* 33         */ {"q4_0_8_8(removed)", 0, 0, false},
        /* 34 TQ1_0   */ {"tq1_0",   QK_K,    sizeof(block_tq1_0),     false},
        /* 35 TQ2_0   */ {"tq2_0",   QK_K,    sizeof(block_tq2_0),     false},
        /* 36         */ {"iq4_nl_4_4(removed)", 0, 0, false},
        /* 37         */ {"iq4_nl_4_8(removed)", 0, 0, false},
        /* 38         */ {"iq4_nl_8_8(removed)", 0, 0, false},
        /* 39 MXFP4   */ {"mxfp4",   QK_MXFP4, sizeof(block_mxfp4),    true},
        /* 40 NVFP4   */ {"nvfp4",   QK_NVFP4, sizeof(block_nvfp4),    true},
        /* 41 Q1_0    */ {"q1_0",    QK1_0,   sizeof(block_q1_0),      true},
        /* 42 Q2_0    */ {"q2_0",    QK2_0,   sizeof(block_q2_0),      true},
    };
    return type < GGML_TYPE_COUNT ? table[type] : k_unknown;
}

} // namespace p100e

int ggml_cuda_get_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info i{};
        int n = 0;
        cudaGetDeviceCount(&n);
        for (int d = 0; d < n && d < 16; ++d) {
            cudaDeviceProp p;
            cudaGetDeviceProperties(&p, d);
            i.devices[d].cc = 100 * p.major + 10 * p.minor;
        }
        return i;
    }();
    return info;
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}
