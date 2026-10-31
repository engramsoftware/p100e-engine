// p100e fast: wrm instantiations for the q4kr layout (matvec_wrm.cuh).
#include "matvec_wrm.cuh"

namespace p100e::fast {

bool wrm_run_q4kr(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s) {
    return wrm_t<dec_q4kr<0>>(vx, a, ncols, geo, kpg, s);
}

} // namespace p100e::fast
