// p100e fast: wrm instantiations for the q6r3 layout (matvec_wrm.cuh).
#include "matvec_wrm.cuh"

namespace p100e::fast {

bool wrm_run_q6r3(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s) {
    if (a.row_bytes % 16 == 0) return wrm_t<dec_q6r3<0, 16>>(vx, a, ncols, geo, kpg, s);
    if (a.row_bytes % 8 == 0)  return wrm_t<dec_q6r3<0, 8>>(vx, a, ncols, geo, kpg, s);
    return false;
}

} // namespace p100e::fast
