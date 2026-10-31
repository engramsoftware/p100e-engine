// p100e fast: wrm instantiations for the q6r3 layout with the dec_q6h tail F = 1 (P100E_MV_Q6_H16 = 1; the default is 2;
// see matvec_dec_q6h.cuh). Every T = 1..8 runs through here when the switch is on (matvec.cu), so the result of a row
// does not depend on T. One translation unit per F (ptxas host memory, as matvec_wrm_<layout>.cu).
#include "matvec_wrm.cuh"
#include "matvec_dec_q6h.cuh"

namespace p100e::fast {

bool wrm_run_q6h1(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s) {
    if (a.row_bytes % 16 == 0) return wrm_t_any<dec_q6h<0, 16, 1>>(vx, a, ncols, geo, kpg, s);
    if (a.row_bytes % 8 == 0)  return wrm_t_any<dec_q6h<0, 8, 1>>(vx, a, ncols, geo, kpg, s);
    return false;
}

} // namespace p100e::fast
