#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// pf-xchg, GGML_PF_AR_F16=1: same as ggml_cuda_ar_allreduce, but when f16_wire is true and the reduction takes the
// copy-engine path (F32, no BF16 round trip, no GGML_CUDA_AR_Q8 block), each GPU's partial crosses PCIe as f16.
// The caller sets f16_wire only for partials it has established to be f16-exact (ggml-cuda.cu), so the sum is
// bit-identical to the F32 wire.  f16_wire = false is exactly ggml_cuda_ar_allreduce.
bool ggml_cuda_ar_allreduce_ex(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors,
    bool                    f16_wire);

// pf-xchg: true when an F32 reduction of nbytes would take this pipeline's copy-engine path.
bool ggml_cuda_ar_pipeline_copy_engine_for(const ggml_cuda_ar_pipeline * p, size_t nbytes);

// GGML_CUDA_AR_PASCAL=1 only: aborts with a clear message if an in-kernel
// AllReduce spin timed out (its result is invalid).  No-op (one atomic load of
// nullptr) when no Pascal pipeline exists.
void ggml_cuda_ar_pascal_check_status();

// True when the pipeline runs the GGML_CUDA_AR_PASCAL=1 path (false for nullptr).
bool ggml_cuda_ar_pipeline_is_pascal(const ggml_cuda_ar_pipeline * p);

