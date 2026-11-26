#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using bf16 = __nv_bfloat16;
using f32 = float;
using u32 = uint32_t;
using u64 = uint64_t;

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

struct alignas(8) RawEvent {
  u64 clock;
  u32 meta;
};

static constexpr u32 MAX_PER_WARP = 64;

__device__ __forceinline__ u64 globaltimer_ns() {
  u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__device__ __forceinline__ u64 clk() {
  u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t;
}

__device__ __forceinline__ void store_cs_128(void *addr, float4 val) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "r"(__float_as_int(val.x)), "r"(__float_as_int(val.y)), "r"(__float_as_int(val.z)), "r"(__float_as_int(val.w)));
}

template<bool T, int NWARPS>
__global__ void rmsnorm_fwd(
  const bf16 *__restrict__ input,
  const bf16 *__restrict__ weight,
  bf16 *__restrict__ output,
  u32 hidden_dim,
  f32 epsilon,
  RawEvent *__restrict__ trace_buf,
  u32 *__restrict__ trace_counts
) {
  const u32 row = blockIdx.x;
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31;
  const u32 warp = tid >> 5;

  extern __shared__ char smem_raw[];
  f32 *warp_sums = reinterpret_cast<f32 *>(smem_raw);

  // Shared block-wide anchor: globaltimer + clock64 baseline, written by thread 0
  volatile u64 *block_anchor_ns  = reinterpret_cast<volatile u64 *>(smem_raw + 128);
  volatile u64 *block_anchor_clk = reinterpret_cast<volatile u64 *>(smem_raw + 136);
  RawEvent *smem_trace = reinterpret_cast<RawEvent *>(smem_raw + 144);

  if constexpr (T) {
    if (tid == 0) {
      *block_anchor_clk = clk();
      *block_anchor_ns = globaltimer_ns();
    }
  }
  __syncthreads();

  const bool do_trace = T && (lane == 0);
  u32 ei = 0;
  RawEvent *my_smem = nullptr;
  if constexpr (T) {
    my_smem = smem_trace + warp * MAX_PER_WARP;
    if (lane == 0) {
      my_smem[ei++] = {*block_anchor_ns, 0xFFFFFFFF};
      my_smem[ei++] = {*block_anchor_clk, 0xFFFFFFFE};
    }
  }

  #define MARK() do { if (__builtin_expect(do_trace, 0)) { my_smem[ei++] = {clk(), ei}; } } while(0)

  const bf16 *x = input + (u64)row * hidden_dim;
  bf16 *out = output + (u64)row * hidden_dim;
  const u32 vec_dim = hidden_dim >> 3;
  f32 local_vals[8];
  f32 sum_sq = 0.0f;

  // T0: start of load+sumsq
  MARK();
  if (tid < vec_dim) {
    float4 v = *reinterpret_cast<const float4 *>(x + tid * 8);
    const bf16 *vals = reinterpret_cast<const bf16 *>(&v);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      local_vals[j] = __bfloat162float(vals[j]);
      sum_sq += local_vals[j] * local_vals[j];
    }
  }
  __syncwarp();

  // T1: start of warp reduce
  MARK();
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
  }

  // T2: start of block reduce (write to smem)
  MARK();
  if (lane == 0) {
    warp_sums[warp] = sum_sq;
  }

  // T3: start of barrier 1
  MARK();
  __syncthreads();

  // T4: start of cross-warp reduce (warp 0 only does work)
  MARK();
  if (warp == 0) {
    sum_sq = (lane < NWARPS) ? warp_sums[lane] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (lane == 0) {
      warp_sums[0] = sum_sq;
    }
  }

  // T5: start of barrier 2
  MARK();
  __syncthreads();

  // T6: start of normalize
  MARK();
  f32 inv_rms = 0.0f;
  bf16 result[8];
  if (tid < vec_dim) {
    inv_rms = rsqrtf(warp_sums[0] / f32(hidden_dim) + epsilon);
    float4 wv = *reinterpret_cast<const float4 *>(weight + tid * 8);
    const bf16 *wvals = reinterpret_cast<const bf16 *>(&wv);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
      f32 fw = __bfloat162float(wvals[j]);
      result[j] = __float2bfloat16(local_vals[j] * inv_rms * fw);
    }
  }
  __syncwarp();

  // T7: start of store
  MARK();
  if (tid < vec_dim) {
    store_cs_128(out + tid * 8, *reinterpret_cast<float4 *>(result));
  }
  __syncwarp();

  // T8: end
  MARK();

  if constexpr (T) {
    if (lane == 0) {
      RawEvent *dst = trace_buf + (u64)(row * NWARPS + warp) * MAX_PER_WARP;
      for (u32 i = 0; i < ei; i++) {
        dst[i] = my_smem[i];
      }
      trace_counts[row * NWARPS + warp] = ei;
    }
  }

  #undef MARK
}

template<int NWARPS>
void launch(bool trace, u32 rows, u32 nthreads, u32 hidden_dim, f32 eps, bf16 *d_in, bf16 *d_w, bf16 *d_out, RawEvent *d_trace, u32 *d_counts) {
  dim3 grid(rows), block(nthreads);
  u32 smem = 144 + (trace ? NWARPS * MAX_PER_WARP * sizeof(RawEvent) : 0);
  if (trace) {
    rmsnorm_fwd<true, NWARPS><<<grid, block, smem>>>(d_in, d_w, d_out, hidden_dim, eps, d_trace, d_counts);
  }
  else {
    rmsnorm_fwd<false, NWARPS><<<grid, block, smem>>>(d_in, d_w, d_out, hidden_dim, eps, nullptr, nullptr);
  }
}

void dispatch(bool trace, u32 nwarps, u32 rows, u32 nthreads, u32 hidden_dim, f32 eps, bf16 *d_in, bf16 *d_w, bf16 *d_out, RawEvent *d_trace, u32 *d_counts) {
  switch (nwarps) {
    case 1: launch<1>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
    case 2: launch<2>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
    case 4: launch<4>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
    case 8: launch<8>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
    case 16: launch<16>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
    case 20: launch<20>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
    case 32: launch<32>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
    default: launch<20>(trace, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts); break;
  }
}

void run_benchmark(u32 rows, u32 hidden_dim, int warmup, int iters) {
  const u32 vec_dim = hidden_dim / 8;
  u32 nthreads = ((vec_dim + 31) / 32) * 32;
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const u32 nwarps = nthreads / 32;

  size_t sz_in = (size_t)rows * hidden_dim * sizeof(bf16);
  size_t sz_w = (size_t)hidden_dim * sizeof(bf16);

  bf16 *d_in, *d_w, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, sz_in));
  CHECK_CUDA(cudaMalloc(&d_w, sz_w));
  CHECK_CUDA(cudaMalloc(&d_out, sz_in));

  std::vector<bf16> h_in(rows * hidden_dim), h_w(hidden_dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : h_in) v = __float2bfloat16(dist(rng));
  for (auto &v : h_w) v = __float2bfloat16(0.5f + 0.5f * dist(rng));
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), sz_in, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), sz_w, cudaMemcpyHostToDevice));

  f32 eps = 1e-6f;

  for (int i = 0; i < warmup; i++) {
    dispatch(false, nwarps, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, nullptr, nullptr);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));
  CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaEventRecord(t0));
  for (int i = 0; i < iters; i++) {
    dispatch(false, nwarps, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, nullptr, nullptr);
  }
  CHECK_CUDA(cudaEventRecord(t1));
  CHECK_CUDA(cudaEventSynchronize(t1));

  float ms;
  CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
  float us = ms * 1000.0f / iters;

  double bytes = (double)rows * hidden_dim * 2 * 2 + (double)hidden_dim * 2;
  double gbps = bytes / (us * 1e-6) / 1e9;

  printf("  rows=%5u  hidden=%5u  warps=%2u | %7.2f us  %7.1f GB/s\n", rows, hidden_dim, nwarps, us, gbps);

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaEventDestroy(t0));
  CHECK_CUDA(cudaEventDestroy(t1));
}

void run_trace_export(u32 rows, u32 hidden_dim, const char *out_path) {
  const u32 vec_dim = hidden_dim / 8;
  u32 nthreads = ((vec_dim + 31) / 32) * 32;
  if (nthreads > 1024) nthreads = 1024;
  const u32 nwarps = nthreads / 32;

  size_t sz_in = (size_t)rows * hidden_dim * sizeof(bf16);
  size_t sz_w = (size_t)hidden_dim * sizeof(bf16);

  bf16 *d_in, *d_w, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, sz_in));
  CHECK_CUDA(cudaMalloc(&d_w, sz_w));
  CHECK_CUDA(cudaMalloc(&d_out, sz_in));

  std::vector<bf16> h_in(rows * hidden_dim), h_w(hidden_dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : h_in) v = __float2bfloat16(dist(rng));
  for (auto &v : h_w) v = __float2bfloat16(0.5f + 0.5f * dist(rng));
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), sz_in, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), sz_w, cudaMemcpyHostToDevice));

  u32 total_warps = rows * nwarps;
  RawEvent *d_trace;
  u32 *d_counts;
  CHECK_CUDA(cudaMalloc(&d_trace, total_warps * MAX_PER_WARP * sizeof(RawEvent)));
  CHECK_CUDA(cudaMalloc(&d_counts, total_warps * sizeof(u32)));
  CHECK_CUDA(cudaMemset(d_counts, 0, total_warps * sizeof(u32)));

  f32 eps = 1e-6f;

  for (int i = 0; i < warmup; i++) {
    dispatch(false, nwarps, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, nullptr, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  void *d_flush;
  size_t flush_sz = 48 * 1024 * 1024;
  CHECK_CUDA(cudaMalloc(&d_flush, flush_sz));
  CHECK_CUDA(cudaMemset(d_flush, 0, flush_sz));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(d_flush));

  CHECK_CUDA(cudaMemset(d_counts, 0, total_warps * sizeof(u32)));
  dispatch(true, nwarps, rows, nthreads, hidden_dim, eps, d_in, d_w, d_out, d_trace, d_counts);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<RawEvent> h_trace(total_warps * MAX_PER_WARP);
  std::vector<u32> h_counts(total_warps);
  CHECK_CUDA(cudaMemcpy(h_trace.data(), d_trace, total_warps * MAX_PER_WARP * sizeof(RawEvent), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, total_warps * sizeof(u32), cudaMemcpyDeviceToHost));

  FILE *fp = fopen(out_path, "wb");
  if (!fp) { fprintf(stderr, "cannot open %s\n", out_path); exit(1); }

  fwrite(&rows, sizeof(u32), 1, fp);
  fwrite(&hidden_dim, sizeof(u32), 1, fp);
  fwrite(&nwarps, sizeof(u32), 1, fp);
  u32 mpw = MAX_PER_WARP;
  fwrite(&mpw, sizeof(u32), 1, fp);
  fwrite(h_counts.data(), sizeof(u32), total_warps, fp);
  fwrite(h_trace.data(), sizeof(RawEvent), total_warps * MAX_PER_WARP, fp);
  fclose(fp);

  u32 max_ev = *std::max_element(h_counts.begin(), h_counts.end());
  u32 min_ev = *std::min_element(h_counts.begin(), h_counts.end());
  printf("  exported %u warp traces to %s (events/warp: %u-%u)\n", total_warps, out_path, min_ev, max_ev);

  CHECK_CUDA(cudaFree(d_trace));
  CHECK_CUDA(cudaFree(d_counts));
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_out));
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "trace") == 0) {
    u32 rows = 256, hidden = 5120;
    const char *path = "trace.bin";
    if (argc > 2) rows = atoi(argv[2]);
    if (argc > 3) hidden = atoi(argv[3]);
    if (argc > 4) path = argv[4];
    printf("trace export: rows=%u hidden=%u\n", rows, hidden);
    run_trace_export(rows, hidden, path);
  }
  else {
    printf("benchmark (warmup=20, iters=100):\n");
    u32 shapes[][2] = {{256, 5120}, {1024, 5120}, {4096, 5120}, {16384, 5120}, {32768, 5120}};
    for (auto &s : shapes) {
      run_benchmark(s[0], s[1], 20, 100);
    }
  }
  return 0;
}
