#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

using bf16 = __nv_bfloat16;
using f32 = float;
using u32 = uint32_t;
using u64 = uint64_t;

__device__ __forceinline__ void store_cs_128(void *addr, float4 val) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "r"(__float_as_int(val.x)), "r"(__float_as_int(val.y)), "r"(__float_as_int(val.z)), "r"(__float_as_int(val.w)));
}

template<int NWARPS>
__global__ void rmsnorm_singlepass(
  const bf16 *__restrict__ input,
  const bf16 *__restrict__ weight,
  bf16 *__restrict__ output,
  u32 hidden_dim,
  f32 epsilon
) {
  const u32 row = blockIdx.x;
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31;
  const u32 warp = tid >> 5;
  const u32 nthreads = blockDim.x;

  const bf16 *x = input + (u64)row * hidden_dim;
  bf16 *out = output + (u64)row * hidden_dim;
  const u32 vec_dim = hidden_dim >> 3;
  f32 local_vals[8];
  f32 sum_sq = 0.0f;

  if (tid < vec_dim) {
    float4 v = *reinterpret_cast<const float4 *>(x + tid * 8);
    const bf16 *vals = reinterpret_cast<const bf16 *>(&v);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      local_vals[j] = __bfloat162float(vals[j]);
      sum_sq += local_vals[j] * local_vals[j];
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
  }

  __shared__ f32 warp_sums[32];
  if (lane == 0) {
    warp_sums[warp] = sum_sq;
  }
  __syncthreads();

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
  __syncthreads();

  if (tid < vec_dim) {
    f32 inv_rms = rsqrtf(warp_sums[0] / f32(hidden_dim) + epsilon);
    float4 wv = *reinterpret_cast<const float4 *>(weight + tid * 8);
    const bf16 *wvals = reinterpret_cast<const bf16*>(&wv);

    bf16 result[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      f32 fw = __bfloat162float(wvals[j]);
      result[j] = __float2bfloat16(local_vals[j] * inv_rms * fw);
    }
    store_cs_128(out + tid * 8, *reinterpret_cast<float4 *>(result));
  }
}

template<int NWARPS, int VECS_PER_THREAD>
__global__ void rmsnorm_multipass(
  const bf16 *__restrict__ input,
  const bf16 *__restrict__ weight,
  bf16 *__restrict__ output,
  u32 hidden_dim,
  f32 epsilon
) {
  const u32 row = blockIdx.x;
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31;
  const u32 warp = tid >> 5;
  const u32 nthreads = blockDim.x;

  const bf16 *x = input + (u64)row * hidden_dim;
  bf16 *out = output + (u64)row * hidden_dim;
  const u32 vec_dim = hidden_dim >> 3;
  f32 local_data[VECS_PER_THREAD * 8];
  f32 sum_sq = 0.0f;

  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    u32 idx = tid + v * nthreads;
    if (idx < vec_dim) {
      float4 ld = *reinterpret_cast<const float4 *>(x + idx * 8);
      const bf16 *vals = reinterpret_cast<const bf16*>(&ld);
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        f32 fv = __bfloat162float(vals[j]);
        local_data[v * 8 + j] = fv;
        sum_sq += fv * fv;
      }
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
  }

  __shared__ f32 warp_sums[32];
  if (lane == 0) {
    warp_sums[warp] = sum_sq;
  }
  __syncthreads();

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
  __syncthreads();

  f32 inv_rms = rsqrtf(warp_sums[0] / f32(hidden_dim) + epsilon);

  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    u32 idx = tid + v * nthreads;
    if (idx < vec_dim) {
      float4 wv = *reinterpret_cast<const float4 *>(weight + idx * 8);
      const bf16 *wvals = reinterpret_cast<const bf16*>(&wv);

      bf16 result[8];
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        f32 fw = __bfloat162float(wvals[j]);
        result[j] = __float2bfloat16(local_data[v * 8 + j] * inv_rms * fw);
      }
      store_cs_128(out + idx * 8, *reinterpret_cast<float4 *>(result));
    }
  }
}

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps) {
  TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
  TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16);
  TORCH_CHECK(input.dim() == 2);

  auto rows = input.size(0);
  auto hidden_dim = input.size(1);
  TORCH_CHECK(hidden_dim % 8 == 0, "hidden_dim must be divisible by 8");

  auto output = torch::empty_like(input);
  auto stream = c10::cuda::getCurrentCUDAStream();
  u32 vec_dim = hidden_dim / 8;

  if (vec_dim <= 1024) {
    u32 nthreads = ((vec_dim + 31) / 32) * 32;
    u32 nwarps = nthreads / 32;
    dim3 grid(rows);
    dim3 block(nthreads);

    switch (nwarps) {
      case 1: rmsnorm_singlepass<1><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 2: rmsnorm_singlepass<2><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 4: rmsnorm_singlepass<4><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 8: rmsnorm_singlepass<8><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 16: rmsnorm_singlepass<16><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 20: rmsnorm_singlepass<20><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 32: rmsnorm_singlepass<32><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      default: rmsnorm_singlepass<20><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
    }
  }
  else {
    u32 nthreads = 1024;
    u32 vecs_per_thread = (vec_dim + nthreads - 1) / nthreads;
    dim3 grid(rows);
    dim3 block(nthreads);

    switch (vecs_per_thread) {
      case 1: rmsnorm_multipass<32, 1><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 2: rmsnorm_multipass<32, 2><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      case 4: rmsnorm_multipass<32, 4><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
      default: rmsnorm_multipass<32, 2><<<grid, block, 0, stream>>>((const bf16*)input.data_ptr(), (const bf16*)weight.data_ptr(), (bf16*)output.data_ptr(), hidden_dim, (f32)eps); break;
    }
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "rmsnorm forward (bf16)");
}
