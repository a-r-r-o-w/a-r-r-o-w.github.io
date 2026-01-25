---
{
  "title": "Optimizing diffusion for production-ready speeds - III",
  "code": "https://github.com/a-r-r-o-w/productionizing-diffusion",
  "date": "2026-01-25",
  "tags": ["diffusion", "optimization", "mlsys"]
}
---

Diffusion models have rapidly advanced generative modeling across a wide range of modalities - from images and video to music, 3D objects, and even text generation and world simulations recently. They are now central to state-of-the-art image and video generation, offering high-quality, controllable, and diverse outputs. However, their computational cost remains a bottleneck for real-world deployment. In this series, we explore techniques to optimize diffusion inference for text-to-image and text-to-video generation.

This post is first in a four-part series. We will cover the following topics:
1. How text-to-image diffusion models work and their computational challenges?
2. Standard optimizations for transformer-based diffusion models
3. Going deep: using faster kernels, non-trivial fusions, precomputations
4. Context parallelism
5. Quantization
6. Caching
7. LoRA
8. Training
9. Practice: Wan text-to-video
10. Optimizing inference for uncommon deployment environments using Triton

| Post | Topics covered |
|------|----------------|
| Optimizing diffusion for production-ready speeds - I   | 1, 2 |
| Optimizing diffusion for production-ready speeds - II  | 3, 4 |
| Optimizing diffusion for production-ready speeds - III | 5, 6 |
| Optimizing diffusion for production-ready speeds - IV  | 7, 8, 9, 10 |

The code for the entire series is available at [a-r-r-o-w/productionizing-diffusion](https://github.com/a-r-r-o-w/productionizing-diffusion). For this post, refer to the `post_3` directory. The guides are written to work on Nvidia's A100/H100 or newer GPUs, but the ideas can be adapted to other hardware as well.

## Table of contents

- [CUDA Streams](#cuda-streams)
- [Quantization](#quantization)
  - [fp8 inference](#fp8-inference)
  - [fp8 benchmarks](#fp8-benchmarks)
- [To distill or to cache?](#to-distill-or-to-cache)
- [Caching in diffusion models](#caching-in-diffusion-models)
- [Benchmarks](#benchmarks)
- [Additional reading](#additional-reading)
- [Citation](#citation)

## CUDA Streams

CUDA streams are a powerful feature available on Nvidia GPUs that allow for concurrent execution of multiple operations (kernel executions, memory transfers from CPU-to-GPU or GPU-to-CPU). A stream can be thought of as a sequentially executing task queue. Whenever executing a pytorch or cuda program, a "default" stream is utilized, on which all your code runs sequentially. GPUs are inherently parallel devices, so using a single task queue is not the most efficient way to utilize the hardware. Multiple non-default streams can be created to parallelize the execution of different kernel/memory operations and gain significant speedups!

Note: Other modern GPUs also have similar features, such as AMD's HIP streams.

By default, operations queued into a stream are performed asynchronously, meaning that the CPU can continue executing other code while the GPU is processing the tasks in the stream. When utilizing multiple streams, each stream executes its tasks independently of other streams (in parallel when available GPU resources permit overlaps), but to ensure the correctness, we, as the programmers, need to make sure that the operations in different streams do not depend on each other. If data is not independent between streams, we need to _synchronize_ them to ensure the correct order of execution. Excessive synchronization can lead to performance degradation, so it is important to use streams judiciously and only when there is sufficient compute/data transfer independence involved.

Let's take a look at a simple example. In pytorch, a cuda stream can be created using [`torch.cuda.Stream`](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html).

```python
import torch

# Functions F1 and F2 are some arbitrary computations
def F1(x, w):
    a = torch.mm(x, w)
    for _ in range(2):
        a = torch.mm(a, a)
    b = torch.sin(a) + torch.cos(a)
    return b

def F2(x, w):
    for _ in range(10):
        x = torch.mm(x, x) + w
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
    x = torch.sum(x)
    return x

# Function G combines the results of F1 and F2
# F1 and F2 perform independent computations and do not depend on each other
def G(x, y, w, stream: torch.cuda.Stream = None):
    if stream is None:
        out1 = F1(x, w)
        out2 = F2(y, w)
        out = out1 + out2
    else:
        stream.wait_stream(torch.cuda.current_stream())
        # Perform F2 in parallel with F1 by using a separate CUDA stream
        with torch.cuda.stream(stream):
            out2 = F2(y, w)
        # Perform F1 in the default stream
        out1 = F1(x, w)
        # We must wait for the side stream to finish computing out2 before we can combine the results
        torch.cuda.current_stream().wait_stream(stream)
        out = out1 + out2
    return out

def benchmark(func, x, y, w, stream: torch.cuda.Stream = None, num_warmup=32, num_repeats=128):
    # Warmup
    for _ in range(num_warmup):
        out = func(x, y, w, stream)

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_repeats)]
    for i in range(num_repeats):
        start_events[i].record()
        func(x, y, w, stream)
        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_time = sum(start.elapsed_time(end) for start, end in zip(start_events, end_events)) / num_repeats
    
    return elapsed_time, out

torch.manual_seed(42)
N = 1024
x = torch.randn((N, N), device="cuda")
y = torch.randn((N, N), device="cuda")
w = torch.randn((N, N), device="cuda")

stream = torch.cuda.Stream()

time1, out1 = benchmark(G, x, y, w, None)
time2, out2 = benchmark(G, x, y, w, stream)

print(f"Time without stream: {time1:.2f} ms")
print(f"Time with stream   : {time2:.2f} ms")
print(f"Outputs are equal: {torch.allclose(out1, out2)}")

# Example output (A100):
# Time without stream: 2.76 ms
# Time with stream   : 2.09 ms
# Outputs are equal: True
```

The above example does not clearly demonstrate how the memory loads and compute operations are overlapped, but it does happen under-the-hood and can be revealed by reviewing the [profile](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) traces.

Inspired by this, we added support for asynchronous offloading in [Diffusers](https://github.com/huggingface/diffusers/blob/3d2f8ae99b88c50a340dd879ec7f44981ffd12fe/src/diffusers/hooks/group_offloading.py#L466), which leverages CUDA streams for prefetching model weights from CPU and overlapping that with the computation of the model. This allows us to benefit from the memory savings of CPU offloading without a large hit to performance. The idea follows from gau-nernst's work [here](https://gist.github.com/gau-nernst/9408e13c32d3c6e7025d92cce6cba140), which was also a starting point for the ideas that follow below.

In the Flux model, the following locations are some opportunities for applying streams (as they either exhibit data/compute independence):
- Computing the adaptive layer norm shift/scale/gate conditioning. They are independent for the double and single transformer blocks, so can be computed independently in different streams.
- Attention QKV projections for image and text stream can also be computed in different streams as the inputs are independent.
- Single stream transformer block linear projection and attention
- Dual stream transformer block normalization and MLP layers

<details>
<summary> Expand for plot </summary>

<table>
<tr>
  <th>A100</th>
  <th>H100</th>
</tr>
<tr>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_3_cuda_stream-a100.png" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_3_cuda_stream-h100.png" /></td>
</tr>
</table>
</details>

As you can probably tell, naively applying streams to any kind of data/compute independence may not always be beneficial. Trying to overlap compute-bound operations with other compute-bound operations, like the QKV projections of image and text stream, is not a good idea. While the example demonstrated here for Flux may not be convincing, the best gains I've seen in production inference is about 1.3-1.4x on some operations with realtime autoregressive video generation models I work on--this requires lots of profiling and making sure you've eliminated most other bottlenecks. Just keep the idea in mind and it may come handy some day.

## Quantization

To understand the ideas behind quantization, we need to first understand how floating point numbers are represented in computers. The [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) standard defines this representation and is widely adopted in modern systems. Let's take a look at the 32-bit floating point representation:

| Sign bit | Exponent (8 bits) | Fraction (23 bits) |
|----------|-------------------|--------------------|
| 0        | 10000001          | 10010000000000000000000 |

The sign bit indicates whether the number is positive or negative, the exponent is used to scale the number, and the fraction represents the significant digits of the number. The range of representable numbers is determined by the exponent, while the precision is determined by the fraction. To calculate the value of this representation, we use the formula:

```
fraction = x0 * 2^(-1) + x1 * 2^(-2) + ... + x22 * 2^(-23)
fraction = 1 * 0.5 + 0 * 0.25 + 0 * 0.125 + 1 * 0.0625 + ... + 0
fraction = 0.5625

exponent = 2^7 + 2^0 = 128 + 1 = 129

# https://en.wikipedia.org/wiki/Exponent_bias
value = (-1)^sign * (1 + fraction) * 2^(exponent - bias)
value = (-1)^0 * (1 + 0.5625) * 2^(129 - 127)
value = 1.5625 * 4 = 6.25
```

where the bias is 127 for 8-bit exponent in fp32. This allows us to represent a wide range of numbers. To quickly test conversions between fp32 and binary representations, we can use the following snippet:

```python
import struct

def float32_to_binary(value):
    packed = struct.pack(">f", value)
    bits = struct.unpack(">I", packed)[0]
    return f"{bits:032b}"

def binary_to_float32(binary):
    bits = int(binary, 2)
    packed = struct.pack(">I", bits)
    return struct.unpack(">f", packed)[0]

print(binary_to_float32("01000000110010000000000000000000"))
```

The maximum and minimum value representable by floating point formats is very important for quantization and will come handy later. To calculate it, we set the sign bit to `1`, the exponent to `11111110`  (we don't set the last bit, which represents nan/infinity), and the fraction to all `1`s. For the maximum value, we do the same but set the sign bit to `0`.
- minimum value: `-3.4e+38`
- maximum value: `3.4e+38`

The float32 format has a very high precision and range which makes it great for stable training of models. However, it is extremely slow to compute any kind of operations with it, especially matrix multiplications - the most common operation in modern deep learning models. A less precise variant known as [TensorFloat (TF32)](https://en.wikipedia.org/wiki/TensorFloat-32) is available on NVIDIA GPUs, which can speedup FP32 matmuls by an order of magnitude by utilizing special hardware called "tensor cores", but lower precision data types like [float-16 (fp16)](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) and [brain-float-16 (bf16)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) are even faster. Nowadays, these formats are more widely adopted in both training and inference settings, especially for large models.

The `bf16` format is probably the most popular data type for LLMs and diffusion models, and is what we are using in this series for Flux, so let's take a look at its representation:

| Sign bit | Exponent (8 bits) | Fraction (7 bits) |
|----------|-------------------|-------------------|
| 0        | 10000001          | 1001000           |

```
fraction = x0 * 2^(-1) + x1 * 2^(-2) + ... + x6 * 2^(-7)
fraction = 1 * 0.5 + 0 * 0.25 + 0 * 0.125 + 1 * 0.0625 + ... + 0
fraction = 0.5625

exponent = 2^7 + 2^0 = 128 + 1 = 129

value = (-1)^sign * (1 + fraction) * 2^(exponent - bias)
value = (-1)^0 * (1 + 0.5625) * 2^(129 - 127)
value = 1.5625 * 4 = 6.25
```

The BF16 format has lower precision than FP32, but it is still sufficient for most deep learning tasks. The range of representable numbers is the same as FP32 but, as the precision is much lower, multiple FP32 numbers map to the same BF16 number. For example, the numbers `6.24`, `6.25`, and `6.26` all map to `6.25`. This is known as "quantization error" and can lead to loss of information. It is usually not very problematic for inference tasks in practice since underlying algorithm implementations are precision-aware most of the time (see [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) or [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance) as an example).

More formally, when we convert from a higher precision format to a lower precision format, we need to handle two main challenges:
- Range mismatch: Lesser exponent bits mean it can only represent a much smaller range of values
- Precision loss: Fewer mantissa bits mean multiple high-precision values map to the same low-precision value

Let's now take a look at even lower precision formats. Why? Smaller bit representations allow us to move data much faster in memory as well as perform any computations faster if the hardware permits it. Most modern hardware have dedicated support for low precision arithmetic, and leveraging this for deep learning inference systems is essential to meet the demands of many production workloads.

There are numerous low precision formats that have been adopted in the deep learning community. Commonly used are:
- FPx - 3-8 bit floating point (additionally, see [OCP microscaling formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf))
- INTx - 2-8 bit integer representation
- GGUF - Quantization used in the GGML ecosystem (see [this](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes) and [this](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9))
- many other algorithmic techniques with custom block/group sizes, numerical representations, scaling strategies, etc.

Each format has its pros/cons depending on the model and usecase but, for the purposes of this article and keeping in mind what's used commonly in generative image production deployments, we will limit our focus to the FP8 dtype for inference.

### fp8 inference

FP8 comes in two main variants defined by the [OCP (Open Compute Project) standard](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1):
- E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits (better for forward pass)
- E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits (better for gradients)

For inference, we typically use E4M3 as it provides better precision for the range of values usually seen in diffusion model forward passes. Let's examine its structure:

| Sign bit | Exponent (4 bits) | Fraction (3 bits) |
|----------|-------------------|-------------------|
| 0        | 1001              | 100               |

```
fraction = x0 * 2^(-1) + x1 * 2^(-2) + x2 * 2^(-3)
fraction = 1 * 0.5 + 0 * 0.25 + 0 * 0.125 = 0.5

exponent = 2^3 + 2^0 = 8 + 1 = 9

value = (-1)^sign * (1 + fraction) * 2^(exponent - bias)
value = (-1)^0 * (1 + 0.5) * 2^(9 - 7)
value = 1.5 * 4 = 6.0
```

As described above about range mismatch and precision loss, we face a more extreme case here. Naively applying FP8 quantization to a model will lead to quality degradation. What can be done to mitigate this?

The solution is to use so-called "scaling factors" that map the range of our input values to the representable range of FP8. There are two simple approaches to computing scaling factors for quantization:
- **Per-tensor scaling**: In this approach, we compute a single scaling factor for the entire tensor. This is the simplest approach:

  ```python
  # Simple absmax scaling:
  #   - dividing the tensor by it's maximum value scales it to [-1, 1] range
  #   - multiplying the tensor by fp8 max value scales it to the FP8 range
  # The quantization operation is typically done by dividing the input tensor by the scale factor.
  # So, we compute the inverse of what we just described: abs_max / fp8_max
  # and this results in the quantized tensor having fp8 range.
  def compute_scale_per_tensor(tensor):
      max_val = tensor.abs().max()
      scale = max_val / torch.finfo(torch.float8_e4m3fn).max
      return scale
  ```

  Example:

  ```python
  tensor = torch.tensor([[-127.589, 45.2], [89.3, -67.152]])
  scale = compute_scale_per_tensor(tensor)  # 0.2848
  quantized = (tensor / scale).clamp(-448, 448).to(torch.float8_e4m3fn)  # [[-448.0, 160.0], [320.0, -240.0]]
  dequantized = quantized.float() * scale  # [[-127.58899688720703, 45.56749725341797], [91.13499450683594, -68.35124969482422]]
  ```

- **Per-row scaling**: In this approach, we compute a separate scale factor for each row of the tensor. This is particularly useful when each row may have a different range of values, such as attention activations. The scaling factor is computed as follows:

  ```python
  # Simple absmax scaling as above, but on a per-row basis
  def compute_scale_per_row(tensor):
      max_vals = tensor.abs().amax(dim=1, keepdim=True)
      scale = max_vals / torch.finfo(torch.float8_e4m3fn).max
      return scale
  ```

  Example:

  ```python
  tensor = torch.tensor([[-127.589, 45.2], [89.3, -67.152]])
  scale = compute_scale_per_row(tensor)  # [[0.2848], [0.1993]]
  quantized = (tensor / scale).clamp(-448, 448).to(torch.float8_e4m3fn)  # [[-448.0, 160.0], [448.0, -352.0]]
  dequantized = quantized.float() * scale  # [[-127.58899688720703, 45.56749725341797], [89.30000305175781, -70.1642837524414]]
  ```

Notice how per-row scaling better utilizes the FP8 range for each row. The second row's values now span a larger portion of the FP8 range, preserving more information. There are various other approaches that have been proposed in literature, such as axis-wise scaling, block-wise scaling, and so on, but these two are the simplest to understand and implement. We will deep-dive into other approaches separately in the future.

Let's now benchmark the performance of these two quantization approaches and understand the quality differences.

### fp8 benchmarks

As matrix multiplcations are at the heart of transformers, optimizing them with fp8 precision leads to a large speedup. This comes with potential loss to quality but it is usually acceptable for diffusion models, and is a tradeoff made by many inference providers.

<div class="scrollable-table">
<table>
<tr>
  <th>A16W16 + A16 attention</th>
  <th>A16W16 + A8 attention</th>
  <th>A8W8 per_tensor + A16 attention</th>
  <th>A8W8 per_tensor + A8 attention</th>
  <th>A8W8 per_row + A16 attention</th>
  <th>A8W8 per_row + A8 attention</th>
</tr>
<tr>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/output---bf16---attention_none.png" height="272px" width="272px" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/output---bf16---attention_unscaled.png" height="272px" width="272px" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/output---fp8_pt---attention_none.png" height="272px" width="272px" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/output---fp8_pt---attention_unscaled.png" height="272px" width="272px" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/output---fp8_pr---attention_none.png" height="272px" width="272px" /></td>
  <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/output---fp8_pr---attention_unscaled.png" height="272px" width="272px" /></td>
</tr>
</table>
</div>
<sup>`Ax` denotes activation precision and `Wx` denotes weight precision (bf16 or fp8).</sup>

The images above compare different precision combinations of activations, weights and attention. For many image models, FP8 attention or lower precision, hurts the quality, and so a common choice for generative image serving is to use FP8 weights/activations and BF16 attention - striking a balance between speed/quality.

How much speedup do we get from fp8 weights and activations though? Since modern hardware has special tensor-core support for low-precision computation and because it's faster to move low-precision data in memory, the gains can be quite significant. The following plot shows the speedup over BF16 when using FP8 matmuls. We compare the [`_scaled_mm`](https://github.com/pytorch/pytorch/blob/9a680e14b74b3d17ea3979518e659196ad037251/aten/src/ATen/native/cuda/Blas.cpp#L1227) implementation available natively in Pytorch against Triton implementations. The code can be found in [`matmul_fp8_benchmark.py`](https://github.com/a-r-r-o-w/productionizing-diffusion/blob/master/post_3/matmul_fp8_benchmark.py).

<details>
<summary> Expand for plot </summary>

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_fp8_matmul.png" alt="Torch/Triton fp8 matmul benchmark against bf16" />

</details>

The pytorch per-tensor and per-row implementations, calling into cuBLASLt kernels, are the fastest. The triton implementations are compiled into efficient GPU programs (Triton IR, Triton GPUIR, LLVM IR, [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/)) from simple user code, and perform quite fast. Although significantly slower than pytorch, the ease of programming in triton, coupled with the continuous performance upgrades by the triton team, make it suitable for developing high performant kernels easily. Pytorch also supports generating scaledmm triton kernels during compilation (see [code](https://github.com/pytorch/pytorch/blob/9a680e14b74b3d17ea3979518e659196ad037251/torch/_inductor/kernel/mm.py#L494)), which can be beneficial if you don't want to spend time writing them yourself.

Note:
- FP8 precision hardware support is only available on the more modern GPU architectures like Ada/Hopper/Blackwell. It would be possible to test fp8 matmuls on 40 and 50 series consumer GPUs, but older hardware like A100 or T4 do not support it.
- There's a Triton conference presentation that shows FP8 matmuls in Triton outperform cuBLAS [here](https://youtu.be/PAsL680eWUw). This definitely seems possible, although I've failed to reproduce the numbers on my end at this time. There could be various reasons for this, but it's not very important for the purpose of this article, so we will not spend much time investigating.

Another potential optimization is the quantization operation itself. For every matmul, we need to compute the quantization scales for the input and weights. Since weights are static tensors, we can precompute them before we use our implementation for inference serving. For inputs, however, the naive implementations with eager pytorch run quite slow. This can be optimized significantly using triton (or compilation). The code can be found in [`quantize_fp8_benchmark.py`](https://github.com/a-r-r-o-w/productionizing-diffusion/blob/master/post_3/quantize_fp8_benchmark.py).

<details>
<summary> Expand for plot </summary>

<table>
<tr>
<td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_fp8_quantize.png" alt="Torch/Triton fp8 quantize benchmark" /></td>
</tr>
</table>

</details>

## To distill or to cache?

If you've played around with large language models, you might have heard of KV caching. This technique caches the key and value projections at each transformer block, so that you don't have to recompute them for every token on each forward pass (see [this post](https://huggingface.co/blog/kv-cache) for a quick refresher). This is particularly useful for autoregressive models, where you sample one token at a time. However, bidirectional attention diffusion models are another beast entirely. They require iteratively computing different QKV projections at each inference step for the entire sequence of tokens, and so the idea of KV caching doesn't apply.

Although not explored in this series, you could cache and re-use the KV projections for models containing cross-attention layers if they only depend on some additional conditioning signal (like text/image/other modality) that do not vary across inference steps. We discuss Wan-T2V later where this is applicable but implementing it is left as an exercise. It should bring a few seconds worth of speedup.

To recap, in diffusion models of the flow matching variety, the model predicts a velocity field at each timestep which is used to gradually move from a source distribution (gaussian noise) to a target distribution (the high quality images and videos of cats we all love).

<table>
<tr>
  <td><video height="272" controls autoplay loop muted><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/steps_double_block_0.mp4">Your browser does not support the video tag.</video></td>
  <td><video height="272" controls autoplay loop muted><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/steps_double_block_18.mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
  <td><video height="272" controls autoplay loop muted><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/steps_single_block_14.mp4">Your browser does not support the video tag.</video></td>
  <td><video height="272" controls autoplay loop muted><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/steps_single_block_37.mp4">Your browser does not support the video tag.</video></td>
</tr>
</table>

<sup> Velocity predictions (left part of video) and denoised samples (right part) visualized at different blocks of Flux. </sup>

Typically, in modern models like Flux and Wan, the number of steps required to generate a good quality sample is ~20 to 30 steps. Each step is a full forward pass through the model, which is quite expensive and takes a lot of time. One way of speeding up the generation process is to "timestep distill" the model. Another approach is to explore the idea of caching in diffusion models.

Timestep distillation is a technique that allows you to sample from a diffusion model with a very low amount of steps (typically 1 to 8), while preserving quality and diversity. We do not explore this technique in this series of post, so we'll not go into the details here. However, you must read [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) and [this blog](https://sander.ai/2024/02/28/paradox.html) by Sander Dieleman if you are interested in this topic.

From an optimization perspective, this technique results in the most significant amount of speedup compared to just system-level optimizations. A rough hand-wavy estimate of the speedup can be found easily: assuming a 4-step distilled model, we can get a 5x speedup compared to a 20-step base model. This is simply because we perform much fewer forward passes through the model. However, distillation requires further training the model and is not a trivial task; not to mention the additional cost associated with compute.

## Caching in diffusion models

From the videos above, you can notice that the velocity predictions and generated samples are similar-ish across different steps while only gradually being refined across the blocks (this can be seen better if you look through the outputs of each block instead of the limited set shown above). This observation is key to developing caching strategies in diffusion models!

Let's try to quantify the amount of difference between the velocity predictions in successive steps. One simple idea is to compute the mean squared error (MSE) of the output at each block:

<details>
<summary> Code </summary>

```python
import itertools

import diffusers
import pandas as pd
import torch
from diffusers.hooks import hooks


# A simple hook to capture the outputs of any pytorch module when attached
class OutputCaptureHook(hooks.ModelHook):
    _is_stateful = False

    def __init__(self):
        super().__init__()
        self.outputs = []

    def post_forward(self, module, output):
        self.outputs.append(output)
        return output


# Uses the Diffusers library hook system to attach a custom hook
def apply_output_capture_hook(transformer: diffusers.FluxTransformer2DModel):
    transformer_blocks = [(f"double_{name}", m) for name, m in transformer.transformer_blocks.named_children()]
    single_transformer_blocks = [(f"single_{name}", m) for name, m in transformer.single_transformer_blocks.named_children()]
    for (name, module) in itertools.chain(transformer_blocks, single_transformer_blocks):
        registry = hooks.HookRegistry.check_if_exists_or_initialize(module)
        hook = OutputCaptureHook()
        registry.register_hook(hook, f"output_capture_hook---{name}")


# Computes the stepwise mean squared difference between the outputs of each transformer block
@torch.inference_mode()
def compute_stepwise_mean_squared_diff(transformer: diffusers.FluxTransformer2DModel):
    transformer_blocks = [(f"double_{name}", m) for name, m in transformer.transformer_blocks.named_children()]
    single_transformer_blocks = [(f"single_{name}", m) for name, m in transformer.single_transformer_blocks.named_children()]
    stepwise_mean_squared_diff = {}

    for (name, module) in itertools.chain(transformer_blocks, single_transformer_blocks):
        registry = hooks.HookRegistry.check_if_exists_or_initialize(module)
        hook = registry.get_hook(f"output_capture_hook---{name}")
        outputs = hook.outputs
        if len(outputs) < 2:
            continue
        stepwise_mean_squared_diff[name] = []
        for i in range(1, len(outputs)):
            hs_i = outputs[i][1]
            hs_i_1 = outputs[i - 1][1]
            diff = hs_i - hs_i_1
            stepwise_mean_squared_diff[name].append(torch.mean(diff ** 2).item())
    
    df = pd.DataFrame(stepwise_mean_squared_diff)
    df.index = [f"Step {i+1}" for i in range(len(outputs) - 1)]
    df.index.name = "Step"
    return df


pipe = diffusers.FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
apply_output_capture_hook(pipe.transformer)

prompt = "A cat holding a sign that says 'Hello, World'"
image = pipe(prompt, height=1024, width=1024, num_inference_steps=28, guidance_scale=4.0).images[0]
image.save("output.png")

compute_stepwise_mean_squared_diff(pipe.transformer).to_csv("dump_stepwise_mse.csv")
```

Code for visualization below avaiable in the repository within the `additional/plots` directory.

</details>

<br />

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/layerwise_mse_plot.png" alt="Stepwise mean squared error" width="1280">

It is visibly apparent that some layers have minimal effect on the output, while others have significant overall changes across the steps. Could we, then, simply skip the layers that have no effect on the output? Not quite. Notice that we're only testing a single prompt here. If we were to try and quantify the effect of each layer across a suite of prompts, we would find different layers to be important for different prompts, and naive discarding of layers would lead to an overall degradation in quality. Don't lose hope yet--we can use this information to develop caching strategies that allow us to skip some layers or inference steps completely, in an adaptive/algorithmic manner, as we shall see below.

### Understanding the U-shape pattern of the MSE

When the MSE between successive steps is low, it indicates that the velocity prediction of inference step `i` is similar to the velocity prediction of inference step `i - 1`.

```python
import math
import numpy as np
import torch

BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15
M = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
B = BASE_SHIFT - M * BASE_IMAGE_SEQ_LEN

device = "cuda"
image_seq_len = 4096
num_inference_steps = 28
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
mu = B + image_seq_len * M
sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** 1.0)
sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
dt = sigmas[1:] - sigmas[:-1]

print([round(x, 4) for x in dt.tolist()])
# -0.0116, -0.0122, -0.0128, -0.0135, -0.0143, -0.0151, -0.016, -0.0169, -0.018, -0.0192, -0.0204, -0.0219, -0.0234, -0.0252, -0.0271, -0.0293, -0.0317, -0.0345, -0.0376, -0.0412, -0.0453, -0.0501, -0.0557, -0.0622, -0.07, -0.0794, -0.0907, -0.1047]
```

Notice how the timestep deltas are very small. If we skipped a full step of velocity prediction and re-used the previous step's prediction for current step, we would overall be moving in the right direction towards the target data distribution but with a potentially noisier trajectory and larger time delta. Arbitrarily skipping steps is not a good idea (for example, skipping step 11 degrades quality), but we can use the MSE information to adaptively skip steps where the difference is smaller than a pre-selected threshold. This is essentially the idea behind two novel caching strategies: [TeaCache](https://huggingface.co/papers/2411.19108) and [First Block Cache](https://github.com/chengzeyi/ParaAttention/blob/4de137c5b96416489f06e43e19f2c14a772e28fd/README.md#first-block-cache-our-dynamic-caching) (a TeaCache-inspired method that makes the implementation much simpler for different model families).

### Overview of different caching techniques

There are several caching techniques that have been proposed specifically for diffusion image and video models. They may extend to other modalities as well, but we will focus on ones that have been adopted by the diffusion community and are easy to implement.

Below is a brief summary of some techniques. An implementation of first block cache with Wan-T2V is provided [here](https://github.com/a-r-r-o-w/productionizing-diffusion).

#### Pyramid Attention Broadcast

Paper: https://huggingface.co/papers/2408.12588

- per-layer stepwise differences are small
- the difference is greater in spatial-attention blocks, followed by temporal-attention blocks, and least in cross-attention blocks
- the difference is typically small after the first few inference steps and before the last few inference steps
- for the range where the difference is small, computing the attention can be skipped. The cached output of previous inference step can be reused instead

The speedup from this method is limited in comparison to latest methods and the quality degradation is also significant and observable easily. We will not be implementing this method in this series, but it is definitely worth checking out the paper to learn more about how caching ideas developed over time.

Note: this method is supported in [Diffusers](https://github.com/huggingface/diffusers/blob/cde02b061b6f13012dfefe76bc8abf5e6ec6d3f3/src/diffusers/hooks/pyramid_attention_broadcast.py#L179) natively for some models.

#### FasterCache

Paper: https://huggingface.co/papers/2410.19355

- finds that naively caching and re-using features often leads to unacceptable quality degradation
- analyzes per-layer stepwise differences for different attention block types and MLPs (for context, we analyzed full transformer blocks above). Finds the same U-shape pattern highlighting feature redundancy
- proposes CFG-cache: finds significant redundancy between features of conditional and unconditional branch of CFG. Proposes a method, which involves separating the low and high frequency components of the latent, to approximate the unconditional branch output using the conditional output

Implementing this method generically to work with all standard transformer-based model implementations is very difficult. We will not be using this method in this series of posts, but it is definitely worth checking out!

Note: this method is supported in [Diffusers](https://github.com/huggingface/diffusers/blob/cde02b061b6f13012dfefe76bc8abf5e6ec6d3f3/src/diffusers/hooks/faster_cache.py#L487) natively for some models.

#### TeaCache & First Block Cache

Paper: https://huggingface.co/papers/2411.19108

- observes that differences among model outputs across timesteps are not uniform, making uniform caching suboptimal for balancing speed and quality
- focuses on model inputs (noisy latents) rather than expensive outputs to estimate these differences, modulating them with timestep embeddings to better approximate what the output variations might seem
- refines the estimates via a rescaling strategy and uses them to adaptively select which timesteps to cache and reuse outputs, effectively skipping less critical inference steps
- training-free approach

First Block Cache is a TeaCache-inspired simplification that eases implementation across model families. This is supported in [Diffusers](https://github.com/huggingface/diffusers/blob/e8e88ff2ce3c8706883f4a90b5b2c023d213625a/src/diffusers/hooks/first_block_cache.py) for most models.
- computes only the residual output of the first transformer block (computationally cheap) and compares its difference (via MSE) to the previous step's output
- if the difference falls below a tunable threshold, reuses the prior full-model residual and skips all subsequent blocks

#### TaylorSeers

Paper: https://huggingface.co/papers/2503.06923

- observes that features in diffusion models evolve slowly and continuously across timesteps, but similarity drops sharply over larger intervals, limiting simple reuse in caching
- shifts from a "cache-then-reuse" to a "cache-then-forecast" paradigm, storing prior features and predicting future ones via Taylor series expansion
- approximates higher-order feature derivatives using a differential method to enable accurate forecasting, bridging gaps in timestep sampling
- applies forecasting selectively for significant timestep jumps

We will not be implementing this method in this series.

#### MagCache

Paper: https://huggingface.co/papers/2506.09045

#### Other caching methods

There are several papers studying caching methods and pushing the frontier in diffusion models. It is not possible to exhaustively cover all of them here, but some notable mentions are:
- [AdaCache](https://huggingface.co/papers/2411.02397)
- [SmoothCache](https://huggingface.co/papers/2411.10510)
- [ToCa](https://huggingface.co/papers/2410.05317)
- [SkipCache](https://huggingface.co/papers/2411.17616)
- [ABCache](https://huggingface.co/papers/2504.10540)
- [BACache](https://huggingface.co/papers/2506.13456)

## Additional reading

- [CUDA Stream by Lei Mao](https://leimao.github.io/blog/CUDA-Stream/)
- [A visual guide to quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [How to optimize a CUDA matmul kernel for cuBLAS-like performance: a worklog](https://siboehm.com/articles/22/CUDA-MMM)

> [!NOTE]
> This post was originally written in July 2025 to be published on the [HF Blog](https://huggingface.co/blog). However, shortly after this, and before the last blog was fully completed, I decided to leave my job at HF and start another chapter of my life elsewhere. This stalled progress on the blogs and I did not have the energy to finish up the series. Originally, walkthroughs of custom GEMM in CUDA, non-trivial fusions, showcasing end-to-end deployment on HF ZeroGPU stack, having accompanying video explanations, and so much more was planned. But due to other important priorities, and the ever-delaying schedule to release this series of posts, I've now decided to cut it short and post just the written posts at this time. My target audience is students–folks finding it difficult to transition from outdated university courses to relevant real-world ML engineering, but I hope all readers may find it useful and valuable!
>
> Many thanks to my team and colleagues at Hugging Face for allowing me the time to work on this. And, of course, for the unlimited H100 GPUs used in testing and debugging things! Lastly, very appreciative of [Linoy Tsaban](https://x.com/linoy_tsaban) for taking the time to read everything I put down and providing valuable feedback <3

## Citation

If you found this post useful in your work, please consider citing it as:

```
@misc{avs2026optdiff,
  author = {Aryan V S},
  title = {Optimizing Diffusion for Production-Ready Speeds},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/a-r-r-o-w/productionizing-diffusion}}
}
```
