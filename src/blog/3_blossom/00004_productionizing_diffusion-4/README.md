---
{
  "title": "Optimizing diffusion for production-ready speeds - IV",
  "code": "https://github.com/a-r-r-o-w/productionizing-diffusion",
  "date": "2026-01-29",
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

The code for the entire series is available at [a-r-r-o-w/productionizing-diffusion](https://github.com/a-r-r-o-w/productionizing-diffusion). For this post, refer to the `post_4` directory. The guides are written to work on Nvidia's A100/H100 or newer GPUs, but the ideas can be adapted to other hardware as well.

- [LoRA](#lora)
- [Training](#training)
- [Wan text-to-video](#wan-text-to-video)
- [Uncommon deployment environments](#uncommon-deployment-environments)
- [Lower level languages](#lower-level-languages)
- [Triton kernels](#triton)
  - [Pointwise Add and SiLU](#pointwise-add-and-silu)
  - [Fused Adaptive LayerNorm Zero Dual](#fused-adaptive-layernorm-zero-dual)
  - [Fused Adaptive LayerNorm Zero Single](#fused-adaptive-layernorm-zero-single)
  - [RoPE](#rope)
- [Others Ideas](#others-ideas)
- [Additional reading](#additional-reading)

## LoRA

[LoRA](https://arxiv.org/abs/2106.09685), short for Low-Rank Adaptation, offers a parameter-efficient approach to finetuning models by targeting only a small subset of new parameters. At its core, it decomposes weight updates into low-rank matrices injected into the bigger linear layers, enabling the model to adapt to new tasks while preserving the pretrained knowledge. During training, this method optimizes just the low-rank adapters, typically comprising mere millions of parameters against billions in the base model, which dramatically reduces memory footprint and accelerates the tuning process (even on constrained hardware).

In a standard linear layer, the forward pass is `y = W.x + b`, where `W` is the weight matrix (say, `d x d`) and `b` is the bias. LoRA approximates the update `ΔW` as `B.A`, with `A` shaped `d x r` (tall and skinny) and `B` as `r x d` (short and wide), where `r << d` is the pre-selected low rank like `32`. This keeps the update tiny: just `2 * d * r` parameters instead of `d * d`. The LoRA output `y_lora = B.(A.x)` is added to the base output `y`. Typically, `y_lora` is scaled by a factor `α / r` to stabilize training, where `α` is a hyperparameter often set equal to `r`. For more references on the inner workings of LoRA, please refer to the "Additional Reading" section.

In the context of diffusion model lora training, loras are typically applied to the attention and MLP layers. This enables personalization of the model to specific styles, subjects, or concepts, with minimal data, and without needing to retrain the entire model. The power of this technique has resulted in widespread adoption in the community, with thousands of LoRA models available on Hugging Face and other model hubs. The logical question, in the context of production inference, is how to efficiently serve these LoRA models.

Here's a dummy implementation of a linear layer with LoRA:

```python
class LinearLoRATorch(torch.nn.Module):
    def __init__(
        self, in_features, out_features, r: int = 32, alpha: float = 32.0, scaling_factor: float | None = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)
        self.register_buffer("scaling_factor", torch.tensor(alpha / r if scaling_factor is None else scaling_factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling_factor
        return linear_out + lora_out

# Checkout HF guides to gain more insights about implementation, training and usage.
# PEFT: https://huggingface.co/docs/peft/developer_guides/lora
# Diffusers LoRA: https://huggingface.co/docs/diffusers/training/lora
```

For production-ready implementations, this typically ends up becoming much more complex - to support multiple LoRAs or faster kernels - but for now, let's simplify and assume that production systems need to cater to these use cases:
1. Serving a base model without any LoRA
2. Serving a base model with a single LoRA. LoRA may change per request (hot swapping).
3. Serving a base model with multiple LoRAs. LoRAs may change per request (hot swapping).

Cases 1 and 2 are straightforward - you can apply all the optimizations from this series for good speedups. Case 3 is trickier and deserves its own deep dive later. Running the linear + LoRA math fast is key, but so are fast training pipelines and dynamic hot-swapping to load/unload adapters on the fly. For now, we will focus on fast inference.

Note: My colleagues at HF have written a nice blog on LoRA optimization [here](https://huggingface.co/blog/lora-fast) - readers should definitely check it out! That covers a lot, but there's more to explore with LoRA-specific kernel tweaks. I originally planned to dive into inference/training optimizations and hot-swapping here, but the topic's breadth (and the kernel authoring prerequisites) convinced me to spin it off into a standalone article.

To give a taste of improved kernel-level gains, consider this optimized variant inspired by community favorites like [Unsloth](https://github.com/unslothai/unsloth). It calls directly into cuBLAS and performs in-place operations (for buffer reuse), which can even outperform `torch.compile` generated Triton - so you see, `torch.compile` isn't always the end-all-be-all solution for speedups! Benchmark everything yourself!

```python
class LinearLoRAUnsloth(torch.nn.Module):
    def __init__(
        self, in_features, out_features, r: int = 32, alpha: float = 32.0, scaling_factor: float | None = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)
        self.register_buffer("scaling_factor", torch.tensor(alpha / r if scaling_factor is None else scaling_factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = x.view(-1, self.in_features)
        out = torch.addmm(self.linear.bias, x, self.linear.weight.t())
        xa = torch.mm(x, self.lora_A.weight.t())
        out.addmm_(xa, self.lora_B.weight.t(), beta=1.0, alpha=self.scaling_factor)
        return out.view(*x_shape[:-1], self.out_features)
```

With some very simple changes, the speedups here are quite significant. This is just the tip of the iceberg though. With more complex kernels, or even basic ideas like utilizing a CUDA stream for computation of the lora output branch, much more speedups are possible. The following benchmark compares a few implementations and showcases that. All the code is available in `post_4/lora.py`.

<details>
<summary> Expand for benchmarking results </summary>

<table>
<tr>
  <th align="center"> 4090 </th>
  <th align="center"> H100 </th>
</tr>
<tr>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/linearlora_seqlen_4090.png" /></td>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/linearlora_seqlen_h100.png" /></td>
</tr>
</table>

</details>

## Training

Although we're mainly interested in inference optimizations with this series, it is worth noting that training optimizations can also be applied to inference.

Training a diffusion model isn’t all that different from inference at its core - you’re still running forward passes, just with added backward passes to update weights. But when it comes to optimizing training for speed, there are a few tricks that can make a big difference.

- Using mixed precision. Training with base model weights in `bf16` and lora weights in `fp32` is common practice for finetuning diffusion models. Some more speedups can be squeeze by keeping lora weights in `bf16` as well. Personal experiments show little to no quality degradation for personalization tasks, but results may vary.
- Slow I/O can bottleneck your training. Use data loaders with multiple workers and pin memory to keep the GPU fed constantly. Preprocess datasets offline if possible - resize images, precompute embeddings, or cache tokenized text to avoid repeated work each epoch, or each time you restart training. Make sure data prefetching works well.
- Avoid activation checkpointing unless you absolutely need it. If using an 80 GB A100/H100, you can often squeeze in sequence lengths upto 16k tokens.
- Context parallelism can speed up large sequence length training. Attempting to maximally overlap communication and computation can yield further speedups.
- [Compiled autograd](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html) and optimizers can speed up backward passes significantly.
- Rewriting common operations as `torch.autograd.Function`'s using in-place operations and fused kernels.
- Low-bit optimizers (like 8-bit Adam) can reduce memory footprint and speed up training.
- Low-bit (fp8) forward pass, but bf16 backward pass using custom autograd functions.
- CUDAgraph'ed training loops.
- Precomputing embeddings that do not change often, or don't take too many unique values (e.g. time embeddings, pooled text embeddings, attention masks, etc.)

Some of these ideas are lossless while others are lossy. Be sure to validate your training to ensure quality is not compromised. An example for training [Flux-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) is available in `post_4/flux_train.py`. Some simple benchmarks:

```python
# A100
baseline: 2.49 it/s
compiled autograd: 2.88 it/s
compiled forward + compiled autograd: 3.20 it/s
```

## Wan text-to-video

Over the course of the last few articles, we learnt about some of the optimizations to make Flux run extremely fast on single GPU and multi-GPU setups. We can now apply the same series of optimizations on Wan, a text-to-video and image-to-video diffusion model, or more generally, on most models that use a transformer-based architecture. To recap, the optimizations we will apply are:
- `torch.compile`
- Some custom torch inductor/dynamo flags that result in slightly faster speedup
- Fused QKV for self-attention layers and fused KV for cross-attention layers
- CUDAGraphs
- Modeling rewrites (e.g. `torch.addcmul`) and in-place operations
- Flash attention 2 (on A100) and Flash attention 3 (on H100)
- Context parallelism

Additionally, we will explore a different caching technique in this article compared to previous article - First Block Cache. This technique is simple to implement and very effective in practice. We could combine in other ideas like FP8 or lower precision dtypes with caching, but since both are lossy optimizations, combining them can lead to severe quality degradation, so for this article, we will limit our testing to caching.

Wan is capable of generating 5 second videos with resolutions upto 720P at 16 FPS (frames per second)<sup>*</sup> of very high quality and realistic motion. The model is transformer-based architecture, as is Flux, but instead of pixel-unshuffling an image to obtain a sequence of image tokens, a learned 3D convolution is used for patchification followed by flattening the spatiotemporal volume of tokens into a sequence of video tokens. This result is passed through a stack of transformer blocks followed by output norm, projection and unpatchification to produce the final latent video. The inner dimension of transformer blocks of 14B T2V model is `5120` and MLP hidden dimension of `13824`, which is considerable larger than Flux's `3072` and `12288`, but it has fewer overall layers (`40` vs `19 + 38 = 57`). As the model is larger, it is expected to be slightly slower. But, the model weights and parameters are not the only contributors to inference speed of a model!

Video models require more compute than image models due to an order of magnitude more tokens being processed. In our case, we will be evaluating inference speed for a 5-second 480OP video at 16 FPS. The following calculation shows the number of tokens that will be processed by the model:

```
((duration * fps) / vae_temporal_downsample_factor * patch_size_temporal) * ((height * width) / (vae_spatial_downsample_factor * patch_size_spatial)**2) + text_tokens
  => ((5 * 16) / (4 * 1)) * ((480 * 832) / (8 * 2)**2) + 512
  => 20 * 1560 + 512
  => 31200 video tokens + 512 text tokens
  => 31712 tokens
```

This is significantly more tokens being processed per denoising forward pass than Flux, which had only `4096` image tokens + `512` text tokens for a 1024px image! More tokens and larger internal model dimensions mean more compute requirements, and thus more time taken for inference. The following graphs show the latency to generate videos with Wan on A100 and H100 GPUs respectively.

<table>
<tr>
  <th align="center"> A100 </th>
  <th align="center"> H100 </th>
</tr>
<tr>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_4_wan-t2v-14b_a100.png" /></td>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/benchmark_post_4_wan-t2v-14b_h100.png" /></td>
</tr>
</table>

<sub><sup></sup>16 FPS should not be confused with the FPS at which a video is saved. Here, it means a total of 81 frames being generated and stored at 16 FPS, resulting in a 5 second video.</sub>

Congratulations, Wan runs super fast now! This is the speed at which most inference providers serve their models, but can we do better? Ofcourse!

### Sparse Attention

Sparse attention is a rapidly evolving area of research that aims to reduce the computational cost of the attention mechanism in transformer models. Instead of computing attention scores for all pairs of tokens, sparse attention focuses on a subset of tokens. Imagine a scenario where you have a very large number of N tokens, and you could simply drop half of them, or even more, while still maintaining the quality of the output -- that's sparse attention for you!

Studying activation patterns of attention layers in the models allows us to find these subsets of tokens that are most relevant to the generation process at each layer. In video models, utilizing this sparsity has shown significant promise for reducing computational overhead while maintaining generation quality, particularly due to excessive temporal and spatial redundancies (pixels between consecutive frames do not differ much and generally have high spatial similarity as well). See papers [1](https://arxiv.org/abs/2506.19852), [2](https://arxiv.org/abs/2505.13389) and [3](https://arxiv.org/abs/2502.18137) for more details.

Let's try to think of ways we could discover some sparse attention patterns ourselves. Because we know consecutive frames do not differ much and that spatially adjacent pixels are similar, we could try to drop every alternate spatial token (say, in a checkerboard pattern):
- For latent frame 0, keep tokens at indices `[(0, 0), (0, 2), (0, 4), ..., (1, 1), (1, 3), (1, 5), ...]`
- For latent frame 1, keep tokens at indices `[(0, 1), (0, 3), (0, 5), ..., (1, 0), (1, 2), (1, 4), ...]`
- ...similarly for other frames

For the attention shapes and matmuls to work out correctly, we can only drop tokens for the key and value projections. With the naive approach described above, we enjoy a 4x reduction of KV tokens per attention layer. However, as this is very naive, the quality degradation is high.

What else could be done? This is exactly what [Video Sparse Attention](https://arxiv.org/abs/2505.13389) examines in their ablation studies. For the sake of simplicity, we do not explore all the possible approaches here. For preserving quality with fewer tokens, typically we might have to post-train the model with sparse attention patterns. This is a topic for another day, so for now, we will skip that discussion.

Instead, we will quickly look into [Radial Attention](https://arxiv.org/abs/2506.19852), and examine the inference speedups we can achieve with it. This approach cleverly taps into how attention "energy" in video models fades exponentially with spatial and temporal distance - think of it as focusing compute on nearby frames and pixels where it matters most, while skimping on the far away tokens that barely, or don't, contribute useful information in the generation process. By crafting a sparse pattern with decaying density bands (halving attention scores as distances double temporally, and shrinking spatial diagonals accordingly), it reduces attention computational complexity from `O(N^2)` to `O(N log N)` using a static mask - this is a massive speedup. It's a neat fit for models like Wan, enabling trimming down on token count without noticeable quality hits.

That said, we will continue talking about these methods in-depth, along with implementation, in a future standalone post because the current density of this post is already high, and my intention here was to just put the general ideas for speeding up inference out there. Feel free to experiment on your own, and stay tuned for more info on this!

## Uncommon Deployment Environments

So far, we've looked at how we can optimize models to run fast. For deploying these models, you'd typically take an optimized model implementation and deploy it over multi-GPU nodes, behind a load balancer and web server, and expose an API endpoint to your users. For the purposes of this blog, we're only concerned about deploying a model over a single/multi-GPU replica. Before a replica is available to serve users, the following things need to be taken care of:
- Loading the model weights onto the GPUs.
- Warming up the model—PyTorch's forward passes can be slower on the first run due to CUDA initialization, and custom kernels often need autotuning to pick the best configurations.
- Just-in-time (JIT) compilation, if you're using tools like `torch.compile` to optimize the graph on the fly.
- Handling any initial setup for distributed communication, like initializing process groups.
- Precomputing static elements, such as embeddings, that don't change between requests or are limited in variety.

These steps work great in traditional setups where your servers are always running, and GPUs are dedicated. But what if you're dealing with more dynamic, shared environments? Enter Hugging Face's ZeroGPU Spaces. ZeroGPU is a clever shared infrastructure that lets you use powerful NVIDIA H200. It's perfect for demos or hobby projects or experimental tools where you don't want to pay for constant GPU uptime. You just add a simple `@spaces.GPU` decorator to your functions, and the system forks a process, grabs a GPU slice, runs your task, and releases the hardware right after.

The catch? In these serverless setups, processes are short-lived to maximize sharing. That means cold starts are common, and anything that adds overhead, like JIT compilation with `torch.compile`, may result in extremely long wait times for users.

That's where Ahead-of-Time (AoT) compilation comes in. Instead of compiling on the fly, you export and compile your model once upfront, producing a reusable shared library that loads instantly. PyTorch supports this through tools like `AOTInductor`, which processes models, optimizes them, and generates artifacts you can reload without the compilation hit. Read the PyTorch [docs](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details.

Note: I planned to dive deep into AoT here with Wan, but my colleagues already nailed it in their blog post: [Make your ZeroGPU Spaces go brrr with ahead-of-time compilation](https://huggingface.co/blog/zerogpu-aoti). If you're optimizing for ZeroGPU, that should be your go-to guide.

But AoT isn't the end of the road for uncommon environments. When you're pushing the limits in shared or constrained setups, you might need to drop down a level and write custom kernels. Take as example the speedups obtained from Flash Attention 3.

## Lower level languages

When you feel PyTorch or `torch.compile` have hit their limits, you can drop down to lower-level languages like [CUDA](https://docs.nvidia.com/cuda/) for more control over GPUs. CUDA is powerful but tricky to write and has a large learning curve. That's where domain-specific languages (DSLs) come in. [Triton](https://openai.com/index/triton/), from OpenAI, feels like Python but compiles to fast GPU programs, often beating handtuned CUDA with less headache. [CuTE DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), from NVIDIA [CUTLASS](https://docs.nvidia.com/cutlass/index.html), is great for writing complex logic at a level between CUDA and Triton and gives you finer control. Other languages like [TileLang](https://github.com/tile-ai/tilelang) or [Mojo](https://www.modular.com/mojo) lower the barrier for entry, but are still maturing in terms of ecosystem and adoption.

## Triton kernels

Triton makes it easy to write fused kernels that can stay on par with, or sometimes outperform, optimized cuBLAS/cuDNN implementations (those called by PyTorch under the hood). It is worth noting that triton kernels are fully compatible with pytorch subsystems like `torch.compile`, so you can use them in conjunction and benefit from all the goodness of compilation. Following is a quick look at some operations for which we can write custom Triton kernels to speed up eager mode execution.

### Pointwise Add and SiLU

```python
def pointwise_add3_silu(time_embeds: torch.Tensor, guidance_embeds: torch.Tensor, pooled_text_embeds: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(time_embeds + guidance_embeds + pooled_text_embeds)
```

This operation is used as the input the adaptive layernorm layers in the transformer blocks. It is simply pointwise addition of three tensors followed by a SiLU activation. The following kernel implements this operation:

```python
# Mark this function as triton program
@triton.jit
def pointwise_add3_silu_kernel(x_ptr, y_ptr, z_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each triton kernel is compiled and launched with a grid of programs.
    # Each program handles a small chunk of the full data.
    pid = tl.program_id(0)
    # Compute the starting index of the block this program should handle; prepare offsets; prepare mask
    # to handle out-of-bounds accesses.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load inputs from memory
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    # Perform computation
    res = x + y + z
    res = res / (1 + tl.exp(-res))  # SiLU activation
    # Store result back to memory
    tl.store(out_ptr + offsets, res, mask=mask)

# Python wrapper to call the kernel
def pointwise_add3_silu_triton(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape == z.shape
    N = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    pointwise_add3_silu_kernel[grid](x, y, z, out, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=1)
    return out
```

`torch.compile` also generates a similar kernel, but writing explicit kernels allows you to benefit from speedups in eager mode as well, without waiting for a long compilation step.

### Fused Adaptive LayerNorm Zero Dual

```python
class AdaLayerNormZeroEager(torch.nn.Module):
    def __init__(self, embedding_dim: int, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x)
        x = torch.addcmul(shift_msa, x, 1 + scale_msa)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
```

This operation is used to calculate the modulation conditioning to facilitate interactions between image/text tokens and time/guidance/pooled text embeddings. The kernel implementations, along with benchmarking code, is available in `post_4/fused_adaln_zero_triton.py`.

<details>
<summary> Expand for benchmarking results </summary>

<table>
<tr>
  <th align="center"> A100 </th>
  <th align="center"> H100 </th>
</tr>
<tr>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/fused_adaln_zero_triton_a100.png" /></td>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/fused_adaln_zero_triton_h100.png" /></td>
</tr>
</table>

<sub>Note: Please make sure to read the note at the top of the file to understand how to fairly benchmark against `torch.compile` and why our numbers here show it to be slower than it actually is.</sub>

</details>

### Fused Adaptive LayerNorm Zero Single

```python
class AdaLayerNormZeroSingleEager(torch.nn.Module):
    def __init__(self, embedding_dim: int, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)
        x = self.norm(x)
        x = torch.addcmul(shift_msa, x, 1 + scale_msa)
        return x, gate_msa
```

This operation is used to calculate the modulation conditioning to facilitate interactions between image/text tokens and time/guidance/pooled text embeddings. The kernel implementations, along with benchmarking code, is available in `post_4/fused_adaln_zero_single_triton.py`. The only difference between this and the dual version is that this version only computes modulation for attention layers, while the dual version computes modulation for both attention and MLP layers.

<details>
<summary> Expand for benchmarking results </summary>

<table>
<tr>
  <th align="center"> A100 </th>
  <th align="center"> H100 </th>
</tr>
<tr>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/fused_adaln_zero_single_a100.png" /></td>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/fused_adaln_zero_single_h100.png" /></td>
</tr>
</table>

<sub>Note: Please make sure to read the note at the top of the file to understand how to fairly benchmark against `torch.compile` and why our numbers here show it to be slightly slower than it actually is.</sub>

</details>

### RoPE

```python
def apply_rotary_pos_emb_torch(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = q.dtype
    q_real, q_imag = q.unflatten(-1, (-1, 2)).unbind(-1)
    k_real, k_imag = k.unflatten(-1, (-1, 2)).unbind(-1)
    q_rotated = torch.stack([-q_imag, q_real], dim=-1).flatten(3)
    k_rotated = torch.stack([-k_imag, k_real], dim=-1).flatten(3)
    q_out = q * cos + q_rotated * sin
    k_out = k * cos + k_rotated * sin
    return q_out.to(dtype), k_out.to(dtype)
```

[RoPE](https://arxiv.org/abs/2104.09864) is an essential component of the modern transformer architecture. It is used to inject positional information into the query and key projections of attention layers. The operation involves elementwise multiplications and additions, while also treating pairs of elements as complex numbers to perform rotations.

RoPE is generally implemented in two ways:
- Half the head dimension is treated as real part and the other half as imaginary part: `[R1, R2, ..., I1, I2, ...]`
- Each pair of values in the head dimension is treated as a complex number: `[R1, I1, R2, I2, ...]`

Although the first approach is more common, models like Flux and Wan use the second approach. `torch.compile` does not generate the most efficient kernel for this operation, so we can write some custom triton code to speed it up. The kernel implementations, along with benchmarking code, is available in `post_4/rope_2.py`.

<details>
<summary> Expand for benchmarking results </summary>

<table>
<tr>
  <th align="center"> A100 </th>
  <th align="center"> H100 </th>
</tr>
<tr>
  <th align="center" colspan="2"> batch sizes </th>
</tr>
<tr>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/rope_fwd_bs_a100.png" /></td>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/rope_fwd_bs_h100.png" /></td>
</tr>
<tr>
  <th align="center" colspan="2"> sequence lengths </th>
</tr>
<tr>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/rope_fwd_seq_a100.png" /></td>
  <td align="center"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F555/blog/productionizing-diffusion/rope_fwd_seq_h100.png" /></td>
</tr>
</table>

</details>

## Others Ideas

There's a lot of other cool fusions or techniques that can be applied for further speedups:
- KV caching for cross attention layers: notice how the outputs of key and value projections remain the same across inference steps -- this means we could compute them once to shave off a few milliseconds.
- RMSNorm + RoPE for both Q and K projections
- Replacing all convolutions with `kernel_size == stride` with linear layers
- LayerNorm + Modulate/Gate + (optionally) Quantize
- Preallocating and reusing output buffers across layers

This is only scratching the surface of what can be done. Another cool idea is to use [megakernels](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles). This is something I plan to write more about in a future post.

## Additional reading

- [HF LLM-Course LoRA](https://huggingface.co/learn/llm-course/en/chapter11/4)
- [Practical Tips for Finetuning LLMs using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [What is LoRA? by AI Coffee Break with Letitia](https://www.youtube.com/watch?v=KEv-F5UkhxU)
- [Radial Attention](https://arxiv.org/abs/2506.19852)
- [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)

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
