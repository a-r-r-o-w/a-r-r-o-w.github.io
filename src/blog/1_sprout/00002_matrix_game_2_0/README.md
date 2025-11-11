---
{
  "title": "Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model",
  "paper": "https://arxiv.org/abs/2508.13009",
  "authors": ["Xianglong He", "Chunli Peng", "Zexiang Liu", "Boyang Wang", "Yifan Zhang", "Qi Cui", "Fei Kang", "Biao Jiang", "Mengyin An", "Yangyang Ren", "Baixin Xu", "Hao-Xiang Guo", "Kaixiong Gong", "Cyrus Wu", "Wei Li", "Xuchen Song", "Yang Liu", "Eric Li", "Yahui Zhou"],
  "code": "https://github.com/SkyworkAI/Matrix-Game/",
  "date": "2025-10-03",
  "tags": ["computer-vision", "diffusion", "autoregression", "world-models"]
}
---

Challenges of existing methods for interactive video generation:
- lack of large-scale interactive video datasets with rich annotations for training (accuracte actions, camera dynamics, etc. due to high cost of collection)
- generating a single frame requires full bidirectional diffusion model
- unsuitable for real-time, streaming applications where model must adapt to dynamic user commands and produce frames on the fly
- computationally intensive and economically impractical -- quadratic scaling of compute and memory wrt. frame length + large number of denoising iterations
- severe error accumulation in existing autoregressive diffusion models

Questions:
- action injection module enables frame-level mouse and keyboard inputs as interactive conditions -- assuming a standard 4x temporal VAE, is this frame-level at latents or pixel-space?
- continuous mouse actions are directly concatenated to the input latent representations (pass through MLP + temporal self attention alyer), but keyboard actions are queried by fused features through a cross-attention layer. how are these design choices made?
- as the oldest KV cache tokens are evicted when exceeding capacity, this model should not be able to maintain a consistent world. it's great that infinite generation is possible, but consistency and history of the world really matter in a real game. what architectural and algorithmic design choices would have to be made to make that possible?

generalization
real-time long video generation (25 FPS+, minute-level)
long term consistency (memory and anti-drift)

Sometimes datasets only have camera trajectory data
- if user is climbing a mountain, camera trajectory is moving up and forward (slanting), but keyboard will be just moving forward. how do we create annotated keyboard and mouse actions from camera trajectory data?

todo reading list:
- GameFactory: Creating New Games with Generative Interactive Video
- CameraCtrol II: Dynamic Scene Exploration via Camera-controlled Video Diffusion Models
- DeepVerse: 4D Autoregressive Video Generation as a World Model
- CausVid: From Slow Bidrectional to Fast Autoregressive Video Diffusion Models
- Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
- Self Forcing+: Towards Minute-Scale High-Quality Video Generation
- Video World Models with Long-term Spatial Memory
- Context as Memory: Scene-Consistent Interactive Long Video Generation
- Mixture of Contexts for Long Video Generation
- One-Minute Video Generation with Test-Time Training
