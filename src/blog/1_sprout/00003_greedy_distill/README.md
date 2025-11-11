---
{
  "title": "Greedy Distill: Efficient Video Generative Modeling With Linear Time Complexity",
  "paper": "https://openreview.net/forum?id=51pcTCVQ90",
  "authors": ["TODO", "ICLR 2026 Conference Submission5458 Authors"],
  "code": "TODO",
  "date": "2025-11-04",
  "tags": ["computer-vision", "diffusion", "autoregression", "distillation", "world-models"]
}
---

Iterative sampling process, large denoising networks -> prohibitively high computational requirements for practical deployment

For $T$ sampling steps, $F$ frames, and per-frame feature length $L$, total complexity is $T \times F^2 \times L^2$.

Most existing models optimize the sampling steps but overlook other expensive factors. Authors note that optimization along the frame dimension, which contributes, $O(f^2)$ complexity, remains underexplored.

Recent approaches like CausVid employs asymmetric structural distillation strategy - bidirectional-attention teacher diffusion model and causal student model--mainly reduces cost by fewer sampling steps.

Authors find that attention to first frame and neighbouring frames is substantially higher than to distant ones across multiple foundation video-generation models--Wan, Hunyuan Video and CogVideoX.
