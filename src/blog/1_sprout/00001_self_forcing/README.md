---
{
  "title": "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion",
  "paper": "https://arxiv.org/abs/2506.08009",
  "authors": ["Xun Huang", "Zhengqi Li", "Guande He", "Mingyuan Zhou", "Eli Shechtman"],
  "code": "https://github.com/guandeh17/Self-Forcing",
  "date": "2025-09-02",
  "tags": ["computer-vision", "diffusion", "autoregression", "world-models"]
}
---

- introduces novel paradigm for autogressive diffusion models
- addresses exposure bias (models trained on ground truth context must generate sequences conditioned on their own imperfect predictions at inference time)
- design of Diffusion Transformers (DiTs) till date denoise all frames simultaneously using bidirectional attention (future frames can affect past frames, and entire video must be generated at once) - authors argue that this is fundamentally limiting their applicability for real-time applications
- autoregressive models may help, but usually struggle to match sota video model performance
- teacher forcing recap: predict next token conditioned on ground-truth tokens. in context of video diffusion, TF trains model to predict each frame conditioned on previous context frames
- self forcing: instead of conditioning on ground-truth frames, model conditions on its own previous predictions, allowing it to learn to generate more realistic frames over time
  $$p(\hat{x}^1)p(\hat{x}^2|x^1)p(\hat{x}^3|x^1,x^2)...p(\hat{x}^T|x^{<T}) = p(\hat{x}^1, \hat{x}^2, ..., \hat{x}^{T-1})$$
- diffusion forcing: trains model on videos with noise levels independently sampled for each frame (denoising each frame based on past noisy context frames) - ensures autoregressive inference scenario where context frames are clean and current frame is noisy
  $$p(\hat{x}^1)p(\hat{x}^2|x_{t^1}^1)p(\hat{x}^3|x_{t^1}^1, x_{t^2}^2)...p(\hat{x}^T|x_{t^{<T}}^{<T}) \neq p(\hat{x}^1, \hat{x}^2, ..., \hat{x}^{T-1})$$
- introduces "self forcing": addresses exposure bias. inspired by RNN-era sequence modeling techniques to bridge train-test distribution gap by explicitly unrolling autoregressive generation during training.
  - each frame is conditioned on previously self-generated frames rather than ground truth frames
  - supervision with distribution-matching losses
  $$p(\hat{x}^1)p(\hat{x}^2|\hat{x}^1)p(\hat{x}^3|\hat{x}^1,\hat{x}^2)...p(\hat{x}^T|\hat{x}^{<T}) = p(\hat{x}^1, \hat{x}^2, ..., \hat{x}^{T-1})$$

todo reading and re-reading:
- [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867)
- [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)
- [Score Identity Distillation](https://arxiv.org/abs/2404.04057)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

reading list for exposure bias - some approaches attempt to mitigate distributional mismatch by incorporating noisy context frames during inference:
- [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392)
- [Oasis: A Universe in a Transformer](https://oasis-model.github.io/)
- [Packing Input Frame Context in Next-Frame Prediction Models for Video Generation](https://arxiv.org/abs/2504.12626)

reading list for distribution-matching losses:
- [Long-Context Autoregressive Video Modeling with Next-Frame Prediction](https://arxiv.org/abs/2503.19325)
- [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867)
- [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://arxiv.org/abs/2412.07772)
