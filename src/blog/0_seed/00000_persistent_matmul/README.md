---
{
  "title": "Persistent Matmul",
  "authors": ["Aryan V S"],
  "code": "https://gist.github.com/a-r-r-o-w/28339b442d164084506c0967029968a8",
  "date": "2025-08-05",
  "tags": ["deep-learning", "matmul"]
}
---

You would've implemented the 3-loop matrix multiplication many times as a ML practitioner, but the naive implementation is terrible for GPU performance. Modern GPUs achieve peak performance through careful memory access patterns and minimizing scheduling overhead.

In naive matmul (MxK . KxN), the computation happens in tiles - both for the output matrix and for how you read chunks from the input matrices. Each thread-block processes one output tile by loading corresponding tiles from input (for sum-reduction across K dimension), performing the computation, then terminating. The GPU launches many thread-blocks and schedules them across available streaming multiprocessors (SMs). When an SM finishes one tile, it gets assigned a new thread-block for the next uncomputed tile. This way, multiple output tiles are computed in parallel across the SMs, but we pay the cost for launching thread-blocks each time a new tile is computed.

Persistent matmul changes this approach. Instead of launching thread-blocks to compute some output tiles, computing the results on SMs in parallel, and repeating until all output tiles are computed, you launch only as many thread-blocks as you have SMs available (typically 80-132 on modern GPUs). These thread-blocks stay alive until all output tiles are computed, looping through multiple tiles sequentially. Each persistent thread-block may handle multiple output tiles.

The key benefit is the reduced thread-block launch latency. This persistence strategy, combined with other optimizations like coalesced memory loads/stores, block-tiling, warp-tiling, warp-specialization, double-buffering, ping-pong scheduling and other tricks, helps achieve peak performance. More on this in the future!

Code snippet for testing: https://gist.github.com/a-r-r-o-w/28339b442d164084506c0967029968a8
