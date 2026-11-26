import gc
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HIDDEN_DIM = 5120
WARMUP = 10
REPEATS = 50
SHAPES = [(r, HIDDEN_DIM) for r in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]]

DARK_BG = "#0d1117"
DARK_FACE = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#c9d1d9"
COLORS = [
    "#58a6ff",
    "#f778ba",
    "#3fb950",
    "#a371f7",
    "#d29922",
    "#ffa657",
    "#56d4dd",
    "#8b949e",
]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

L2_FLUSH_SIZE_BYTES = 64 * 1024 * 1024


def _get_flush_buffer():
    return torch.empty(L2_FLUSH_SIZE_BYTES // 4, dtype=torch.float32, device="cuda")


def build_seed_list(base_seed, count):
    g = torch.Generator(device="cpu")
    g.manual_seed(base_seed)
    return torch.randint(0, 2**31 - 1, (count,), generator=g).tolist()


def benchmark_cuda(
    fn, input_fn, seeds, warmup=10, repeats=50, sleep_before=0.05, flush_l2=True
):
    flush_buf = _get_flush_buffer() if flush_l2 else None
    for i in range(warmup):
        input_fn(seeds[i])
        fn()
    torch.cuda.synchronize()
    if sleep_before > 0:
        time.sleep(sleep_before)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        if flush_buf is not None:
            flush_buf.zero_()
        input_fn(seeds[warmup + i])
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return float(sum(times) / len(times))


def compute_bandwidth(total_bytes, ms):
    return total_bytes / (ms * 1e-3) / 1e9


def setup_dark_style():
    matplotlib.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_FACE,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.facecolor": DARK_FACE,
            "legend.edgecolor": GRID_COLOR,
            "legend.labelcolor": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.4,
            "font.size": 10,
            "font.family": "monospace",
        }
    )


def plot_series(ax, x_vals, series, title, x_label, y_label):
    for i, (label, data) in enumerate(series.items()):
        ax.plot(
            x_vals,
            data,
            color=COLORS[i % len(COLORS)],
            label=label,
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            linewidth=1.8,
            markersize=5,
        )
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)


def make_single_plot(x_vals, series, x_label, y_label, title, save_path):
    matplotlib.use("Agg")
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_series(ax, x_vals, series, title, x_label, y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fwd_plots(x_vals, results, x_label, title_prefix, dump_dir):
    os.makedirs(dump_dir, exist_ok=True)
    if "lat" in results:
        make_single_plot(
            x_vals,
            results["lat"],
            x_label,
            "Latency (ms)",
            f"{title_prefix} — Latency",
            os.path.join(dump_dir, "latency.png"),
        )
    if "bw" in results:
        make_single_plot(
            x_vals,
            results["bw"],
            x_label,
            "Bandwidth (GB/s)",
            f"{title_prefix} — Bandwidth",
            os.path.join(dump_dir, "bandwidth.png"),
        )


def load_rmsnorm_extension():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return load(
        name="rmsnorm_ext",
        sources=[os.path.join(this_dir, "rmsnorm_ext.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++20",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )


def rmsnorm_bytes(rows, hidden_dim):
    return (rows * hidden_dim * 2 + hidden_dim) * 2


def main():
    ext = load_rmsnorm_extension()
    from liger_kernel.ops.rms_norm import rms_norm_forward as liger_fwd

    def pytorch_fn(x, w):
        return F.rms_norm(x, (x.shape[-1],), w, 1e-6)

    def liger_fn(x, w):
        return liger_fwd(x, w, 1e-6, 0.0, "gemma", "llama")[0]

    def ours_fn(x, w):
        return ext.rmsnorm_forward(x, w, 1e-6)

    backends = {"pytorch": pytorch_fn, "liger-kernel": liger_fn, "ours": ours_fn}
    names = list(backends.keys())
    seeds = build_seed_list(42, WARMUP + REPEATS + 10)

    print("correctness:")
    x_test = torch.randn(128, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    w_test = torch.randn(HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    ref = pytorch_fn(x_test, w_test)
    for name, fn in backends.items():
        if name == "pytorch":
            continue
        out = fn(x_test, w_test)
        diff = (ref.float() - out.float()).abs().max().item()
        status = "OK" if diff < 0.05 else "FAIL"
        print(f"  {name:>14s}  max_diff={diff:.2e}  [{status}]")
    print()

    print(
        f"{'rows':>8s}  {'pytorch':>12s}  {'liger-kernel':>12s}  {'ours':>12s}  (GB/s)"
    )
    print("-" * 60)

    x_labels = []
    results = {"lat": {n: [] for n in names}, "bw": {n: [] for n in names}}

    for rows, hidden_dim in SHAPES:
        x_labels.append(str(rows))
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(hidden_dim, device="cuda", dtype=torch.bfloat16)
        total_bytes = rmsnorm_bytes(rows, hidden_dim)

        def seed_fn(seed, _x=x, _w=w):
            g = torch.Generator(device=_x.device)
            g.manual_seed(seed)
            _x.copy_(
                torch.randn(_x.shape, device=_x.device, dtype=_x.dtype, generator=g)
            )

        bws = []
        for name in names:
            fn = backends[name]
            run_fn = lambda _fn=fn, _x=x, _w=w: _fn(_x, _w)
            lat_ms = benchmark_cuda(
                run_fn, seed_fn, seeds, warmup=WARMUP, repeats=REPEATS
            )
            bw = compute_bandwidth(total_bytes, lat_ms)
            results["lat"][name].append(lat_ms)
            results["bw"][name].append(bw)
            bws.append(bw)

        print(f"{rows:>8d}  {bws[0]:>12.0f}  {bws[1]:>12.0f}  {bws[2]:>12.0f}")

        del x, w
        torch.cuda.empty_cache()
        gc.collect()

    dump_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    save_fwd_plots(
        x_labels, results, "Rows (batch*seq)", "RMSNorm (hidden=5120, bf16)", dump_dir
    )


if __name__ == "__main__":
    main()
