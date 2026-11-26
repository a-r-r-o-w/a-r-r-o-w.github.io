import struct
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PHASE_NAMES = [
  "load_sumsq",
  "warp_reduce",
  "block_reduce",
  "barrier",
  "cross_reduce",
  "barrier",
  "normalize",
  "store",
]

PHASE_COLORS = [
  (230, 60, 60),     # load_sumsq - red
  (60, 200, 60),     # warp_reduce - green
  (80, 120, 255),    # block_reduce - blue
  (255, 200, 40),    # barrier - yellow
  (40, 200, 200),    # cross_reduce - cyan
  (255, 200, 40),    # barrier - yellow
  (200, 80, 255),    # normalize - purple
  (255, 140, 60),    # store - orange
]

PALETTE = {i: PHASE_COLORS[i] for i in range(len(PHASE_COLORS))}
EVENT_NAMES = {i: PHASE_NAMES[i] for i in range(len(PHASE_NAMES))}

BG_COLOR = (15, 15, 15)
TEXT_COLOR = (220, 220, 220)
GRID_COLOR = (50, 50, 50)


def load_trace(path):
  with open(path, "rb") as f:
    data = f.read()

  offset = 0
  rows, hidden_dim, nwarps, max_per_warp = struct.unpack_from("IIII", data, offset)
  offset += 16
  total_warps = rows * nwarps
  counts = np.frombuffer(data, dtype=np.uint32, count=total_warps, offset=offset)
  offset += total_warps * 4

  event_dt = np.dtype([("clock", "<u8"), ("meta", "<u4"), ("_pad", "<u4")])
  raw = np.frombuffer(data, dtype=event_dt, count=total_warps * max_per_warp, offset=offset)
  raw = raw.reshape(total_warps, max_per_warp)

  # First two events per warp are headers:
  #   [0] meta=0xFFFFFFFF: clock field = globaltimer_ns (block-wide anchor)
  #   [1] meta=0xFFFFFFFE: clock field = clock64 baseline (block-wide)
  # All subsequent events are clock64 cycle counts.
  # We subtract the global minimum anchor to get small relative values,
  # then compute: relative_time = (anchor - global_min) * cycles_per_ns + (clk64 - baseline)
  # This keeps everything in cycle units with a global offset in cycles.
  # A100 SM clock ~1410 MHz => 1.41 cycles/ns
  cycles_per_ns = 1.41

  # First pass: find global minimum anchor
  global_min_anchor = None
  for w in range(total_warps):
    n = int(counts[w])
    if n >= 2 and int(raw[w, 0]["meta"]) == 0xFFFFFFFF:
      a = int(raw[w, 0]["clock"])
      if global_min_anchor is None or a < global_min_anchor:
        global_min_anchor = a
  if global_min_anchor is None:
    global_min_anchor = 0

  events = []
  for w in range(total_warps):
    n = int(counts[w])
    warp_events = []
    if n < 3:
      events.append(warp_events)
      continue

    meta0 = int(raw[w, 0]["meta"])
    meta1 = int(raw[w, 1]["meta"])
    if meta0 == 0xFFFFFFFF and meta1 == 0xFFFFFFFE:
      anchor_ns = int(raw[w, 0]["clock"])
      baseline_clk = int(raw[w, 1]["clock"])
      anchor_offset_cycles = int((anchor_ns - global_min_anchor) * cycles_per_ns)
      for e in range(2, n):
        cyc = int(raw[w, e]["clock"])
        global_cycles = anchor_offset_cycles + (cyc - baseline_clk)
        warp_events.append(global_cycles)
    else:
      for e in range(n):
        warp_events.append(int(raw[w, e]["clock"]))

    events.append(warp_events)

  return rows, hidden_dim, nwarps, events


def events_to_spans(warp_events):
  spans = []
  for i in range(len(warp_events) - 1):
    t_start = warp_events[i]
    t_end = warp_events[i + 1]
    spans.append((t_start, t_end, i))
  return spans


def load_fonts():
  try:
    return {
      "title": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18),
      "label": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12),
      "small": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11),
      "tiny": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10),
    }
  except (IOError, OSError):
    f = ImageFont.load_default()
    return {"title": f, "label": f, "small": f, "tiny": f}


def draw_legend(draw, fonts, y, x_start):
  x = x_start
  seen = set()
  for ev_id in range(len(PHASE_NAMES)):
    name = PHASE_NAMES[ev_id]
    color = PHASE_COLORS[ev_id]
    key = (name, color)
    if key in seen:
      continue
    seen.add(key)
    draw.rectangle([x, y, x + 14, y + 14], fill=color)
    draw.text((x + 18, y), name, fill=TEXT_COLOR, font=fonts["small"])
    x += 140


def compute_time_range(spans_list):
  all_starts = []
  all_ends = []
  for spans in spans_list:
    for s, e, _ in spans:
      all_starts.append(s)
      all_ends.append(e)
  if not all_starts:
    return 0, 1
  return min(all_starts), max(max(all_ends) - min(all_starts), 1)


def draw_spans_pixel(pixels, spans, y, h, t_min, t_range, margin_left, plot_w, min_px=1):
  for s_start, s_end, ev_id in spans:
    x0 = margin_left + int((s_start - t_min) / t_range * plot_w)
    x1 = margin_left + int((s_end - t_min) / t_range * plot_w)
    x1 = max(x1, x0 + min_px)
    x0 = max(margin_left, min(x0, margin_left + plot_w - 1))
    x1 = max(margin_left, min(x1, margin_left + plot_w))
    color = PALETTE.get(ev_id, (128, 128, 128))
    if x1 > x0:
      pixels[y:y + h, x0:x1] = color


def render_overview(rows, nwarps, events, out_path):
  fonts = load_fonts()

  block_spans = []
  block_mins = []
  block_durations = []
  for block in range(rows):
    spans = []
    for w in range(nwarps):
      spans.extend(events_to_spans(events[block * nwarps + w]))
    block_spans.append(spans)
    if spans:
      bmin = min(s[0] for s in spans)
      bmax = max(s[1] for s in spans)
      block_mins.append(bmin)
      block_durations.append(bmax - bmin)
    else:
      block_mins.append(0)
      block_durations.append(0)

  if not any(block_spans):
    return

  row_h = 6
  margin_left, margin_top, margin_bottom = 80, 50, 60
  width = 2400
  plot_w = width - margin_left - 20
  height = margin_top + rows * row_h + margin_bottom

  img = Image.new("RGB", (width, height), BG_COLOR)
  draw = ImageDraw.Draw(img)
  draw.text((margin_left, 8), "block-level overview", fill=TEXT_COLOR, font=fonts["title"])

  all_flat = [s for bl in block_spans for s in bl]
  t_min = min(s[0] for s in all_flat)
  t_max = max(s[1] for s in all_flat)
  t_range = max(t_max - t_min, 1)

  pixels = np.array(img)
  for block in range(rows):
    spans = block_spans[block]
    if not spans:
      continue
    y = margin_top + block * row_h
    draw_spans_pixel(pixels, spans, y, row_h - 1, t_min, t_range, margin_left, plot_w, min_px=2)

  img = Image.fromarray(pixels)
  draw = ImageDraw.Draw(img)

  for b in range(0, rows, max(rows // 10, 1)):
    draw.text((5, margin_top + b * row_h - 5), f"B{b}", fill=TEXT_COLOR, font=fonts["small"])

  draw_legend(draw, fonts, height - margin_bottom + 10, margin_left)
  img.save(out_path)


def render_detail(rows, nwarps, events, out_path):
  fonts = load_fonts()

  total = rows * nwarps
  all_spans = [events_to_spans(events[i]) for i in range(total)]

  block_mins = []
  block_durations = []
  for block in range(rows):
    bmin, bmax = None, None
    for w in range(nwarps):
      for s, e, _ in all_spans[block * nwarps + w]:
        if bmin is None or s < bmin:
          bmin = s
        if bmax is None or e > bmax:
          bmax = e
    block_mins.append(bmin or 0)
    block_durations.append(max((bmax or 1) - (bmin or 0), 1))

  row_h = max(2, min(4, 4000 // total))
  margin_left, margin_top, margin_bottom = 80, 50, 60
  width = 2400
  plot_w = width - margin_left - 20
  height = margin_top + total * row_h + margin_bottom

  img = Image.new("RGB", (width, height), BG_COLOR)
  draw = ImageDraw.Draw(img)
  draw.text((margin_left, 8), "block-warp detail", fill=TEXT_COLOR, font=fonts["title"])

  all_flat = [s for sp in all_spans for s in sp]
  t_min = min(s[0] for s in all_flat)
  t_max = max(s[1] for s in all_flat)
  t_range = max(t_max - t_min, 1)

  pixels = np.array(img)
  for idx in range(total):
    y = margin_top + idx * row_h
    draw_spans_pixel(pixels, all_spans[idx], y, row_h, t_min, t_range, margin_left, plot_w, min_px=2)

  img = Image.fromarray(pixels)
  draw = ImageDraw.Draw(img)

  step = max(1, rows // 10)
  for b in range(0, rows, step):
    draw.text((5, margin_top + b * nwarps * row_h - 5), f"B{b}", fill=TEXT_COLOR, font=fonts["small"])

  draw_legend(draw, fonts, height - margin_bottom + 10, margin_left)
  img.save(out_path)


def render_warp_overview(rows, nwarps, events, out_path, block=0):
  fonts = load_fonts()

  warp_spans = [events_to_spans(events[block * nwarps + w]) for w in range(nwarps)]
  t_min, t_range = compute_time_range(warp_spans)

  row_h = 28
  margin_left, margin_top, margin_bottom = 80, 50, 60
  width = 2400
  plot_w = width - margin_left - 20
  height = margin_top + nwarps * row_h + margin_bottom

  img = Image.new("RGB", (width, height), BG_COLOR)
  draw = ImageDraw.Draw(img)
  draw.text((margin_left, 8), f"warp overview (block {block})", fill=TEXT_COLOR, font=fonts["title"])

  pixels = np.array(img)
  for w in range(nwarps):
    y = margin_top + w * row_h + 2
    draw_spans_pixel(pixels, warp_spans[w], y, row_h - 4, t_min, t_range, margin_left, plot_w, min_px=2)

  img = Image.fromarray(pixels)
  draw = ImageDraw.Draw(img)

  for w in range(nwarps):
    draw.text((5, margin_top + w * row_h + 6), f"W{w:02d}", fill=TEXT_COLOR, font=fonts["label"])

  draw_legend(draw, fonts, height - margin_bottom + 10, margin_left)
  img.save(out_path)


def render_warp_detail(rows, nwarps, events, out_path, block=0):
  fonts = load_fonts()

  warp_spans = [events_to_spans(events[block * nwarps + w]) for w in range(nwarps)]
  t_min, t_range = compute_time_range(warp_spans)

  row_h = 32
  margin_left, margin_top, margin_bottom = 80, 50, 60
  width = 3200
  plot_w = width - margin_left - 20
  height = margin_top + nwarps * row_h + margin_bottom

  img = Image.new("RGB", (width, height), BG_COLOR)
  draw = ImageDraw.Draw(img)
  draw.text((margin_left, 8), f"warp detail (block {block})", fill=TEXT_COLOR, font=fonts["title"])

  for i in range(11):
    x = margin_left + int(i / 10.0 * plot_w)
    draw.line([(x, margin_top), (x, margin_top + nwarps * row_h)], fill=GRID_COLOR)
    cycles = int(i / 10.0 * t_range)
    draw.text((x - 15, margin_top - 14), str(cycles), fill=TEXT_COLOR, font=fonts["tiny"])

  pixels = np.array(img)
  for w in range(nwarps):
    y = margin_top + w * row_h + 3
    draw_spans_pixel(pixels, warp_spans[w], y, row_h - 6, t_min, t_range, margin_left, plot_w, min_px=2)

  img = Image.fromarray(pixels)
  draw = ImageDraw.Draw(img)

  for w in range(nwarps):
    draw.text((5, margin_top + w * row_h + 7), f"W{w:02d}", fill=TEXT_COLOR, font=fonts["label"])

  draw_legend(draw, fonts, height - margin_bottom + 10, margin_left)
  img.save(out_path)


def main():
  trace_path = sys.argv[1] if len(sys.argv) > 1 else "trace.bin"
  rows, hidden_dim, nwarps, events = load_trace(trace_path)

  render_overview(rows, nwarps, events, "trace_overview.png")
  render_detail(rows, nwarps, events, "trace_detail.png")
  warp_block = min(rows * 3 // 4, rows - 1)
  render_warp_overview(rows, nwarps, events, "trace_warp_overview.png", block=warp_block)
  render_warp_detail(rows, nwarps, events, "trace_warp_detail.png", block=warp_block)


if __name__ == "__main__":
  main()
