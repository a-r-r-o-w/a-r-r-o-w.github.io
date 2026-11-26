from email.utils import format_datetime
from urllib.parse import urlparse
import datetime
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import textwrap

import mistletoe
from mistletoe.html_renderer import HtmlRenderer
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.filter import Filter
from pygments.formatters import HtmlFormatter
from pygments.lexer import RegexLexer, bygroups
from pygments.lexers import CudaLexer, get_lexer_by_name
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Text
from pygments.util import ClassNotFound


SITE_URL = "https://a-r-r-o-w.github.io"

SOCIAL_LINKS_HTML = (
    '<a href="https://github.com/a-r-r-o-w" target="_blank" rel="noopener noreferrer" aria-label="GitHub">'
    '<svg class="icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.57.1.78-.25.78-.55 0-.27-.01-1.17-.02-2.12-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.25.45-2.28 1.18-3.08-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.62 1.58.23 2.75.11 3.04.74.8 1.18 1.83 1.18 3.08 0 4.41-2.69 5.39-5.25 5.67.41.36.78 1.06.78 2.13 0 1.54-.01 2.78-.01 3.16 0 .31.21.66.79.55A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z"/>'
    '</svg></a>\n'
    '<a href="https://x.com/aryanvs_" target="_blank" rel="noopener noreferrer" aria-label="X (Twitter)">'
    '<svg class="icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>'
    '</svg></a>\n'
    '<a href="https://linkedin.com/in/aryan-v-s" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">'
    '<svg class="icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.36V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.26 2.37 4.26 5.45v6.29zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z"/>'
    '</svg></a>\n'
    '<a href="mailto:contact.aryanvs@gmail.com" aria-label="Email">'
    '<svg class="icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M2 5.5A2.5 2.5 0 0 1 4.5 3h15A2.5 2.5 0 0 1 22 5.5v13a2.5 2.5 0 0 1-2.5 2.5h-15A2.5 2.5 0 0 1 2 18.5v-13zm2.3-.5L12 10.7 19.7 5H4.3zM20 7.25l-7.4 5.47a1 1 0 0 1-1.2 0L4 7.25V18.5c0 .28.22.5.5.5h15a.5.5 0 0 0 .5-.5V7.25z"/>'
    '</svg></a>\n'
    '<a href="/feed.xml" aria-label="RSS Feed" title="RSS Feed">'
    '<svg class="icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M6.18 15.64a2.18 2.18 0 1 1 0 4.36 2.18 2.18 0 0 1 0-4.36zM4 4.44A15.56 15.56 0 0 1 19.56 20h-2.83A12.73 12.73 0 0 0 4 7.27V4.44zm0 5.66a9.9 9.9 0 0 1 9.9 9.9h-2.83A7.07 7.07 0 0 0 4 12.93V10.1z"/>'
    '</svg></a>'
)

SVG_GITHUB = '<svg class="icon icon-sm" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.57.1.78-.25.78-.55 0-.27-.01-1.17-.02-2.12-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.25.45-2.28 1.18-3.08-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.62 1.58.23 2.75.11 3.04.74.8 1.18 1.83 1.18 3.08 0 4.41-2.69 5.39-5.25 5.67.41.36.78 1.06.78 2.13 0 1.54-.01 2.78-.01 3.16 0 .31.21.66.79.55A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z"/></svg>'
SVG_PAPER = '<svg class="icon icon-sm" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 2a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6H6zm7 1.5L18.5 9H13V3.5zM8 13h8v1.5H8V13zm0 3.5h8V18H8v-1.5z"/></svg>'
SVG_EXTERNAL = '<svg class="icon icon-sm" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3zM5 5h6v2H7v10h10v-4h2v6H5V5z"/></svg>'

KATEX_HTML = (
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">\n'
    '    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>\n'
    '    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js" '
    'onload="renderMathInElement(document.body, {delimiters: [{left: \'$$\', right: \'$$\', display: true}, '
    '{left: \'$\', right: \'$\', display: false}, {left: \'\\\\(\', right: \'\\\\)\', display: false}, '
    '{left: \'\\\\[\', right: \'\\\\]\', display: true}], throwOnError: false});"></script>'
)


class PtxSassLexer(RegexLexer):
    name = "PTX/SASS"
    aliases = ["ptx", "sass"]

    tokens = {
        "root": [
            (r"//[^\n]*", Comment.Single),
            (r"/\*[\s\S]*?\*/", Comment.Multiline),
            (r'"[^"]*"', String),
            (r"@!?%?[A-Za-z0-9_.]+", Name.Decorator),
            (r"\b[A-Za-z_][A-Za-z0-9_]*:", Name.Label),
            (r"\.(?:version|target|address_size|visible|entry|func|reg|param|shared|local|global|const|align|b8|b16|b32|b64|b128|u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64|bf16|pred|v2|v4|sync|cta|gpu|sys|ca|cg|cs|lu|cv|wb|wt|rn|rz|rm|rp|approx|ftz|sat|hi|lo|wide|cc|nc|relaxed|acquire|release)\b", Keyword.Type),
            (r"\b(?:ld|st|mov|add|sub|mul|mad|fma|div|rem|neg|abs|min|max|and|or|xor|not|shl|shr|selp|setp|bra|ret|exit|call|bar|membar|fence|atom|red|cvt|rcp|sqrt|rsqrt|sin|cos|lg2|ex2|shfl|vote|mma|wgmma|cp|mbarrier|tex|tld4|suld|sust|prmt|bfi|bfe|lop3|LDG|STG|LDS|STS|LDC|LDCU|ULDC|LD|ST|MOV|UMOV|FADD|FMUL|FFMA|DADD|DMUL|DFMA|HADD2|HMUL2|HFMA2|IADD3|UIADD3|IMAD|IMUL|ISETP|FSETP|SEL|BRA|EXIT|RET|BAR|MEMBAR|SHFL|LOP3|LEA|S2R|CS2R|S2UR|CS2UR|R2P|P2R|NOP|WARPSYNC|DEPBAR|HMMA|GMMA|UTCOMMA|UTMABAR|UTMALDG|UTMASTG|LDGSTS)(?:\.[A-Za-z0-9_]+)*\b", Keyword),
            (r"(?:%[a-zA-Z0-9_.]+|\b(?:RZ|PT|URZ|UPT|SR_[A-Z0-9_]+|R[0-9]+(?:\.[0-9]+)?|UR[0-9]+|P[0-9]+|UP[0-9]+|B[0-9]+|SB[0-9]+|c\[0x[0-9a-fA-F]+\]|desc)\b)", Name.Variable),
            (r"0[xX][0-9a-fA-F]+[uUlL]*", Number.Hex),
            (r"-?\b\d+\.\d+(?:[eE][+-]?\d+)?[fF]?\b", Number.Float),
            (r"-?\b\d+[uUlL]*\b", Number.Integer),
            (r"[+\-*/%&|^~!<>?=]+", Operator),
            (r"[\[\]{}(),;:]", Punctuation),
            (r"\s+", Text.Whitespace),
            (r"[A-Za-z_.][A-Za-z0-9_.]*", Name),
            (r".", Text),
        ]
    }


CUDA_C_TYPES = {
    "CUdevice", "CUcontext", "CUmodule", "CUfunction", "CUdeviceptr", "CUresult",
    "CUstream", "CUevent", "CUtexObject", "CUsurfObject", "CUgraph", "CUgraphExec",
    "cudaError_t", "cudaStream_t", "cudaEvent_t", "cudaDeviceProp", "dim3",
    "half", "half2", "__half", "__half2", "__nv_bfloat16", "__nv_bfloat162",
    "float2", "float3", "float4", "int2", "int3", "int4", "uint2", "uint3", "uint4",
    "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "FILE",
}
CUDA_C_BUILTINS = {
    "threadIdx", "blockIdx", "blockDim", "gridDim", "warpSize",
    "__syncthreads", "__syncwarp", "__threadfence", "__threadfence_block",
    "CUDA_SUCCESS", "cudaSuccess", "NULL", "nullptr",
}
CUDA_C_QUALIFIERS = {
    "__global__", "__device__", "__host__", "__shared__", "__constant__",
    "__managed__", "__restrict__", "__forceinline__", "__noinline__", "__launch_bounds__",
}


class CudaCFilter(Filter):
    def filter(self, lexer, stream):
        tokens = list(stream)
        n = len(tokens)
        for i, (ttype, value) in enumerate(tokens):
            if ttype is Name:
                if value in CUDA_C_QUALIFIERS:
                    ttype = Keyword
                elif value in CUDA_C_TYPES:
                    ttype = Keyword.Type
                elif value in CUDA_C_BUILTINS:
                    ttype = Name.Builtin
                else:
                    # Highlight function calls: Name followed by '(' (skipping whitespace)
                    j = i + 1
                    while j < n and tokens[j][0] in Text.Whitespace:
                        j += 1
                    if j < n and tokens[j][0] is Punctuation and tokens[j][1] == "(":
                        ttype = Name.Function
            yield ttype, value


PYGMENTS_FORMATTER = HtmlFormatter(nowrap=True)


def highlight_code(code, lang):
    lang_norm = (lang or "").strip().lower()
    lexer = None
    if lang_norm in ("cu", "cuda"):
        lexer = CudaLexer()
        lexer.add_filter(CudaCFilter())
    elif lang_norm in ("ptx", "sass"):
        lexer = PtxSassLexer()
    elif lang_norm:
        try:
            lexer = get_lexer_by_name(lang_norm)
            if lang_norm in ("c", "h", "cpp", "c++", "cxx", "cc", "hpp"):
                lexer.add_filter(CudaCFilter())
        except ClassNotFound:
            lexer = None
    if lexer is None:
        return html.escape(code)
    return highlight(code, lexer, PYGMENTS_FORMATTER).rstrip("\n")


SVG_DARK_COLOR_MAP = {
    "#ffffff": "#0d0f14",
    "#fff": "#0d0f14",
    "#f3f3f3": "#1d222e",
    "#f2f2f2": "#1d222e",
    "#e5e5e5": "#252b38",
    "#111111": "#f0f4f9",
    "#111": "#f0f4f9",
    "#000000": "#f0f4f9",
    "#000": "#f0f4f9",
    "#444444": "#9aa4b2",
    "#444": "#9aa4b2",
    "#555555": "#74b9ff",
    "#555": "#9aa4b2",
    "#666666": "#9aa4b2",
    "#666": "#9aa4b2",
    "#888888": "#f0b24a",
    "#888": "#f0b24a",
}


def convert_svg_to_dark(svg_text):
    def repl(match):
        raw = match.group(0)
        return SVG_DARK_COLOR_MAP.get(raw.lower(), raw)
    return re.sub(r"#[0-9a-fA-F]{3,6}\b", repl, svg_text)


class Renderer(HtmlRenderer):
    def __init__(self, *extras):
        super().__init__(*extras)
        self.headings = []
        self._id_counts = {}
        self.theme_svgs = set()

    def render_heading(self, token):
        inner = self.render_inner(token)
        plain = self.render_to_plain(token).strip()
        base_id = re.sub(r"[^\w\s-]", "", plain.lower())
        base_id = re.sub(r"[-\s]+", "-", base_id).strip("-") or "section"
        count = self._id_counts.get(base_id, 0)
        self._id_counts[base_id] = count + 1
        heading_id = f"{base_id}-{count}" if count > 0 else base_id
        if token.level in (2, 3):
            self.headings.append((token.level, heading_id, plain))
        anchor = (
            f'<a class="heading-anchor" href="#{heading_id}" '
            f'aria-label="Copy link to {html.escape(plain)}">#</a>'
        )
        return f'<h{token.level} id="{heading_id}">{inner}{anchor}</h{token.level}>'

    def render_raw_text(self, token):
        content = token.content
        url_pattern = r'https?://[^\s<>"\'()]+(?:[^\s<>"\'().,;:!?\]])'
        parts = []
        last_end = 0
        for match in re.finditer(url_pattern, content):
            parts.append(self.escape_html_text(content[last_end : match.start()]))
            url = match.group(0)
            parts.append(f'<a href="{url}">{url}</a>')
            last_end = match.end()
        parts.append(self.escape_html_text(content[last_end:]))
        return "".join(parts)

    def render_link(self, token):
        template = '<a href="{target}"{title}{attrs}>{inner}</a>'
        target = self.escape_url(token.target)
        title = ' title="{}"'.format(html.escape(token.title)) if token.title else ""
        attrs = ' target="_blank" rel="noopener noreferrer"' if target.startswith(("http://", "https://")) else ""
        inner = self.render_inner(token)
        return template.format(target=target, title=title, attrs=attrs, inner=inner)

    def render_image(self, token):
        src = token.src
        alt = self.render_to_plain(token)
        title = ' title="{}"'.format(html.escape(token.title)) if token.title else ""
        if src.endswith("#theme"):
            clean_src = src[: -len("#theme")]
            if clean_src.lower().endswith(".svg"):
                dark_src = clean_src[:-4] + "-dark.svg"
                self.theme_svgs.add(clean_src)
                return (
                    f'<span class="theme-img-pair">'
                    f'<img class="theme-img-light" src="{clean_src}" alt="{alt}"{title} loading="lazy" />'
                    f'<img class="theme-img-dark" src="{dark_src}" alt="{alt}"{title} loading="lazy" />'
                    f'</span>'
                )
            return f'<img src="{clean_src}" alt="{alt}" class="theme-adapt"{title} loading="lazy" />'
        if src.endswith("#theme-dark"):
            clean_src = src[: -len("#theme-dark")]
            return f'<img src="{clean_src}" alt="{alt}" class="theme-adapt-dark"{title} loading="lazy" />'
        return f'<img src="{src}" alt="{alt}"{title} loading="lazy" />'

    def render_table(self, token):
        inner = super().render_table(token)
        return f'<div class="scrollable-table">{inner}</div>'

    def render_block_code(self, token):
        raw_code = token.children[0].content if token.children else ""
        lang = (token.language or "").strip()
        highlighted = highlight_code(raw_code, lang)
        cls = f"highlight language-{html.escape(lang)}" if lang else "highlight"
        return (
            f'<div class="code-block-wrapper">'
            f'<button class="copy-code-btn" onclick="copyCode(this)">Copy</button>'
            f'<pre><code class="{cls}">{highlighted}</code></pre></div>'
        )

    def render_quote(self, token):
        alert_types = {
            "[!NOTE]": ("note", "Note"),
            "[!TIP]": ("tip", "Tip"),
            "[!IMPORTANT]": ("important", "Important"),
            "[!WARNING]": ("warning", "Warning"),
            "[!CAUTION]": ("caution", "Caution"),
        }
        if token.children and hasattr(token.children[0], "children"):
            first_child = token.children[0]
            if first_child.children and hasattr(first_child.children[0], "content"):
                first_text = first_child.children[0].content.strip()
                for marker, (alert_class, label) in alert_types.items():
                    if first_text.startswith(marker):
                        first_child.children[0].content = first_child.children[0].content.replace(marker, "", 1).strip()
                        self._suppress_ptag_stack.append(False)
                        inner = "\n".join([self.render(child) for child in token.children])
                        self._suppress_ptag_stack.pop()
                        return f'<div class="alert alert-{alert_class}"><div class="alert-title">{label}</div>{inner}</div>'
        elements = ["<blockquote>"]
        self._suppress_ptag_stack.append(False)
        elements.extend([self.render(child) for child in token.children])
        self._suppress_ptag_stack.pop()
        elements.append("</blockquote>")
        return "\n".join(elements)


def minify_css(css):
    css = re.sub(r"/\*[\s\S]*?\*/", "", css)
    css = re.sub(r"\s+", " ", css)
    css = re.sub(r"\s*([{}:;,>~+])\s*", r"\1", css)
    css = re.sub(r";\s*}", "}", css)
    return css.strip()


def minify_js(js):
    lines = js.split("\n")
    out = []
    for line in lines:
        stripped = line.rstrip()
        if stripped.lstrip().startswith("//"):
            continue
        out.append(stripped)
    return "\n".join(out)


def short_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def has_math(body):
    stripped = re.sub(r"```[\s\S]*?```", "", body)
    stripped = re.sub(r"`[^`]+`", "", stripped)
    return bool(re.search(r"(\$\$[\s\S]+?\$\$|\$[^\s$][^$\n]*?\$|\\\(|\\\[)", stripped))


def parse_front_matter(content):
    if not content.startswith("---\n"):
        return {}, content
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return {}, content
    fm = json.loads(parts[1].strip())
    return fm, parts[2]


def get_excerpt(body):
    paras = re.split(r"\n\s*\n", body.strip())
    return (paras[0][:150] + "...") if paras else ""


def get_word_count(body):
    text = re.sub(r"```[\s\S]*?```", "", body)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"[#*_\[\]()>-]", " ", text)
    return len(text.split())


def get_read_time(word_count, wpm=200):
    minutes = word_count / wpm
    if minutes < 1:
        return "< 1 min"
    return f"~{round(minutes)} min"


def get_last_modified(filepath):
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci", filepath],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(filepath) or ".",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except Exception:
        pass
    return None


def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]
FONT_MONO = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
]


def generate_og_card(out_path, title, subtitle="", tags=None):
    w, h = 1200, 630
    img = Image.new("RGB", (w, h), (13, 15, 18))
    draw = ImageDraw.Draw(img)

    # Subtle dot grid
    for gx in range(40, w, 40):
        for gy in range(40, h, 40):
            draw.point((gx, gy), fill=(28, 33, 41))

    # Top accent bar
    draw.rectangle([0, 0, w, 6], fill=(116, 185, 255))

    # Decorative particle-life cluster on the right
    palette = [
        (255, 56, 77),
        (26, 217, 255),
        (64, 255, 89),
        (255, 224, 26),
        (224, 71, 255),
        (116, 185, 255),
    ]
    seed = int(hashlib.md5(title.encode("utf-8")).hexdigest()[:8], 16)
    cx, cy = 1045, 315
    for i in range(42):
        angle = (i * 2.39996 + (seed % 100) * 0.1)
        radius = 24 + math.sqrt(i) * 23
        px = int(cx + math.cos(angle) * radius)
        py = int(cy + math.sin(angle) * radius)
        r = 4 + (i % 4) * 2
        col = palette[(i + seed) % len(palette)]
        draw.ellipse([px - r, py - r, px + r, py + r], fill=col)

    font_site = load_font(FONT_MONO, 24)
    font_title = load_font(FONT_BOLD, 54 if len(title) < 42 else 44)
    font_sub = load_font(FONT_MONO, 24)
    font_tag = load_font(FONT_MONO, 20)

    # Header brand
    draw.text((72, 64), "a-r-r-o-w.github.io", fill=(116, 185, 255), font=font_site)

    # Wrapped title
    max_chars = 26 if len(title) < 42 else 32
    lines = textwrap.wrap(title, width=max_chars)[:4]
    y = 165
    line_h = 68 if len(title) < 42 else 56
    for line in lines:
        draw.text((72, y), line, fill=(240, 246, 252), font=font_title)
        y += line_h

    # Subtitle (date / read time / tagline)
    if subtitle:
        sub_y = y + 24
        for sline in textwrap.wrap(subtitle, width=44)[:2]:
            draw.text((72, sub_y), sline, fill=(139, 148, 158), font=font_sub)
            sub_y += 34

    # Tag pills at bottom
    if tags:
        tx = 72
        ty = 528
        for tag in tags[:5]:
            label = tag
            bbox = draw.textbbox((0, 0), label, font=font_tag)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pw = tw + 24
            ph = th + 16
            if tx + pw > 920:
                break
            draw.rounded_rectangle(
                [tx, ty, tx + pw, ty + ph],
                radius=6,
                fill=(22, 32, 44),
                outline=(52, 92, 132),
                width=1,
            )
            draw.text((tx + 12, ty + 7), label, fill=(116, 185, 255), font=font_tag)
            tx += pw + 12

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "PNG", optimize=True)


def collect_posts(src_dir):
    posts = []
    for root, dirs, files in os.walk(src_dir):
        if "README.md" not in files:
            continue
        path = os.path.join(root, "README.md")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        fm, body = parse_front_matter(content)
        if "title" not in fm:
            continue
        orig_rel_path = os.path.relpath(root, src_dir)
        rel_path = orig_rel_path.removeprefix("3_blossom/")
        date_str = fm.get("date")
        date = date_str.split(" - ")[0] if " - " in date_str else date_str
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        word_count = get_word_count(body)
        read_time = get_read_time(word_count)
        last_modified = get_last_modified(path)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", rel_path)
        posts.append(
            {
                "title": fm["title"],
                "url": f"/blog/{rel_path}/index.html",
                "date": date_str,
                "tags": fm.get("tags", []),
                "excerpt": get_excerpt(body),
                "fm": fm,
                "body": body,
                "root": root,
                "rel_path": rel_path,
                "orig_rel_path": orig_rel_path,
                "slug": slug,
                "date_obj": date_obj,
                "word_count": word_count,
                "read_time": read_time,
                "last_modified": last_modified,
            }
        )
    posts.sort(key=lambda p: p["date_obj"], reverse=True)
    return posts


def collect_news(src_file):
    with open(src_file, "r", encoding="utf-8") as f:
        news = json.load(f)
    parsed_news = []
    for date_str, n in news.items():
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        parsed_news.append({"title": n, "date": date_str, "date_obj": date_obj})
    parsed_news.sort(key=lambda p: p["date_obj"], reverse=True)
    return parsed_news


def load_template(template_dir, file_name):
    with open(os.path.join(template_dir, file_name), "r", encoding="utf-8") as f:
        return f.read()


def build_toc_html(headings):
    if len(headings) < 2:
        return "", ""
    items = []
    for level, hid, text in headings:
        cls = "toc-item toc-h3" if level == 3 else "toc-item toc-h2"
        items.append(f'<li class="{cls}"><a class="toc-link" href="#{hid}">{html.escape(text)}</a></li>')
    list_html = '<ul class="toc-list">' + "".join(items) + "</ul>"
    desktop = (
        '<aside class="post-toc" id="post-toc" aria-label="Table of contents">'
        '<div class="toc-title">On this page</div>'
        f"{list_html}</aside>"
    )
    mobile = (
        '<details class="post-toc-mobile">'
        "<summary>Table of contents</summary>"
        f"{list_html}</details>"
    )
    return desktop, mobile


def get_author_info(name, authors):
    entry = authors.get(name)
    if not entry:
        return None
    if isinstance(entry, str):
        return {"url": entry, "affiliation": "", "bio": ""}
    return {
        "url": entry.get("url", ""),
        "affiliation": entry.get("affiliation", ""),
        "bio": entry.get("bio", ""),
    }


def render_author_link(name, authors):
    if name == "Aryan V S":
        return "<strong>Aryan V S</strong>"
    info = get_author_info(name, authors)
    if not info or not info["url"]:
        return html.escape(name)
    url = info["url"]
    host = urlparse(url).netloc.replace("www.", "")
    return (
        f'<a class="coauthor-link" href="{html.escape(url)}" target="_blank" rel="noopener noreferrer" '
        f'data-name="{html.escape(name)}" data-affil="{html.escape(info["affiliation"])}" '
        f'data-host="{html.escape(host)}">'
        f"{html.escape(name)}</a>"
    )


def render_author_list(names, authors):
    return ", ".join(render_author_link(n, authors) for n in names)


def generate_post_html(template, post, authors, css_path, js_path, pl_js_path):
    with Renderer() as r:
        body_html = r.render(mistletoe.Document(post["body"]))
        headings = r.headings
        post["theme_svgs"] = r.theme_svgs

    toc_html, toc_mobile_html = build_toc_html(headings)
    tags_html = " ".join(f'<span class="tag">{t}</span>' for t in post["tags"])

    authors_list = list(post["fm"].get("authors", []))
    for i, author in enumerate(authors_list):
        info = get_author_info(author, authors)
        if not info or not info["url"]:
            continue
        authors_list[i] = render_author_link(author, authors)
    authors_html = (
        f'<div class="post-authors">{" • ".join(authors_list)}</div>' if authors_list else ""
    )

    links = []
    code_url = post["fm"].get("code")
    paper_url = post["fm"].get("paper")
    openreview_url = post["fm"].get("openreview")

    if code_url and str(code_url).strip() and str(code_url).lower() != "todo":
        links.append(f'<a href="{code_url}" class="post-link code">{SVG_GITHUB} Code</a>')
    if paper_url and str(paper_url).strip():
        links.append(f'<a href="{paper_url}" class="post-link paper">{SVG_PAPER} Paper</a>')
    if openreview_url and str(openreview_url).strip():
        links.append(f'<a href="{openreview_url}" class="post-link openreview">{SVG_EXTERNAL} OpenReview</a>')
    links_html = '<div class="post-links">' + " ".join(links) + "</div>" if links else ""

    stats_items = [f'{post["word_count"]:,} words', post["read_time"]]
    if post.get("last_modified"):
        stats_items.append(f'Updated: {post["last_modified"]}')
    post_stats = " • ".join(stats_items)

    excerpt_plain = re.sub(r"<[^>]+>", "", post["excerpt"]).replace('"', "&quot;")
    katex_html = KATEX_HTML if has_math(post["body"]) else ""
    og_image = f'{SITE_URL}/assets/og/{post["slug"]}.png'

    return template.format(
        title=post["title"],
        css_path=css_path,
        date=post["date"],
        authors=authors_html,
        tags_html=tags_html,
        post_links=links_html,
        post_stats=post_stats,
        body_html=body_html,
        js_path=js_path,
        pl_js_path=pl_js_path,
        rel_path=post["rel_path"],
        meta_description=excerpt_plain,
        katex_html=katex_html,
        og_image=og_image,
        toc_html=toc_html,
        toc_mobile_html=toc_mobile_html,
        social_links_html=SOCIAL_LINKS_HTML,
    )


def generate_home_html(template, news, posts, authors, css_path, js_path, pl_js_path):
    combined_news_posts = []
    combined_news_posts.extend(news)
    combined_news_posts.extend(posts[:5])
    combined_news_posts.sort(key=lambda p: p["date_obj"], reverse=True)
    news_html = '<ul class="news-list">'
    for p in combined_news_posts:
        if p.get("url") is not None:
            news_html += f'<li class="news-item"><span class="news-date">{p["date"]}</span><span class="news-text"><a href="{p["url"]}">[Post] {p["title"]}</a></span></li>'
        else:
            news_html += f'<li class="news-item"><span class="news-date">{p["date"]}</span><span class="news-text">{p["title"]}</span></li>'
    news_html += "</ul>"

    pixvid_authors = render_author_list(
        [
            "Yanke Song", "Shuo Chen", "Chris Rockwell", "Guande He", "Nan Liu",
            "Desai Xie", "Tianpei Gu", "Yong-Hyun Park", "Aryan V S",
            "Dennis Yuren Menn", "Ming-Hsuan Yang", "Xun Huang",
        ],
        authors,
    )
    mpeg_authors = render_author_list(
        [
            "Dennis Menn", "Tianpei Gu", "Guande He", "Aryan V S", "Yanke Song",
            "Nan Liu", "Desai Xie", "Xun Huang", "Chenfeng Xu", "Diana Marculescu",
        ],
        authors,
    )

    return template.format(
        css_path=css_path,
        js_path=js_path,
        pl_js_path=pl_js_path,
        news_html=news_html,
        pixvid_authors=pixvid_authors,
        mpeg_authors=mpeg_authors,
        social_links_html=SOCIAL_LINKS_HTML,
    )


def generate_blog_html(template, posts, css_path, js_path, pl_js_path):
    post_list = ""
    for p in posts:
        tags_html = " ".join(f'<button type="button" class="tag tag-btn" data-tag="{t}">{t}</button>' for t in p["tags"])
        post_list += (
            f'<li class="post-item" data-tags="{" ".join(p["tags"])}">'
            f'<div class="post-info">'
            f'<span class="post-date">{p["date"]}</span>'
            f'<div class="post-content">'
            f'<a href="{p["url"]}">{p["title"]}</a>'
            f'<div class="post-tags">{tags_html}</div>'
            f'</div></div></li>\n'
        )
    posts_html = f'<ul class="post-list">{post_list}</ul>\n'
    all_tags = set()
    for p in posts:
        all_tags.update(p["tags"])
    tag_list = "".join(
        f'<li><label><input type="checkbox" class="tag-filter" data-tag="{t}"> {t}</label></li>'
        for t in sorted(all_tags)
    )
    return template.format(
        css_path=css_path,
        cat_html=posts_html,
        tag_list=tag_list,
        js_path=js_path,
        pl_js_path=pl_js_path,
        social_links_html=SOCIAL_LINKS_HTML,
    )


def generate_rss_xml(posts):
    now = format_datetime(datetime.datetime.now(datetime.timezone.utc))
    items = []
    for p in posts:
        link = f'{SITE_URL}/blog/{p["rel_path"]}/'
        guid = f'{SITE_URL}/blog/{p["orig_rel_path"]}/'
        pub_dt = p["date_obj"].replace(tzinfo=datetime.timezone.utc)
        pub_date = format_datetime(pub_dt)
        desc = html.escape(re.sub(r"<[^>]+>", "", p["excerpt"]))
        cats = "".join(f"      <category>{html.escape(t)}</category>\n" for t in p["tags"])
        items.append(
            "    <item>\n"
            f'      <title>{html.escape(p["title"])}</title>\n'
            f"      <link>{link}</link>\n"
            f'      <guid isPermaLink="true">{guid}</guid>\n'
            f"      <pubDate>{pub_date}</pubDate>\n"
            f"      <description>{desc}</description>\n"
            f"{cats}"
            "    </item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        "  <channel>\n"
        "    <title>Aryan V S</title>\n"
        f"    <link>{SITE_URL}/</link>\n"
        "    <description>GPU programming, diffusion models, and systems engineering.</description>\n"
        "    <language>en-us</language>\n"
        f"    <lastBuildDate>{now}</lastBuildDate>\n"
        f'    <atom:link href="{SITE_URL}/feed.xml" rel="self" type="application/rss+xml"/>\n'
        + "\n".join(items)
        + "\n  </channel>\n</rss>\n"
    )


def generate_sitemap_xml(posts):
    urls = [
        f"  <url><loc>{SITE_URL}/</loc></url>",
        f"  <url><loc>{SITE_URL}/posts/</loc></url>",
    ]
    for p in posts:
        lastmod = p.get("last_modified") or p["date_obj"].strftime("%Y-%m-%d")
        urls.append(
            f'  <url><loc>{SITE_URL}/blog/{p["rel_path"]}/</loc><lastmod>{lastmod}</lastmod></url>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls)
        + "\n</urlset>\n"
    )


def main():
    template_dir = "templates"
    src_blog_dir = "src/blog"
    assets_dir = "assets"
    src_news_file = "src/news.json"
    src_authors_file = "src/authors.json"
    out_dir = "_site"
    assets_out_dir = os.path.join(out_dir, "assets")
    og_out_dir = os.path.join(assets_out_dir, "og")
    blog_out_dir = os.path.join(out_dir, "blog")
    posts_out_dir = os.path.join(out_dir, "posts")

    post_template = load_template(template_dir, "post.html.template")
    home_template = load_template(template_dir, "home.html.template")
    blog_template = load_template(template_dir, "blog.html.template")
    not_found_template = load_template(template_dir, "404.html.template")

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(og_out_dir, exist_ok=True)
    os.makedirs(blog_out_dir, exist_ok=True)
    os.makedirs(posts_out_dir, exist_ok=True)
    shutil.copytree(assets_dir, assets_out_dir, dirs_exist_ok=True)
    shutil.copyfile(os.path.join(template_dir, "favicon.svg"), os.path.join(out_dir, "favicon.svg"))

    posts = collect_posts(src_blog_dir)
    post_info_keys = {"title", "url", "date", "tags", "excerpt"}
    with open(os.path.join(out_dir, "posts.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in p.items() if k in post_info_keys} for p in posts], f, indent=2)

    news = collect_news(src_news_file)
    with open(src_authors_file, "r", encoding="utf-8") as f:
        authors = json.load(f)

    css_min = minify_css(load_template(template_dir, "dark.css") + "\n" + load_template(template_dir, "light.css"))
    script_min = minify_js(load_template(template_dir, "script.js"))
    worker_min = minify_js(load_template(template_dir, "particle-worker.js"))
    worker_hash = short_hash(worker_min)
    pl_src = load_template(template_dir, "particle-life.js").replace(
        "/assets/particle-worker.js", f"/assets/particle-worker.js?v={worker_hash}"
    )
    pl_min = minify_js(pl_src)

    with open(os.path.join(assets_out_dir, "style.css"), "w", encoding="utf-8") as f:
        f.write(css_min)
    with open(os.path.join(assets_out_dir, "script.js"), "w", encoding="utf-8") as f:
        f.write(script_min)
    with open(os.path.join(assets_out_dir, "particle-life.js"), "w", encoding="utf-8") as f:
        f.write(pl_min)
    with open(os.path.join(assets_out_dir, "particle-worker.js"), "w", encoding="utf-8") as f:
        f.write(worker_min)

    css_path = f"/assets/style.css?v={short_hash(css_min)}"
    js_path = f"/assets/script.js?v={short_hash(script_min)}"
    pl_js_path = f"/assets/particle-life.js?v={short_hash(pl_min)}"

    # Generate OG cards
    generate_og_card(
        os.path.join(og_out_dir, "home.png"),
        "Aryan V S",
        "GPU programming, diffusion models, and systems engineering",
        ["CUDA", "GPU", "Diffusion", "Systems"],
    )
    generate_og_card(
        os.path.join(og_out_dir, "posts.png"),
        "Posts",
        f"{len(posts)} technical writeups on GPUs, kernels, and ML systems",
        ["CUDA", "PTX", "Diffusion", "Optimization"],
    )
    for post in posts:
        generate_og_card(
            os.path.join(og_out_dir, f'{post["slug"]}.png'),
            post["title"],
            f'{post["date"]}  •  {post["read_time"]}',
            post["tags"],
        )

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(generate_home_html(home_template, news, posts, authors, css_path, js_path, pl_js_path))

    with open(os.path.join(posts_out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(generate_blog_html(blog_template, posts, css_path, js_path, pl_js_path))

    with open(os.path.join(out_dir, "404.html"), "w", encoding="utf-8") as f:
        f.write(
            not_found_template.format(
                css_path=css_path,
                js_path=js_path,
                pl_js_path=pl_js_path,
                social_links_html=SOCIAL_LINKS_HTML,
            )
        )

    with open(os.path.join(out_dir, "feed.xml"), "w", encoding="utf-8") as f:
        f.write(generate_rss_xml(posts))

    with open(os.path.join(out_dir, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(generate_sitemap_xml(posts))

    with open(os.path.join(out_dir, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(f"User-agent: *\nAllow: /\n\nSitemap: {SITE_URL}/sitemap.xml\n")

    with open(os.path.join(blog_out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            '<!DOCTYPE html><html><head><meta charset="UTF-8">'
            '<meta http-equiv="refresh" content="0; url=/posts/">'
            '<link rel="canonical" href="https://a-r-r-o-w.github.io/posts/">'
            '</head><body><a href="/posts/">Posts</a></body></html>'
        )

    for post in posts:
        post_out_dir = os.path.join(out_dir, "blog", post["rel_path"])
        os.makedirs(post_out_dir, exist_ok=True)

        with open(os.path.join(post_out_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(
                generate_post_html(
                    post_template,
                    post,
                    authors,
                    css_path,
                    js_path,
                    pl_js_path,
                )
            )

        theme_svgs = post.get("theme_svgs", set())
        for fname in os.listdir(post["root"]):
            if fname == "README.md":
                continue
            src = os.path.join(post["root"], fname)
            dst = os.path.join(post_out_dir, fname)
            if not os.path.isfile(src):
                continue
            shutil.copyfile(src, dst)
            if fname in theme_svgs and fname.lower().endswith(".svg"):
                dark_name = fname[:-4] + "-dark.svg"
                manual_dark = os.path.join(post["root"], dark_name)
                if not os.path.isfile(manual_dark):
                    with open(src, "r", encoding="utf-8") as sf:
                        dark_svg = convert_svg_to_dark(sf.read())
                    with open(os.path.join(post_out_dir, dark_name), "w", encoding="utf-8") as df:
                        df.write(dark_svg)

        if post["orig_rel_path"] != post["rel_path"]:
            old_out_dir = os.path.join(out_dir, "blog", post["orig_rel_path"])
            os.makedirs(old_out_dir, exist_ok=True)
            new_url = f'/blog/{post["rel_path"]}/'
            canonical = f'{SITE_URL}{new_url}'
            with open(os.path.join(old_out_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write(
                    '<!DOCTYPE html><html><head><meta charset="UTF-8">'
                    f'<meta http-equiv="refresh" content="0; url={new_url}">'
                    f'<link rel="canonical" href="{canonical}">'
                    f'</head><body><a href="{new_url}">{html.escape(post["title"])}</a></body></html>'
                )

    projects_dir = "projects"
    if os.path.isdir(projects_dir):
        for slug in sorted(os.listdir(projects_dir)):
            src_proj = os.path.join(projects_dir, slug)
            if not os.path.isdir(src_proj):
                continue
            dst_proj = os.path.join(out_dir, slug)
            if os.path.exists(dst_proj):
                shutil.rmtree(dst_proj)
            shutil.copytree(src_proj, dst_proj)


if __name__ == "__main__":
    main()
